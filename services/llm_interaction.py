# services/llm_interaction.py
"""
LLM API 交互模块（重构版）

提供核心的AI调用接口，已提取以下功能到独立模块：
- JSON修复 → utils.json_repair
- 消息处理 → services.llm.message_processor
- 重试策略 → services.llm.retry_strategy
"""

import json
import logging
import re
import time
import types
from collections.abc import Mapping
from typing import Any, Union, cast, get_args, get_origin

import openai

# 先定义 logger
logger = logging.getLogger(__name__)


def _categorize_error(e: Exception) -> str:
    """分类错误类型以确定重试策略

    Returns:
        str: 错误类别 - 'instructor_retry', 'rate_limit', 'network', 'validation', 'unknown'
    """
    error_str = str(e).lower()
    error_type_name = type(e).__name__

    # InstructorRetryException
    if InstructorRetryException is not None and isinstance(e, InstructorRetryException):
        return "instructor_retry"

    # 速率限制错误
    if any(keyword in error_str for keyword in ["rate limit", "429", "too many requests", "quota exceeded"]):
        return "rate_limit"
    if hasattr(e, "status_code") and getattr(e, "status_code", None) == 429:
        return "rate_limit"

    # 网络错误
    if any(keyword in error_str for keyword in ["timeout", "connection", "network", "ssl", "certificate"]):
        return "network"
    if "RemoteProtocolError" in error_type_name or "ConnectError" in error_type_name:
        return "network"

    # 验证错误
    if "ValidationError" in error_type_name or "validation" in error_str:
        return "validation"

    # 默认
    return "unknown"


# 导入重构后的模块
from services.llm.message_processor import (  # noqa: E402
    clean_text_artifacts,
    coerce_message_content,
    ensure_json_instruction,
)
from services.llm.retry_strategy import (  # noqa: E402
    EmptyResponseFromReasonerError,
    build_retry_exception_types,
    build_retryer,
)
from utils.json_repair import (  # noqa: E402
    massage_structured_payload,
    repair_json_once,
)

# 尝试导入可选依赖
try:
    import instructor
    from instructor.exceptions import InstructorRetryException
    from pydantic import BaseModel as PydanticBaseModel

    instructor_available = True
except ImportError:  # pragma: no cover - optional dependency
    instructor_available = False
    instructor = None  # type: ignore
    InstructorRetryException = None  # type: ignore
    PydanticBaseModel = None  # type: ignore

# 从重构后的模块中导入依赖
from config import Config  # noqa: E402
from config.constants import ModelLimits  # noqa: E402
from utils.progress_tracker import get_tracker  # noqa: E402


def _default_frequency_penalty(config: Config) -> float:
    if hasattr(config, "generation") and getattr(config.generation, "frequency_penalty", None) is not None:
        return config.generation.frequency_penalty
    return getattr(config, "frequency_penalty", 0.0)


def _default_presence_penalty(config: Config) -> float:
    if hasattr(config, "generation") and getattr(config.generation, "presence_penalty", None) is not None:
        return config.generation.presence_penalty
    return getattr(config, "presence_penalty", 0.0)


def _ensure_sync_client(config: Config) -> openai.OpenAI:
    """
    延迟初始化并返回同步的DeepSeek客户端。
    统一的辅助函数，避免重复的回退逻辑。
    """
    client_obj = config.client
    if client_obj is None:
        logging.info("同步客户端未初始化，正在即时初始化...")
        config.initialize_deepseek_client()
        client_obj = config.client
    if client_obj is None:
        raise RuntimeError("DeepSeek 客户端未能初始化。")
    return client_obj


def _build_chat_call_params(
    model_name: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
) -> tuple[dict[str, Any], bool]:
    """
    准备聊天补全调用参数，同时处理推理器特定的约束。
    返回(<call_params>, <is_reasoner_model>)。
    """
    is_reasoner_model = "reasoner" in model_name.lower()
    call_params: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if not is_reasoner_model:
        call_params.update({
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        })
    return call_params, is_reasoner_model


# 以下函数已移至 services.llm.message_processor 和 utils.json_repair
# 此处保留别名以保持向后兼容
_ensure_json_instruction = ensure_json_instruction
_coerce_message_content = coerce_message_content
_clean_text_artifacts = clean_text_artifacts


# 以下JSON处理函数已移至 utils/json_repair，此处保留别名
from utils.json_repair import (  # noqa: E402
    _safe_model_validate,  # 内部使用
)

_massage_structured_payload = massage_structured_payload  # 保持向后兼容


# repair_json_once 现在从 utils.json_repair 导入，支持debug参数
def _repair_json_once_compat(text: str, schema: type[Any]) -> tuple[str, bool]:
    """兼容性包装，调用新的repair_json_once"""
    # 尝试从config获取debug设置
    try:
        from config.config import Config

        debug = getattr(Config().workflow, "debug_json_repair", False)
    except Exception:
        debug = False
    return repair_json_once(text, schema, debug=debug)


# 这个块标记用于下一步删除
def call_ai_with_schema(
    config: Config,
    model_name: str,
    messages: list[dict[str, Any]],
    schema: type[Any],
    kwargs: dict[str, Any],
) -> tuple[Any | str, str]:
    """使用 Pydantic schema 进行结构化调用的增强版本。"""
    if not instructor_available:
        logging.warning("Instructor不可用，回退到普通调用")
        content = call_ai(
            config,
            model_name,
            messages,
            temperature=kwargs.get("temperature"),
            max_tokens_output=kwargs.get("max_tokens_output", -1),
            top_p=kwargs.get("top_p"),
            frequency_penalty=kwargs.get("frequency_penalty"),
            presence_penalty=kwargs.get("presence_penalty"),
            response_format=kwargs.get("response_format"),
            schema=kwargs.get("schema"),
        )
        return content, "instructor_unavailable"

    if PydanticBaseModel is not None:
        try:
            if not issubclass(schema, PydanticBaseModel):
                logging.warning(
                    "提供的 schema %s 不是 Pydantic BaseModel 子类，可能导致结构化调用失败。",
                    schema,
                )
        except TypeError:
            logging.warning("提供的 schema %s 无法用于 issubclass 检查。", schema)

    try:
        # 使用 instructor.patch 创建结构化客户端
        # 注意：我们禁用了 instructor 的内部重试（max_retries=0），因为：
        # 1. instructor 重试时会修改 messages（添加响应历史）
        # 2. 这些修改绕过了 _coerce_message_content 的规范化
        # 3. 外层的 tenacity 重试会重新调用本函数，确保消息始终被规范化
        client = _ensure_sync_client(config)

        if instructor is None:
            raise RuntimeError("结构化调用失败：instructor 库不可用。")

        structured_client = cast(Any, instructor).patch(client)

        # 准备调用参数
        normalized_messages = _coerce_message_content(messages)
        if logging.getLogger(__name__).isEnabledFor(logging.DEBUG):
            for idx, msg in enumerate(normalized_messages):
                content = msg.get("content")
                if not isinstance(content, str):
                    logging.debug(
                        "Message %s content type after coercion is %s (expected str)",
                        idx,
                        type(content).__name__,
                    )
                residual_keys = {"tool_calls", "function_call"} & msg.keys()
                if residual_keys:
                    logging.debug(
                        "Message %s retains structured fields post-coercion: %s",
                        idx,
                        {key: msg[key] for key in residual_keys},
                    )

        call_params: dict[str, Any] = {
            "model": model_name,
            "messages": normalized_messages,
            "response_model": cast(Any, schema),
        }

        # 添加其他参数，处理参数名映射
        for k, v in kwargs.items():
            if k == "response_format":
                continue

            # 🔧 参数名映射：max_tokens_output → max_tokens
            if k == "max_tokens_output":
                # 映射到 max_tokens，但不添加 max_tokens_output 本身
                if isinstance(v, int) and v > 0 and "max_tokens" not in call_params:
                    call_params["max_tokens"] = v
                continue  # 确保 max_tokens_output 不被传递到 API

            # 正常添加其他参数
            call_params[k] = v

        # 确保 max_tokens_output 不会被意外传递（安全检查）
        call_params.pop("max_tokens_output", None)

        # 禁用 instructor 内部重试：instructor 在重试时会将包含 tool_calls 的响应
        # 追加到 messages，绕过我们的 _coerce_message_content 规范化，导致 API 400 错误。
        # 外层的 tenacity 重试机制会在异常时重新调用整个函数，确保每次都重新规范化消息。
        call_params["max_retries"] = 0

        response = structured_client.chat.completions.create(**call_params)
        logging.info(f"结构化调用成功，返回 {type(response).__name__} 类型对象")
        return response, "success"

    except Exception as e:
        error_msg = f"结构化调用失败: {e}"

        # 增强的错误分类和处理
        error_category = _categorize_error(e)

        # 检查是否是可修复的ValidationError
        is_repairable_error = False
        from pydantic import ValidationError

        # 第1步：从异常中提取 ValidationError（可能被 InstructorRetryException 包装）
        validation_error: ValidationError | None = None

        if isinstance(e, ValidationError):
            validation_error = e
            logging.debug("捕获到直接的 ValidationError")
        elif InstructorRetryException is not None and isinstance(e, InstructorRetryException):
            # InstructorRetryException 可能包装了 ValidationError
            # 检查多个可能的属性
            logging.debug("检测到 InstructorRetryException，尝试提取 ValidationError...")
            logging.debug("  - 可用属性: %s", dir(e))

            if hasattr(e, "__cause__") and isinstance(e.__cause__, ValidationError):
                validation_error = e.__cause__
                logging.debug("✓ 从 InstructorRetryException.__cause__ 提取到 ValidationError")
            elif hasattr(e, "last_exception") and isinstance(getattr(e, "last_exception", None), ValidationError):
                validation_error = e.last_exception
                logging.debug("✓ 从 InstructorRetryException.last_exception 提取到 ValidationError")
            elif hasattr(e, "exception") and isinstance(getattr(e, "exception", None), ValidationError):
                validation_error = e.exception
                logging.debug("✓ 从 InstructorRetryException.exception 提取到 ValidationError")
            else:
                # 尝试遍历异常链
                logging.debug("  - 尝试遍历异常链...")
                current = e
                max_depth = 5
                depth = 0
                while current and depth < max_depth:
                    depth += 1
                    if isinstance(current, ValidationError):
                        validation_error = current
                        logging.debug("✓ 从异常链深度 %d 提取到 ValidationError", depth)
                        break
                    current = getattr(current, "__cause__", None)

                if not validation_error:
                    # 尝试更多提取方法
                    if hasattr(e, "args") and len(e.args) > 0:
                        for arg in e.args:
                            if isinstance(arg, ValidationError):
                                validation_error = arg
                                logging.debug("✓ 从 InstructorRetryException.args 提取到 ValidationError")
                                break

                    if not validation_error:
                        logging.warning("✗ 无法从 InstructorRetryException 提取 ValidationError（已检查所有可能位置）")

        # 第2步：如果找到了 ValidationError，尝试提取并修复损坏的 JSON
        rescue_candidate: str | None = None

        if validation_error:
            error_str = str(validation_error)
            # 扩展可修复错误的判断条件
            repairable_keywords = ["trailing characters", "json_invalid", "unterminated string", "expecting", "invalid escape", "unexpected character", "json.decoder.JSONDecodeError", "validation error"]
            if any(keyword in error_str.lower() for keyword in repairable_keywords):
                is_repairable_error = True
                # 暂时不输冺ERROR，先尝试修复
                logging.debug("检测到可修复的验证错误（%s），将尝试使用修复工具链", "InstructorRetryException包装" if isinstance(e, type(InstructorRetryException)) else "直接ValidationError")
            else:
                # 其他验证错误仍然输出但降低日志级别
                logging.warning(error_msg)  # 从 error 改为 warning

            # 从 ValidationError 中提取原始 JSON 文本
            try:
                errors_list = validation_error.errors()
                logging.debug("ValidationError.errors() 返回 %d 个错误项", len(errors_list))

                for idx, err_item in enumerate(errors_list):
                    logging.debug("  - 错误 %d: type=%s, keys=%s", idx, err_item.get("type"), list(err_item.keys()))

                    # 尝试多种提取方法
                    raw_input = None
                    if "input" in err_item:
                        raw_input = err_item["input"]
                        logging.debug("    从 err_item['input'] 提取")
                    elif "ctx" in err_item and isinstance(err_item["ctx"], dict):
                        raw_input = err_item["ctx"].get("input")
                        logging.debug("    从 err_item['ctx']['input'] 提取")

                    if isinstance(raw_input, str) and raw_input.strip():
                        rescue_candidate = raw_input
                        logging.debug("✓ 从 ValidationError 提取到原始 JSON（长度=%s，前80字符=%s...）", len(raw_input), raw_input[:80].replace("\n", " "))
                        break
                    elif raw_input is not None:
                        logging.debug("    raw_input 存在但类型不匹配: %s", type(raw_input).__name__)

                if not rescue_candidate:
                    # 最后尝试：从错误消息中提取 input_value
                    logging.debug("  - 尝试从错误字符串中提取 input_value...")
                    if "input_value=" in error_str:
                        match = re.search(r"input_value='([^']+)'", error_str)
                        if match:
                            rescue_candidate = match.group(1)
                            # 处理转义
                            rescue_candidate = rescue_candidate.replace(r"\'", "'").replace(r"\"", '"')
                            logging.debug("✓ 从错误消息提取到 JSON 片段（长度=%s）", len(rescue_candidate))
            except Exception as err_extract:
                logging.debug("无法从 ValidationError 中提取原始输入: %s", err_extract, exc_info=True)

            # 第3步：调用完整的修复工具链（包括 json-repair 库、LaTeX 处理等）
            if rescue_candidate:
                logging.debug("启动修复工具链：json-repair库 + LaTeX处理 + 7种内置策略")
                try:
                    # 获取 debug 配置，启用详细的修复日志
                    debug_mode = getattr(getattr(config, "workflow", None), "debug_json_repair", False)
                    repaired_text, repaired = repair_json_once(rescue_candidate, schema, debug=debug_mode)
                    candidate_text = repaired_text if repaired else rescue_candidate
                    candidate_text = _clean_text_artifacts(candidate_text)

                    data = json.loads(candidate_text)
                    parsed = _safe_model_validate(schema, _massage_structured_payload(schema, data))

                    if is_repairable_error:
                        logging.info("✓ 结构化调用失败——已通过修复工具链恢复（%s）", "json-repair" if repaired else "内置策略")
                    else:
                        logging.info("结构化调用失败——通过修复工具链恢复")
                    return parsed, "fallback_success"
                except json.JSONDecodeError as rescue_parse_error:
                    logging.debug("修复工具链：JSON 解析失败: %s", rescue_parse_error)
                except Exception as rescue_validate_error:
                    logging.debug("修复工具链：模型校验失败: %s", rescue_validate_error)
        else:
            # 非 ValidationError，正常输出错误
            logging.error(error_msg)

        # 记录详细的调试信息以便排查
        if "BadRequestError" in type(e).__name__ or (hasattr(e, "status_code") and getattr(e, "status_code", None) == 400):
            logging.debug("BadRequestError 详情 - 可能的消息格式问题。请检查 messages 是否包含未序列化的复杂对象（如 tool_calls、function_call）。")

        # 第4步：修复失败后，回退到普通调用（不使用 json_object）
        logging.debug("修复工具链未能恢复结构化输出，回退到普通调用（不使用 json_object）")

        # 保存错误标记到函数作用域外
        _is_repairable_error = is_repairable_error
        plain_kwargs = {k: v for k, v in kwargs.items() if k != "response_format"}
        plain_kwargs["schema"] = None
        if "max_tokens" in plain_kwargs:
            max_tokens_value = plain_kwargs.pop("max_tokens")
            plain_kwargs["max_tokens_output"] = max_tokens_value

        json_retry_kwargs = plain_kwargs.copy()
        parsed_obj: Any | None = None
        force_plaintext = False
        structured_retry_count = 0
        max_structured_retry = 3  # 增加重试次数

        # 根据错误类别决定重试策略
        if error_category == "instructor_retry":
            # InstructorRetryException的智能处理
            if rescue_candidate:
                # 已经尝试过修复，但可能需要改变策略
                if structured_retry_count < max_structured_retry:
                    logging.info("正在尝试第 %d/%d 次结构化修复...", structured_retry_count + 1, max_structured_retry)
                    # 继续尝试结构化重试，但使用不同的温度参数
                    json_retry_kwargs["temperature"] = min(0.7, json_retry_kwargs.get("temperature", 0.3) + 0.1)
                else:
                    logging.warning("InstructorRetryException：达到最大重试次数，回退到纯文本模式。")
                    force_plaintext = True
            elif validation_error:
                # 找到了ValidationError但无法提取JSON，可能需要直接重试
                if structured_retry_count < 1:  # 至少试一次
                    logging.info("检测到ValidationError，尝试直接重新请求...")
                    # 不设置force_plaintext，继续结构化重试
                else:
                    logging.warning("检测到 InstructorRetryException 且找到 ValidationError，但无法提取有效的 JSON input，回退到纯文本檀式。")
                    force_plaintext = True
            else:
                # 既没有rescue_candidate也没有validation_error，可能是其他问题
                logging.warning("检测到 InstructorRetryException 但无法确定具体问题，回退到纯文本模式。")
                force_plaintext = True
        elif error_category == "rate_limit":
            # 速率限制，应该等待并重试
            import random
            import time

            wait_time = random.uniform(2.0, 5.0)
            logging.warning(f"检测到速率限制，等待 {wait_time:.1f} 秒后重试...")
            time.sleep(wait_time)
            # 不设置force_plaintext，继续重试
        elif getattr(e, "status_code", None) == 404 or "404" in str(e):
            logging.warning("结构化调用返回404，强制使用纯文本降级模式。")
            force_plaintext = True

        def _attempt_structured_json(
            base_messages: list[dict[str, Any]],
            attempt_label: str,
        ) -> Any | None:
            nonlocal structured_retry_count
            if structured_retry_count >= max_structured_retry:
                logging.debug(
                    "call_ai_with_schema: 达到结构化重试上限(%s)，跳过 %s",
                    max_structured_retry,
                    attempt_label,
                )
                return None
            structured_retry_count += 1
            attempt_messages = _coerce_message_content([dict(m) for m in base_messages])
            attempt_messages = _ensure_json_instruction(attempt_messages)
            attempt_messages = _coerce_message_content(attempt_messages)
            raw_response = call_ai(
                config,
                model_name,
                attempt_messages,
                temperature=json_retry_kwargs.get("temperature"),
                max_tokens_output=json_retry_kwargs.get("max_tokens_output", -1),
                top_p=json_retry_kwargs.get("top_p"),
                frequency_penalty=json_retry_kwargs.get("frequency_penalty"),
                presence_penalty=json_retry_kwargs.get("presence_penalty"),
                response_format={"type": "json_object"},
                schema=None,
            )
            if not isinstance(raw_response, str) or not raw_response or "AI模型调用失败" in raw_response:
                logging.debug(
                    "call_ai_with_schema: %s 无有效字符串响应，raw_type=%s",
                    attempt_label,
                    type(raw_response).__name__,
                )
                return None

            logging.debug(
                "call_ai_with_schema: %s raw length=%s",
                attempt_label,
                len(raw_response),
            )
            debug_mode = getattr(getattr(config, "workflow", None), "debug_json_repair", False)
            repaired_text, repaired = repair_json_once(raw_response, schema, debug=debug_mode)
            logging.debug(
                "call_ai_with_schema: %s repair_result repaired=%s, length=%s",
                attempt_label,
                repaired,
                len(repaired_text) if isinstance(repaired_text, str) else "n/a",
            )
            candidate_text = repaired_text if repaired else raw_response
            candidate_text = _clean_text_artifacts(candidate_text)
            try:
                data = json.loads(candidate_text)
            except json.JSONDecodeError as parse_error:
                logging.debug(
                    "结构化回退解析失败（JSON 解析，%s）: %s",
                    attempt_label,
                    parse_error,
                )
            else:
                try:
                    parsed = _safe_model_validate(schema, _massage_structured_payload(schema, data))
                    # 如果之前是可修复错误且修复成功，输出简洁信息
                    if _is_repairable_error:
                        logging.info("✓ 结构化调用失败——已修复")
                    else:
                        logging.info("结构化回退解析成功（%s）。", attempt_label)
                    return parsed
                except Exception as parse_error:
                    logging.debug(
                        "结构化回退解析失败（模型校验，%s）: %s",
                        attempt_label,
                        parse_error,
                    )
            return None

        def _describe_annotation(annotation: Any) -> str:
            if annotation is None:
                return "any"
            origin = get_origin(annotation)
            if origin is None:
                if annotation in (str,):
                    return "string"
                if annotation in (int,):
                    return "integer"
                if annotation in (float,):
                    return "number"
                if annotation in (bool,):
                    return "boolean"
                if annotation in (list, tuple, set):
                    return "array"
                if annotation in (dict, Mapping):
                    return "object"
                return str(annotation)

            if origin in (list, tuple, set):
                args = get_args(annotation)
                inner = _describe_annotation(args[0]) if args else "value"
                return f"array<{inner}>"

            if origin in (dict, Mapping):
                return "object"

            if origin in (Union, types.UnionType):
                args = get_args(annotation)
                if not args:
                    return "any"
                parts: list[str] = []
                has_none = False
                for item in args:
                    if item is type(None):
                        has_none = True
                        continue
                    parts.append(_describe_annotation(item))
                desc = " | ".join(parts) if parts else "any"
                if has_none:
                    desc = f"{desc} | null" if parts else "null"
                return desc

            return str(annotation)

        def _build_schema_skeleton_text() -> str | None:
            if not hasattr(schema, "model_fields"):
                return None
            model_fields = getattr(schema, "model_fields")
            if not model_fields:
                return None

            lines: list[str] = []
            for name, field in model_fields.items():
                annotation = getattr(field, "annotation", None)
                type_desc = _describe_annotation(annotation)
                required = "必填" if getattr(field, "is_required", lambda: False)() else "可选"
                lines.append(f'- "{name}": {required}, 类型 {type_desc}')

            if not lines:
                return None

            skeleton_lines = [
                "严格输出单个 JSON 对象，不得包含 Markdown、代码块或额外说明。",
                "仅使用以下字段（不得新增字段）：",
                *lines,
                "数值字段请使用数字，布尔值请使用 true/false。",
            ]
            return "\n".join(skeleton_lines)

        if not force_plaintext:
            parsed_obj = _attempt_structured_json(
                [dict(m) for m in messages],
                "json_object 强制(原始)",
            )
            if parsed_obj is not None:
                return parsed_obj, "fallback_success"

        skeleton_text = _build_schema_skeleton_text()
        if skeleton_text and not force_plaintext:
            logging.info("call_ai_with_schema: 尝试骨架结构化重试。")
            skeleton_messages = [{"role": "system", "content": skeleton_text}]
            skeleton_messages.extend(dict(m) for m in messages)
            parsed_obj = _attempt_structured_json(
                skeleton_messages,
                "json_object 强制(骨架)",
            )
            if parsed_obj is not None:
                return parsed_obj, "fallback_success"

        if force_plaintext:
            logging.info("call_ai_with_schema: 已启动纯文本降级，不再尝试结构化响应。")

        content = call_ai(
            config,
            model_name,
            messages,
            temperature=plain_kwargs.get("temperature"),
            max_tokens_output=plain_kwargs.get("max_tokens_output", -1),
            top_p=plain_kwargs.get("top_p"),
            frequency_penalty=plain_kwargs.get("frequency_penalty"),
            presence_penalty=plain_kwargs.get("presence_penalty"),
            response_format=None,
            schema=None,
        )

        # 简单尝试直接解析为 JSON -> schema；若失败，交由调用方处理
        json_text: str | None = None
        if content and "AI模型调用失败" not in content:
            try:
                from utils.text_processor import extract_json_from_ai_response
            except Exception as import_exc:  # pragma: no cover - defensive
                logging.debug("无法导入 JSON 提取工具: %s", import_exc)
                extract_json_from_ai_response = None  # type: ignore
            logging.debug("call_ai_with_schema: fallback content length=%s", len(content))
            debug_mode = getattr(getattr(config, "workflow", None), "debug_json_repair", False)
            repaired_text, repaired = repair_json_once(content, schema, debug=debug_mode)
            logging.debug(
                "call_ai_with_schema: fallback repair result repaired=%s, length=%s",
                repaired,
                len(repaired_text) if isinstance(repaired_text, str) else "n/a",
            )
            if repaired:
                json_text = repaired_text
            if not json_text and extract_json_from_ai_response:
                json_text = extract_json_from_ai_response(
                    config,
                    content,
                    context_for_error_log=f"{model_name} fallback for {getattr(schema, '__name__', 'schema')}",
                )
            if json_text:
                json_text = _clean_text_artifacts(json_text)
                try:
                    data = json.loads(json_text)
                except json.JSONDecodeError as parse_error:
                    logging.warning(
                        "回退 JSON 无法解析 %s: %s | 片段=%s",
                        getattr(schema, "__name__", "schema"),
                        parse_error,
                        json_text[:200].replace("\n", " ") if isinstance(json_text, str) else "<non-str>",
                    )
                else:
                    try:
                        parsed_obj = _safe_model_validate(schema, _massage_structured_payload(schema, data))
                        logging.info("结构化回退解析成功。")
                        return parsed_obj, "fallback_success"
                    except Exception as parse_error:
                        logging.warning(
                            "回退 JSON 无法构建 %s: %s | 片段=%s",
                            getattr(schema, "__name__", "schema"),
                            parse_error,
                            json_text[:200].replace("\n", " ") if isinstance(json_text, str) else "<non-str>",
                        )
            else:
                logging.debug("回退响应中未能提取到有效 JSON。")
        logging.warning(
            "结构化回退失败：schema=%s，content_len=%s，json_text=%s",
            getattr(schema, "__name__", "schema"),
            len(content) if isinstance(content, str) else "n/a",
            "available" if json_text else "missing",
        )
        return content, "fallback_failed"


def call_ai_core(
    config: Config,
    model_name: str,
    messages: list[dict[str, str]],
    temperature: float,
    effective_max_output_tokens: int,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    response_format: dict[str, Any] | None = None,
) -> str:
    """
    核心 AI 调用逻辑 (同步版本)，由 tenacity 包装以实现重试。
    """
    client = _ensure_sync_client(config)

    start_time = time.perf_counter()
    # Emit a progress pulse to indicate the API request is being sent
    try:
        tracker = get_tracker(config.task_id)
        if tracker:
            tracker.pulse(f"调用模型 {model_name} 中...（准备发送请求）")
    except Exception as e:
        logger.debug(f"进度追踪器初始化失败: {str(e)}")

    call_params, is_reasoner_model = _build_chat_call_params(
        model_name,
        messages,
        effective_max_output_tokens,
        temperature,
        top_p,
        frequency_penalty,
        presence_penalty,
    )

    if response_format is not None:
        call_params["response_format"] = response_format

    response = client.chat.completions.create(**call_params)
    duration = time.perf_counter() - start_time

    message = response.choices[0].message
    content = message.content
    reasoning_content = getattr(message, "reasoning_content", None)

    total_tokens: int | None = None
    if response.usage:
        total_tokens = response.usage.total_tokens
        cache_hit = getattr(response.usage, "prompt_cache_hit_tokens", None)
        cache_miss = getattr(response.usage, "prompt_cache_miss_tokens", None)
        if cache_hit is not None or cache_miss is not None:
            logging.info(f"    - [KV Cache] 命中: {cache_hit or 0} tokens, 未命中: {cache_miss or 0} tokens")
    logging.info("    - Token usage (total): %s", total_tokens if total_tokens is not None else "unknown")

    logging.info(f"    - Raw content from model: {content[:80] if content else 'None'}...")

    if is_reasoner_model and reasoning_content:
        logging.info(f"    - [深度求索推理器] 提取到思考过程 ({len(reasoning_content)} 字符): {reasoning_content[:500]}...")

    final_content = _clean_text_artifacts(content or "")

    logging.info(f"    - API 调用成功 ({duration:.2f}秒), 模型: {model_name}, 最终内容长度: {len(final_content)} 字符.")
    try:
        tracker = get_tracker(config.task_id)
        if tracker:
            tracker.pulse(f"调用模型 {model_name} 完成，用时 {duration:.1f}s")
    except Exception as e:
        logger.debug(f"进度追踪器更新失败: {str(e)}")

    if not final_content or final_content.isspace():
        logging.warning(f"    - AI 调用返回空内容 (模型: {model_name})")
        if is_reasoner_model:
            raise EmptyResponseFromReasonerError(f"模型 {model_name} 返回空内容。")

    return final_content


def _single_completion_with_meta(
    config: Config,
    model_name: str,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
) -> tuple[str, str | None]:
    """
    执行一次非流式补全，并返回 (content, finish_reason)。仅用于写作型输出的自动续写策略。
    """
    client = _ensure_sync_client(config)
    call_params, _ = _build_chat_call_params(
        model_name,
        messages,
        max_tokens,
        temperature,
        top_p,
        frequency_penalty,
        presence_penalty,
    )
    resp = client.chat.completions.create(**call_params)
    choice = resp.choices[0]
    content = choice.message.content or ""
    finish_reason = getattr(choice, "finish_reason", None)
    return _clean_text_artifacts(content), finish_reason


def call_ai_writing_with_auto_continue(
    config: Config,
    model_name: str,
    messages: list[dict[str, str]],
    *,
    temperature: float | None = None,
    max_tokens_output: int = -1,
    top_p: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    continuation_prompt: str = "请从上文继续，补完未完成的句子与段落，完成该章节。",
    max_continues: int = 1,
) -> str:
    """
    写作型调用：若 finish_reason == 'length'，自动续写一次（最多 max_continues 次）。
    仅用于非结构化写作输出（章节生成、长段文本等）。

    注意：本函数会自动规范化所有 messages，确保符合 API 要求。
    """
    # 规范化所有消息，防止包含 tool_calls 等复杂字段
    messages = _coerce_message_content(messages)

    final_temperature = temperature if temperature is not None else config.generation.temperature_creative
    final_top_p = top_p if top_p is not None else config.generation.top_p_creative
    final_frequency_penalty = frequency_penalty if frequency_penalty is not None else _default_frequency_penalty(config)
    final_presence_penalty = presence_penalty if presence_penalty is not None else _default_presence_penalty(config)

    # 使用集中管理的模型限制
    if max_tokens_output > 0:
        effective_max_output_tokens = min(max_tokens_output, ModelLimits.get_max_output(model_name))
    else:
        effective_max_output_tokens = ModelLimits.get_max_output(model_name)

    retry_exception_types = build_retry_exception_types()
    retryer = build_retryer(config, retry_exception_types)

    # 第一次调用
    content, finish_reason = retryer(
        _single_completion_with_meta,
        config,
        model_name,
        messages,
        temperature=final_temperature,
        max_tokens=effective_max_output_tokens,
        top_p=final_top_p,
        frequency_penalty=final_frequency_penalty,
        presence_penalty=final_presence_penalty,
    )

    if finish_reason == "length" and max_continues > 0:
        logging.info("检测到 finish_reason=length，自动触发一次续写。")
        cont_messages = [
            *messages,
            {"role": "assistant", "content": content},
            {"role": "user", "content": continuation_prompt},
        ]
        extra, _ = retryer(
            _single_completion_with_meta,
            config,
            model_name,
            cont_messages,
            temperature=final_temperature,
            max_tokens=effective_max_output_tokens,
            top_p=final_top_p,
            frequency_penalty=final_frequency_penalty,
            presence_penalty=final_presence_penalty,
        )
        content = (content or "") + ("\n" + extra if extra else "")

    return content or ""


def call_ai(
    config: Config,
    model_name: str,
    messages: list[dict[str, str]],
    temperature: float | None = None,
    max_tokens_output: int = -1,
    top_p: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    response_format: dict[str, Any] | None = None,
    schema: type[Any] | None = None,
) -> Any | str:
    """
    带健壮重试机制和智能 token 管理的 AI 调用封装函数 (完全同步版本)。

    新增参数:
        schema: 可选的 Pydantic BaseModel 类型，用于结构化输出

    注意：本函数会自动规范化所有 messages，移除 tool_calls/function_call 等
    复杂字段，确保符合 OpenAI API 要求。调用者无需手动规范化。
    """
    # 规范化所有消息，确保没有 tool_calls/function_call 等复杂字段
    # 这防止了从 API 响应中获取的消息（可能包含 tool_calls）被直接传递给下游
    messages = _coerce_message_content(messages)

    final_temperature = temperature if temperature is not None else config.generation.temperature_factual
    final_top_p = top_p if top_p is not None else config.generation.top_p_factual
    final_frequency_penalty = frequency_penalty if frequency_penalty is not None else _default_frequency_penalty(config)
    final_presence_penalty = presence_penalty if presence_penalty is not None else _default_presence_penalty(config)

    # 使用集中管理的模型限制
    if max_tokens_output > 0:
        effective_max_output_tokens = min(max_tokens_output, ModelLimits.get_max_output(model_name))
    else:
        effective_max_output_tokens = ModelLimits.get_max_output(model_name)

    is_reasoner_model = "reasoner" in model_name.lower()
    if is_reasoner_model and effective_max_output_tokens < ModelLimits.REASONER_MIN_TOKENS:
        logging.info(f"    - 为 {model_name} 调整 max_tokens_output 至 {ModelLimits.REASONER_MIN_TOKENS} (以容纳思维链)。")
        effective_max_output_tokens = ModelLimits.REASONER_MIN_TOKENS

    # 若需要 JSON 强制格式，自动注入 JSON 提示，避免 400 错误
    effective_messages = messages
    if response_format and isinstance(response_format, dict) and response_format.get("type") == "json_object":
        effective_messages = _ensure_json_instruction(messages)

    total_input_tokens = sum(config.count_tokens(m.get("content", "")) for m in effective_messages)
    logging.info(f"    - AI 调用: 模型={model_name}, 输入 Tokens (估算): {total_input_tokens}, 请求输出 Tokens: {max_tokens_output} -> 有效最大值: {effective_max_output_tokens}")

    model_context_limit = ModelLimits.get_context_limit(model_name)
    if total_input_tokens + effective_max_output_tokens > model_context_limit:
        logging.warning(f"    - 警告: 输入+输出 Tokens ({total_input_tokens + effective_max_output_tokens}) 可能超过模型 {model_name} 的上下文限制 ({model_context_limit})。")
        available_for_output = model_context_limit - total_input_tokens
        if available_for_output < effective_max_output_tokens:
            new_max_output = max(100, available_for_output - 100)
            logging.info(f"    - 调整 max_tokens_output 从 {effective_max_output_tokens} 到 {new_max_output} 以适应上下文。")
            effective_max_output_tokens = new_max_output
        if effective_max_output_tokens <= 0:
            logging.error(f"    - 严重错误: 模型 {model_name} 没有可用的输出令牌。输入令牌: {total_input_tokens}, 上下文限制: {model_context_limit}")
            return "AI模型调用失败 (错误): 输入内容已占满上下文窗口，无法生成回复。"

    retry_exception_types = build_retry_exception_types()

    # 如果有 schema，先尝试结构化调用
    if schema is not None:
        try:
            schema_kwargs: dict[str, Any] = {
                "temperature": final_temperature,
                "max_tokens": effective_max_output_tokens,
                "top_p": final_top_p,
                "frequency_penalty": final_frequency_penalty,
                "presence_penalty": final_presence_penalty,
            }
            result, status = call_ai_with_schema(config, model_name, messages, schema, schema_kwargs)
            if status in ["success", "fallback_success"]:
                return result
            elif status == "instructor_unavailable":
                logging.warning("Instructor不可用，继续普通调用")
            elif status == "fallback_failed":
                logging.warning("结构化调用完全失败，继续普通重试机制")
        except Exception as e:
            logging.warning(f"结构化调用异常: {e}，继续普通重试机制")

    retryer = build_retryer(config, retry_exception_types)

    try:
        # Pulse before entering retry loop
        try:
            tracker = get_tracker(config.task_id)
            if tracker:
                tracker.pulse(f"正在请求 {model_name}（预计输出上限 {effective_max_output_tokens} tokens）")
        except Exception as e:
            logger.debug(f"进度追踪器脉冲失败: {str(e)}")
        return retryer(
            call_ai_core,
            config,
            model_name,
            effective_messages,
            final_temperature,
            effective_max_output_tokens,
            final_top_p,
            final_frequency_penalty,
            final_presence_penalty,
            response_format=response_format,
        )
    except openai.APIStatusError as e:
        logging.error(f"    - 模型 {model_name} 的 API 调用状态错误 (未重试或最终尝试失败): {e.status_code} - {e.response.text if e.response else '无响应文本'}")
        # 当服务端不支持 JSON 强制格式时，移除 response_format 重试一次
        if e.status_code == 400 and response_format is not None:
            logging.warning("    - 收到 400 且使用了 response_format，尝试移除 response_format 后重试一次。")
            try:
                return retryer(
                    call_ai_core,
                    config,
                    model_name,
                    messages,
                    final_temperature,
                    effective_max_output_tokens,
                    final_top_p,
                    final_frequency_penalty,
                    final_presence_penalty,
                    response_format=None,
                )
            except Exception as e_retry:
                logging.error(
                    f"    - 无 response_format 重试仍失败: {e_retry}",
                    exc_info=True,
                )
        if e.status_code == 400:
            logging.error(f"提示：请求可能无效 (例如，输入令牌 {total_input_tokens} + 输出 {effective_max_output_tokens} 超出模型限制)。这是一个不可重试的客户端错误。")
        error_message_detail = "未知错误"
        if e.response is not None:
            try:
                error_message_detail = e.response.json().get("error", {}).get("message", "未知错误")
            except json.JSONDecodeError:
                error_message_detail = e.response.text if e.response.text else "无响应文本"
        return f"AI模型调用失败 (API 错误 {e.status_code}): {error_message_detail}"
    except Exception as e:
        logging.error(f"    - 模型 {model_name} 的 AI 调用因未处理的异常或所有重试后失败: {e}", exc_info=True)
        return "AI模型调用失败，请检查网络连接、API密钥或相关设置，或查看详细日志。"


def preflight_llm_connectivity(config: Config, *, model_name: str | None = None) -> bool:
    """执行一次轻量的连通性预检，快速反馈代理/TLS异常。

    返回 True 表示可用；False 表示失败（调用方可终止并提示用户修复环境）。
    """
    try:
        test_model = model_name or config.models.main_ai_model
        messages = [{"role": "user", "content": "ping"}]
        # 低温度、极短输出，尽量减少等待时间
        resp = call_ai(
            config,
            test_model,
            messages,
            temperature=0.0,
            max_tokens_output=8,
        )
        if resp and "AI模型调用失败" not in resp:
            logging.info("LLM预检通过: 模型=%s", test_model)
            return True
        logging.error("LLM预检失败: %s", resp)
        return False
    except Exception as exc:
        logging.error("LLM预检发生异常: %s", exc, exc_info=True)
        return False
