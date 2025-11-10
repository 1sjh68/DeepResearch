"""
LaTeX智能处理模块（生产级）

提供LaTeX控制字符还原、转义验证、统计监控等完整功能。
用于修复AI生成JSON中的LaTeX命令损坏问题。

核心功能:
- 智能识别和还原损坏的LaTeX命令（如\x07lpha → \\alpha）
- 支持50+常用LaTeX命令
- 详细的统计和监控
- 优雅的错误处理

Version: 1.0.0
Author: Deep Research Team
Created: 2025-11-07
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

__version__ = "1.0.0"
__all__ = [
    "smart_remove_control_chars_with_latex_recovery",
    "validate_latex_escaping",
    "process_ai_json_response",
    "count_latex_commands",
    "get_recovery_stats",
    "reset_recovery_stats",
    "LaTeXRecoveryStats",
]

# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class LaTeXRecoveryStats:
    """LaTeX还原统计数据"""
    total_processed: int = 0
    total_recovered: int = 0
    total_removed: int = 0
    recovered_commands: dict[str, int] = field(default_factory=dict)
    processing_times: list[float] = field(default_factory=list)
    last_reset: datetime = field(default_factory=datetime.now)

    def add_recovery(self, command: str, processing_time: float):
        """记录一次成功的还原"""
        self.total_processed += 1
        self.total_recovered += 1
        self.recovered_commands[command] = self.recovered_commands.get(command, 0) + 1
        self.processing_times.append(processing_time)

    def add_removal(self, count: int):
        """记录控制字符移除"""
        self.total_removed += count

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "total_processed": self.total_processed,
            "total_recovered": self.total_recovered,
            "total_removed": self.total_removed,
            "recovered_commands": self.recovered_commands,
            "avg_processing_time_ms": sum(self.processing_times) / len(self.processing_times) * 1000 if self.processing_times else 0,
            "last_reset": self.last_reset.isoformat(),
        }

# 全局统计实例
_global_stats = LaTeXRecoveryStats()

# ============================================================================
# 常量定义
# ============================================================================

CONTROL_CHAR_LATEX_MAP: dict[str, str] = {
    '\x07': 'a',   # BEL (bell) → alpha, approx
    '\x08': 'b',   # BS (backspace) → boldsymbol, beta, big, Big
    '\x0c': 'f',   # FF (form feed) → frac, forall
    '\x0e': 'n',   # SO (shift out) → nabla, neq
    '\x12': 'r',   # DC2 → rho
    '\x14': 't',   # DC4 → tau, times, theta, text
}

LATEX_COMMANDS: set[str] = {
    # 希腊字母 (48个 - 包括大小写)
    'alpha', 'Alpha', 'beta', 'Beta', 'gamma', 'Gamma', 'delta', 'Delta',
    'epsilon', 'Epsilon', 'varepsilon', 'zeta', 'Zeta', 'eta', 'Eta',
    'theta', 'Theta', 'vartheta', 'iota', 'Iota', 'kappa', 'Kappa',
    'lambda', 'Lambda', 'mu', 'Mu', 'nu', 'Nu', 'xi', 'Xi',
    'omicron', 'Omicron', 'pi', 'Pi', 'varpi', 'rho', 'Rho', 'varrho',
    'sigma', 'Sigma', 'varsigma', 'tau', 'Tau', 'upsilon', 'Upsilon',
    'phi', 'Phi', 'varphi', 'chi', 'Chi', 'psi', 'Psi', 'omega', 'Omega',

    # 数学字体命令 (10个)
    'boldsymbol', 'mathbf', 'mathcal', 'mathbb', 'mathit', 'mathrm',
    'mathsf', 'mathtt', 'mathfrak', 'pmb',

    # 分数和根号 (5个)
    'frac', 'dfrac', 'tfrac', 'cfrac', 'sqrt',

    # 求和、积分、极限 (12个)
    'sum', 'prod', 'int', 'oint', 'iint', 'iiint',
    'lim', 'liminf', 'limsup', 'sup', 'inf', 'max', 'min',

    # 特殊符号 (10个)
    'infty', 'partial', 'nabla', 'forall', 'exists', 'nexists',
    'emptyset', 'varnothing', 'angle', 'measuredangle',

    # 运算符号 (15个)
    'times', 'cdot', 'div', 'pm', 'mp', 'circ', 'bullet',
    'oplus', 'ominus', 'otimes', 'oslash', 'odot', 'bigcirc',
    'ast', 'star',

    # 关系符号 (15个)
    'leq', 'geq', 'neq', 'approx', 'equiv', 'propto',
    'sim', 'simeq', 'cong', 'subset', 'supset', 'subseteq', 'supseteq',
    'in', 'notin',

    # 括号和分隔符 (8个)
    'left', 'right', 'big', 'Big', 'bigg', 'Bigg', 'middle', 'vert',

    # 文本和格式 (11个)
    'text', 'textbf', 'textit', 'textrm', 'emph', 'underline',
    'overline', 'widehat', 'widetilde', 'overbrace', 'underbrace',

    # 三角函数 (12个)
    'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
    'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh',

    # 对数和指数 (4个)
    'log', 'ln', 'exp', 'lg',

    # 箭头 (8个)
    'to', 'rightarrow', 'leftarrow', 'leftrightarrow',
    'Rightarrow', 'Leftarrow', 'Leftrightarrow', 'mapsto',

    # 其他常用 (10个)
    'vec', 'hat', 'bar', 'dot', 'ddot', 'check', 'breve', 'acute',
    'grave', 'tilde',
}

# ============================================================================
# 核心功能函数
# ============================================================================

def smart_remove_control_chars_with_latex_recovery(
    text: str,
    *,
    debug: bool = False
) -> tuple[str, int]:
    """
    智能移除控制字符，同时尝试还原损坏的LaTeX命令。

    此函数扫描文本中的控制字符，对于可能是损坏的LaTeX命令的控制字符，
    尝试还原为完整的LaTeX命令；对于无法识别的控制字符，直接移除。

    参数
    ----
    text : str
        输入文本，可能包含控制字符
    debug : bool, optional
        是否输出详细的调试日志，默认False

    返回
    ------
    Tuple[str, int]
        (cleaned_text, recovered_count)
        - cleaned_text: 清理后的文本
        - recovered_count: 成功还原的LaTeX命令数量

    示例
    ----
    >>> smart_remove_control_chars_with_latex_recovery('公式\\x07lpha')
    ('公式\\\\\\\\alpha', 1)

    >>> smart_remove_control_chars_with_latex_recovery('\\x08oldsymbol{x}')
    ('\\\\\\\\boldsymbol{x}', 1)

    >>> smart_remove_control_chars_with_latex_recovery('文本\\x01垃圾')
    ('文本垃圾', 0)

    注意
    ----
    - 返回的LaTeX命令包含双反斜杠（\\\\\\\\），适用于JSON字符串
    - 合法的控制字符（\\t, \\n, \\r）不会被移除
    - 处理失败时返回原始文本，确保不会丢失数据
    """
    import time
    start_time = time.time()

    # 输入验证
    if text is None:
        logging.warning("smart_remove_control_chars_with_latex_recovery: 输入为None，返回空字符串")
        return "", 0

    if not isinstance(text, str):
        logging.error("smart_remove_control_chars_with_latex_recovery: 输入类型错误 %s，返回空字符串", type(text))
        return "", 0

    if not text:
        return text, 0

    try:
        result = []
        i = 0
        recovered_count = 0
        removed_count = 0

        while i < len(text):
            ch = text[i]

            # 检查是否是控制字符（排除合法的\t \n \r）
            if (ord(ch) < 0x20 or ord(ch) == 0x7f) and ch not in {'\t', '\n', '\r'}:
                # 尝试LaTeX还原
                recovered = False

                if ch in CONTROL_CHAR_LATEX_MAP:
                    prefix = CONTROL_CHAR_LATEX_MAP[ch]

                    # 向前扫描后续的字母序列
                    j = i + 1
                    suffix = ''
                    while j < len(text) and j < i + 20 and text[j].isalpha():  # 限制最大长度20
                        suffix += text[j]
                        j += 1

                    # 构造可能的LaTeX命令
                    if suffix:  # 确保有后续字母
                        possible_latex = prefix + suffix

                        # 验证是否是已知的LaTeX命令
                        if possible_latex in LATEX_COMMANDS:
                            # 还原为完整的LaTeX命令
                            result.append('\\\\')  # JSON中需要双反斜杠
                            result.append(possible_latex)
                            recovered_count += 1
                            recovered = True

                            # 记录统计
                            _global_stats.add_recovery(possible_latex, time.time() - start_time)

                            if debug:
                                logging.debug(
                                    "LaTeX还原成功: \\x%02x (位置 %d) + '%s' → \\%s",
                                    ord(ch), i, suffix, possible_latex
                                )

                            i = j  # 跳过已处理的字符
                            continue

                if not recovered:
                    # 无法还原，直接移除控制字符
                    removed_count += 1
                    if debug:
                        logging.debug(
                            "移除控制字符: \\x%02x (位置 %d)",
                            ord(ch), i
                        )
            else:
                result.append(ch)

            i += 1

        cleaned_text = ''.join(result)

        # 记录移除统计
        if removed_count > 0:
            _global_stats.add_removal(removed_count)

        # 输出统计信息
        if recovered_count > 0:
            logging.info(
                "LaTeX智能修复完成: 还原 %d 个命令, 移除 %d 个无关控制字符 (耗时 %.2fms)",
                recovered_count, removed_count, (time.time() - start_time) * 1000
            )
        elif removed_count > 0:
            logging.debug(
                "控制字符清理完成: 移除 %d 个字符",
                removed_count
            )

        return cleaned_text, recovered_count

    except Exception as e:
        logging.error(
            "LaTeX智能修复发生异常: %s，返回原始文本",
            e,
            exc_info=True
        )
        # 优雅降级：返回原始文本
        return text, 0


def validate_latex_escaping(json_str: str) -> tuple[bool, list[str]]:
    """
    验证JSON字符串中的LaTeX命令是否正确转义。

    参数
    ----
    json_str : str
        JSON格式的字符串

    返回
    ------
    Tuple[bool, List[str]]
        (is_valid, issues)
        - is_valid: 是否所有LaTeX都正确转义
        - issues: 发现的问题列表

    示例
    ----
    >>> validate_latex_escaping('{"text": "\\\\\\\\alpha"}')
    (True, [])

    >>> validate_latex_escaping('{"text": "\\\\alpha"}')
    (False, ['发现1处可能未转义的LaTeX命令'])
    """
    issues = []

    try:
        # 检测常见的未转义LaTeX命令模式
        # 在JSON字符串值中查找单反斜杠 + 字母的模式
        pattern = r'"[^"]*\\(?!\\)[a-zA-Z]+[^"]*"'
        matches = re.findall(pattern, json_str)

        if matches:
            issues.append(f"发现{len(matches)}处可能未转义的LaTeX命令")
            if len(matches) <= 5:
                # 只显示前5个示例
                for match in matches:
                    issues.append(f"  示例: {match[:50]}...")

        return len(issues) == 0, issues

    except Exception as e:
        logging.error("LaTeX转义验证发生异常: %s", e, exc_info=True)
        return True, []  # 验证失败时假设有效，避免误报


def process_ai_json_response(
    json_str: str,
    *,
    debug: bool = False
) -> str:
    """
    处理AI返回的JSON响应，自动修复LaTeX控制字符问题。

    这是主要的对外API，整合了控制字符移除、LaTeX还原和验证。

    参数
    ----
    json_str : str
        AI返回的JSON字符串
    debug : bool, optional
        是否启用调试日志，默认False

    返回
    ------
    str
        处理后的JSON字符串

    示例
    ----
    >>> response = call_ai(...)
    >>> cleaned = process_ai_json_response(response, debug=True)
    >>> data = json.loads(cleaned)
    """
    cleaned, recovered = smart_remove_control_chars_with_latex_recovery(
        json_str,
        debug=debug
    )

    if debug and recovered == 0:
        # 验证是否有未转义的LaTeX
        is_valid, issues = validate_latex_escaping(cleaned)
        if not is_valid:
            logging.warning("LaTeX转义验证: %s", "; ".join(issues))

    return cleaned


def count_latex_commands(text: str) -> int:
    """
    统计文本中的LaTeX命令数量。

    参数
    ----
    text : str
        输入文本

    返回
    ------
    int
        LaTeX命令数量
    """
    if not text:
        return 0

    pattern = r'\\\\?[a-zA-Z]+'
    matches = re.findall(pattern, text)
    return len(matches)


def get_recovery_stats() -> dict:
    """
    获取LaTeX还原的统计数据。

    返回
    ------
    Dict
        统计数据字典，包含：
        - total_processed: 总处理次数
        - total_recovered: 总还原次数
        - recovered_commands: 各命令还原次数
        - avg_processing_time_ms: 平均处理时间（毫秒）
    """
    return _global_stats.to_dict()


def reset_recovery_stats():
    """重置LaTeX还原统计数据。"""
    global _global_stats
    _global_stats = LaTeXRecoveryStats()
    logging.info("LaTeX还原统计已重置")


def is_known_latex_command(command: str) -> bool:
    """
    检查是否是已知的LaTeX命令。

    参数
    ----
    command : str
        LaTeX命令名（不包含反斜杠）

    返回
    ------
    bool
        是否是已知命令
    """
    return command in LATEX_COMMANDS

