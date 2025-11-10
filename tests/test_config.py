"""测试Config配置模块"""
import os
from unittest.mock import MagicMock, patch

from config import Config
from config.config import (
    APISettings,
    FetcherSettings,
    GenerationSettings,
    ModelSettings,
    OutputQualitySettings,
    ProxySettings,
    RuntimeSettings,
    VectorRetrievalSettings,
    WorkflowFlags,
)


class TestConfig:
    """测试Config类"""

    def test_config_initialization(self):
        """测试Config初始化"""
        with patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "test_key",
            "USER_PROBLEM": "test problem"
        }):
            config = Config()

            assert config is not None
            assert config.api is not None
            assert config.vector is not None
            assert config.models is not None
            assert config.generation is not None
            assert config.workflow is not None
            assert config.runtime is not None
            assert config.proxies is not None
            assert config.fetch is not None

    def test_api_settings_initialization(self):
        """测试API设置初始化"""
        with patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "test_key_123"
        }):
            config = Config()

            assert config.api.deepseek_api_key == "test_key_123"
            assert isinstance(config.api, APISettings)

    def test_backward_compatibility_attributes(self):
        """测试向后兼容的属性访问"""
        with patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "test_key"
        }):
            config = Config()

            # 测试扁平化属性访问（向后兼容）
            assert config.deepseek_api_key == config.api.deepseek_api_key
            assert config.main_ai_model == config.models.main_ai_model
            assert config.temperature_factual == config.generation.temperature_factual

    def test_token_counting(self):
        """测试token计数功能"""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}):
            config = Config()

            # 测试基本文本
            text = "Hello, world!"
            count = config.count_tokens(text)
            assert count > 0
            assert isinstance(count, int)

            # 测试空文本
            assert config.count_tokens("") == 0
            assert config.count_tokens(None) == 0  # type: ignore

    def test_token_counting_cache(self):
        """测试token计数缓存"""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}):
            config = Config()

            text = "This is a test text for caching"

            # 第一次调用
            count1 = config.count_tokens(text)
            # 第二次调用（应该命中缓存）
            count2 = config.count_tokens(text)

            assert count1 == count2

    def test_session_paths(self):
        """测试会话路径配置"""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}):
            config = Config()

            assert hasattr(config, "session_base_dir")
            assert hasattr(config, "session_dir")
            assert hasattr(config, "log_file_path")

            # 确保session_base_dir存在
            assert os.path.exists(config.session_base_dir)

    def test_client_initialization(self):
        """测试客户端初始化"""
        with patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "sk-test-key"
        }):
            config = Config()

            # 初始状态
            assert config.client is None

            # 初始化客户端
            with patch("config.client_factory.create_deepseek_client") as mock:
                mock.return_value = MagicMock()
                config.initialize_deepseek_client()
                assert config.client is not None

    def test_user_problem_loading(self):
        """测试用户问题从环境变量加载"""
        test_problem = "这是一个测试问题"
        with patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "test",
            "USER_PROBLEM": test_problem
        }):
            config = Config()
            assert config.user_problem == test_problem

    def test_external_files_loading(self):
        """测试外部文件列表加载"""
        test_files = "file1.txt,file2.pdf,file3.md"
        with patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "test",
            "EXTERNAL_FILES": test_files
        }):
            config = Config()
            assert len(config.external_data_files) == 3
            assert "file1.txt" in config.external_data_files
            assert "file2.pdf" in config.external_data_files

    def test_max_iterations_property(self):
        """测试max_iterations向后兼容属性"""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}):
            config = Config()

            # 读取
            assert config.max_iterations == config.runtime.max_refinement_rounds

            # 写入
            config.max_iterations = 5
            assert config.max_refinement_rounds == 5


class TestSettingsGroups:
    """测试各个设置组"""

    def test_model_settings(self):
        """测试模型设置"""
        with patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "test",
            "MAIN_AI_MODEL": "deepseek-chat"
        }):
            config = Config()

            assert config.models.main_ai_model == "deepseek-chat"
            assert isinstance(config.models, ModelSettings)

    def test_generation_settings(self):
        """测试生成设置"""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}):
            config = Config()

            assert isinstance(config.generation, GenerationSettings)
            assert 0 <= config.generation.temperature_factual <= 1
            assert 0 <= config.generation.top_p_factual <= 1

    def test_workflow_flags(self):
        """测试工作流标志"""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}):
            config = Config()

            assert isinstance(config.workflow, WorkflowFlags)
            assert isinstance(config.workflow.enable_web_research, bool)

    def test_vector_settings(self):
        """测试向量检索设置"""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}):
            config = Config()

            assert isinstance(config.vector, VectorRetrievalSettings)
            assert hasattr(config.vector, "vector_db_path")

    def test_fetcher_settings(self):
        """测试抓取器设置"""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}):
            config = Config()

            assert isinstance(config.fetch, FetcherSettings)
            assert config.fetch.fetch_timeout > 0

    def test_proxy_settings(self):
        """测试代理设置"""
        with patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "test",
            "LLM_DISABLE_PROXY": "true"
        }):
            config = Config()

            assert isinstance(config.proxies, ProxySettings)
            assert config.proxies.llm_disable_proxy is True

    def test_runtime_settings(self):
        """测试运行时设置"""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}):
            config = Config()

            assert isinstance(config.runtime, RuntimeSettings)
            assert config.runtime.max_refinement_rounds > 0

    def test_output_quality_settings(self):
        """测试输出质量设置"""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}):
            config = Config()

            assert isinstance(config.output, OutputQualitySettings)
            assert hasattr(config.output, "citation_style")


class TestLogging:
    """测试日志配置"""

    def test_setup_logging(self):
        """测试日志设置"""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}):
            config = Config()

            # 应该能够调用setup_logging而不出错
            with patch("config.logging_setup.setup_session_logging"):
                config.setup_logging()


class TestUserAgentConfiguration:
    """测试User-Agent配置"""

    def test_user_agent_env_configuration(self):
        """测试User-Agent环境变量配置"""
        test_ua = "TestBot/1.0"
        with patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "test",
            "USER_AGENT": test_ua
        }):
            config = Config()
            assert config.user_agent == test_ua


class TestEdgeCases:
    """测试边界情况"""

    def test_empty_external_files(self):
        """测试空的外部文件列表"""
        with patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "test",
            "EXTERNAL_FILES": ""
        }):
            config = Config()
            assert len(config.external_data_files) == 0

    def test_missing_optional_settings(self):
        """测试缺失可选设置"""
        with patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "test"
        }, clear=True):
            config = Config()
            # 应该使用默认值而不报错
            assert config.user_problem is not None or config.user_problem == ""

