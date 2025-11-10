"""测试工作流执行器"""

import os
from unittest.mock import patch

import pytest

from config import Config
from core.workflow_executor import WorkflowResult, run_workflow_pipeline


class TestWorkflowResult:
    """测试WorkflowResult数据类"""

    def test_workflow_result_creation(self):
        """测试WorkflowResult创建"""
        result = WorkflowResult(raw_result="test content", final_answer="processed content", quality_report="quality: good", saved_filepath="/path/to/file.md", success=True)

        assert result.raw_result == "test content"
        assert result.final_answer == "processed content"
        assert result.quality_report == "quality: good"
        assert result.saved_filepath == "/path/to/file.md"
        assert result.success is True
        assert result.error is None

    def test_workflow_result_with_error(self):
        """测试带错误的WorkflowResult"""
        result = WorkflowResult(raw_result="", final_answer=None, quality_report=None, saved_filepath=None, success=False, error="Test error")

        assert result.success is False
        assert result.error == "Test error"


class TestRunWorkflowPipeline:
    """测试run_workflow_pipeline函数"""

    @pytest.fixture
    def mock_config(self, tmp_path):
        """创建mock配置"""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test", "DISABLE_FINAL_QUALITY_CHECK": "true"}):
            config = Config()
            # 使用 pytest 的 tmp_path，跨平台兼容
            config.session_dir = str(tmp_path)
            return config

    def test_pipeline_with_successful_workflow(self, mock_config, tmp_path):
        """测试成功的工作流执行"""
        mock_config.session_dir = str(tmp_path)
        os.makedirs(tmp_path, exist_ok=True)

        with patch("core.workflow_executor.preflight_llm_connectivity") as mock_preflight:
            mock_preflight.return_value = True

            with patch("core.workflow_executor.run_graph_workflow") as mock_workflow:
                mock_workflow.return_value = "# Test Content\n\nThis is test content."

                with patch("core.workflow_executor.consolidate_document_structure") as mock_consolidate:
                    mock_consolidate.return_value = "# Test Content\n\nConsolidated."

                    with patch("core.workflow_executor.final_post_processing") as mock_post:
                        mock_post.return_value = "Final processed content"

                        result = run_workflow_pipeline(mock_config, vector_db_manager=None, save_result=True)

        assert result.success is True
        assert result.final_answer is not None
        assert "Final processed content" in result.final_answer

    def test_pipeline_with_failed_workflow(self, mock_config):
        """测试失败的工作流执行"""
        with patch("core.workflow_executor.preflight_llm_connectivity") as mock_preflight:
            mock_preflight.return_value = True

            with patch("core.workflow_executor.run_graph_workflow") as mock_workflow:
                mock_workflow.return_value = "错误：测试错误"

                result = run_workflow_pipeline(mock_config, vector_db_manager=None, save_result=False)

        assert result.success is False
        assert result.error is not None
        assert "错误" in result.error

    def test_pipeline_with_empty_result(self, mock_config):
        """测试空结果的工作流执行"""
        with patch("core.workflow_executor.preflight_llm_connectivity") as mock_preflight:
            mock_preflight.return_value = True

            with patch("core.workflow_executor.run_graph_workflow") as mock_workflow:
                mock_workflow.return_value = ""

                result = run_workflow_pipeline(mock_config, vector_db_manager=None, save_result=False)

        assert result.success is False
        assert result.final_answer is None

    def test_pipeline_file_saving(self, mock_config, tmp_path):
        """测试文件保存功能"""
        mock_config.session_dir = str(tmp_path)
        os.makedirs(tmp_path, exist_ok=True)

        with patch("core.workflow_executor.preflight_llm_connectivity") as mock_preflight:
            mock_preflight.return_value = True

            with patch("core.workflow_executor.run_graph_workflow") as mock_workflow:
                mock_workflow.return_value = "# Test\n\nContent"

                with patch("core.workflow_executor.consolidate_document_structure") as mock_consolidate:
                    mock_consolidate.return_value = "# Test\n\nContent"

                    with patch("core.workflow_executor.final_post_processing") as mock_post:
                        mock_post.return_value = "Final content"

                        result = run_workflow_pipeline(mock_config, vector_db_manager=None, save_result=True, output_filename="test_output.md")

        assert result.success is True
        assert result.saved_filepath is not None
        assert "test_output.md" in result.saved_filepath

    def test_pipeline_without_saving(self, mock_config):
        """测试不保存结果的执行"""
        with patch("core.workflow_executor.preflight_llm_connectivity") as mock_preflight:
            mock_preflight.return_value = True

            with patch("core.workflow_executor.run_graph_workflow") as mock_workflow:
                mock_workflow.return_value = "Test content"

                with patch("core.workflow_executor.consolidate_document_structure") as mock_consolidate:
                    mock_consolidate.return_value = "Consolidated"

                    with patch("core.workflow_executor.final_post_processing") as mock_post:
                        mock_post.return_value = "Final"

                        result = run_workflow_pipeline(mock_config, vector_db_manager=None, save_result=False)

        assert result.success is True
        assert result.saved_filepath is None

    def test_pipeline_with_quality_check(self, tmp_path):
        """测试启用质量检查的执行"""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test", "DISABLE_FINAL_QUALITY_CHECK": "false"}):
            mock_config = Config()
            mock_config.session_dir = str(tmp_path)
            os.makedirs(tmp_path, exist_ok=True)

            with patch("core.workflow_executor.preflight_llm_connectivity") as mock_preflight:
                mock_preflight.return_value = True

                with patch("core.workflow_executor.run_graph_workflow") as mock_workflow:
                    mock_workflow.return_value = "Content"

                    with patch("core.workflow_executor.consolidate_document_structure") as mock_consolidate:
                        mock_consolidate.return_value = "Content"

                        with patch("core.workflow_executor.final_post_processing") as mock_post:
                            mock_post.return_value = "Final"

                            with patch("core.workflow_executor.quality_check") as mock_quality:
                                mock_quality.return_value = "Quality: Excellent"

                                result = run_workflow_pipeline(mock_config, vector_db_manager=None, save_result=False)

            assert result.success is True
            assert result.quality_report == "Quality: Excellent"

    def test_pipeline_preflight_failure(self, mock_config):
        """测试预检失败的情况"""
        with patch("core.workflow_executor.preflight_llm_connectivity") as mock_preflight:
            mock_preflight.return_value = False

            with patch("core.workflow_executor.run_graph_workflow") as mock_workflow:
                mock_workflow.return_value = "Content"

                with patch("core.workflow_executor.consolidate_document_structure") as mock_consolidate:
                    mock_consolidate.return_value = "Content"

                    with patch("core.workflow_executor.final_post_processing") as mock_post:
                        mock_post.return_value = "Final"

                        # 预检失败不应该阻止工作流
                        result = run_workflow_pipeline(mock_config, vector_db_manager=None, save_result=False)

        # 工作流仍应继续执行
        assert result is not None
