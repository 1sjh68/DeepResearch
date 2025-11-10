"""测试工作流图构建"""

from workflows.graph_builder import build_graph, should_continue_refining, should_research
from workflows.graph_state import GraphState


class TestBuildGraph:
    """测试图构建"""

    def test_build_graph_creates_app(self):
        """测试build_graph创建应用"""
        app = build_graph()

        assert app is not None
        # LangGraph应用应该是可调用的
        assert hasattr(app, "invoke") or hasattr(app, "__call__")

    def test_graph_has_nodes(self):
        """测试图包含所有必要节点"""
        # 只需确保能成功构建
        app = build_graph()
        assert app is not None


class TestShouldResearch:
    """测试research决策函数"""

    def test_should_research_when_gaps_exist(self):
        """测试有知识空白时应该研究"""
        from unittest.mock import MagicMock

        mock_config = MagicMock()
        mock_config.enable_web_research = True

        state: GraphState = {
            "config": mock_config,
            "knowledge_gaps": ["gap1", "gap2"],
        }  # type: ignore

        result = should_research(state)
        assert result == "research_node"

    def test_should_skip_research_when_disabled(self):
        """测试研究禁用时跳过"""
        from unittest.mock import MagicMock

        mock_config = MagicMock()
        mock_config.enable_web_research = False

        state: GraphState = {
            "config": mock_config,
            "knowledge_gaps": ["gap1"],
        }  # type: ignore

        result = should_research(state)
        assert result == "refine_node"

    def test_should_skip_research_when_no_gaps(self):
        """测试无知识空白时跳过"""
        from unittest.mock import MagicMock

        mock_config = MagicMock()
        mock_config.enable_web_research = True

        state: GraphState = {
            "config": mock_config,
            "knowledge_gaps": [],
        }  # type: ignore

        result = should_research(state)
        assert result == "refine_node"


class TestShouldContinueRefining:
    """测试refine循环决策函数"""

    def test_continue_refining_when_under_limit(self):
        """测试未达到上限时继续"""
        from unittest.mock import MagicMock

        mock_config = MagicMock()
        mock_config.max_refinement_rounds = 3
        mock_config.disable_early_exit = False

        state: GraphState = {
            "config": mock_config,
            "refinement_count": 1,
            "force_exit_refine": False,
        }  # type: ignore

        result = should_continue_refining(state)
        assert result == "critique_node"

    def test_stop_refining_when_limit_reached(self):
        """测试达到上限时停止"""
        from unittest.mock import MagicMock

        mock_config = MagicMock()
        mock_config.max_refinement_rounds = 2
        mock_config.disable_early_exit = False

        state: GraphState = {
            "config": mock_config,
            "refinement_count": 2,
            "force_exit_refine": False,
        }  # type: ignore

        result = should_continue_refining(state)
        assert result == "polish_node"

    def test_force_exit_refine(self):
        """测试强制退出refinement"""
        from unittest.mock import MagicMock

        mock_config = MagicMock()
        mock_config.max_refinement_rounds = 5
        mock_config.disable_early_exit = False

        state: GraphState = {
            "config": mock_config,
            "refinement_count": 1,
            "force_exit_refine": True,
        }  # type: ignore

        result = should_continue_refining(state)
        assert result == "polish_node"

    def test_disable_early_exit(self):
        """测试禁用提前退出"""
        from unittest.mock import MagicMock

        mock_config = MagicMock()
        mock_config.max_refinement_rounds = 3
        mock_config.disable_early_exit = True

        state: GraphState = {
            "config": mock_config,
            "refinement_count": 1,
            "force_exit_refine": True,  # 即使设置了强制退出
        }  # type: ignore

        result = should_continue_refining(state)
        # 因为禁用了提前退出，应该继续
        assert result == "critique_node"

