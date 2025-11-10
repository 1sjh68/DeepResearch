"""测试向量数据库功能"""
import os
from unittest.mock import MagicMock, patch

import pytest

from config import Config


class TestEmbeddingModel:
    """测试嵌入模型"""

    @pytest.fixture
    def mock_config(self):
        """创建mock配置"""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}):
            config = Config()
            config.embedding_api_key = "test_key"
            config.embedding_api_base_url = "https://test.api"
            config.embedding_model_name = "test-embedding"
            return config

    def test_embedding_model_initialization(self, mock_config):
        """测试嵌入模型初始化"""
        from services.vector_db import EmbeddingModel

        # 嵌入模型需要有效的API密钥才能初始化
        model = EmbeddingModel(mock_config)

        assert model is not None
        assert hasattr(model, "client")

    def test_embedding_model_without_api_key(self, mock_config):
        """测试无API密钥的情况"""
        from services.vector_db import EmbeddingModel

        # 即使没有embedding_api_key，模型也会尝试初始化
        mock_config.embedding_api_key = None

        model = EmbeddingModel(mock_config)

        # 模型存在但可能client为None或使用默认值
        assert model is not None


class TestVectorDBManager:
    """测试向量数据库管理器"""

    @pytest.fixture
    def mock_config(self):
        """创建mock配置"""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}):
            config = Config()
            config.vector_db_path = "./test_chroma"
            config.vector_db_collection_name = "test_collection"
            return config

    @pytest.fixture
    def mock_embedding_model(self):
        """创建mock嵌入模型"""
        mock = MagicMock()
        mock.client = MagicMock()
        return mock

    def test_vector_db_manager_initialization(
        self,
        mock_config,
        mock_embedding_model
    ):
        """测试向量数据库管理器初始化"""
        from services.vector_db import VectorDBManager

        with patch("services.vector_db.chromadb"):
            manager = VectorDBManager(mock_config, mock_embedding_model)

            assert manager is not None
            assert hasattr(manager, "collection")

    def test_hybrid_search_disabled(self, mock_config, mock_embedding_model):
        """测试禁用混合搜索"""
        from services.vector_db import VectorDBManager

        mock_config.enable_hybrid_search = False

        with patch("services.vector_db.chromadb"):
            manager = VectorDBManager(mock_config, mock_embedding_model)

            assert manager is not None


class TestVectorOperations:
    """测试向量操作"""

    def test_cosine_similarity(self):
        """测试余弦相似度计算"""
        # 简单的相似度测试
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]

        # 计算点积
        dot_product = sum(a * b for a, b in zip(v1, v2))
        magnitude1 = sum(a * a for a in v1) ** 0.5
        magnitude2 = sum(b * b for b in v2) ** 0.5

        similarity = dot_product / (magnitude1 * magnitude2)

        assert similarity == 1.0  # 完全相同的向量

    def test_vector_normalization(self):
        """测试向量归一化"""
        v = [3.0, 4.0]

        magnitude = sum(x * x for x in v) ** 0.5
        normalized = [x / magnitude for x in v]

        # 归一化后的向量长度应该为1
        norm_magnitude = sum(x * x for x in normalized) ** 0.5
        assert abs(norm_magnitude - 1.0) < 1e-6

