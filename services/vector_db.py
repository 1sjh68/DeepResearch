# services/vector_db.py

import hashlib
import importlib
import logging
import math
import time
from typing import Any

try:
    from utils.progress import create_progress_bar
    PROGRESS_AVAILABLE = True
except ImportError:
    PROGRESS_AVAILABLE = False
    create_progress_bar = None  # type: ignore

# From refactored modules
from config import Config

# --- Lazy Loaded Dependencies ---
BM25Okapi = None
BM25_AVAILABLE = False
CrossEncoder = None
CROSS_ENCODER_AVAILABLE = False
chromadb = None
openai = None
tenacity = None

_EMBED_WARNED = False

try:
    rank_bm25_module = importlib.import_module("rank_bm25")
    BM25Okapi = getattr(rank_bm25_module, "BM25Okapi", None)
    BM25_AVAILABLE = BM25Okapi is not None
except ImportError:
    logging.warning("`rank_bm25` is not installed. BM25 search will be disabled.")
    pass  # BM25 is optional

try:
    sentence_transformers_module = importlib.import_module("sentence_transformers")
    CrossEncoder = getattr(sentence_transformers_module, "CrossEncoder", None)
    CROSS_ENCODER_AVAILABLE = CrossEncoder is not None
except ImportError:
    logging.warning("`sentence_transformers` is not installed. Cross-encoder reranking will be disabled.")
    pass  # CrossEncoder is optional


class EmbeddingModel:
    """
    Wraps interactions with the embedding model API.
    Handles converting text batches into vector representations.
    """

    def __init__(self, config: Config):
        self.config = config
        self.client: Any = None
        self.model_name = config.embedding_model_name or config.api.embedding_model_name or "bge-m3"
        self._lazy_load_openai()

        if openai is None:
            return

        if not config.api.embedding_api_base_url or not config.api.embedding_api_key:
            logging.warning("EMBEDDING_API_BASE_URL or EMBEDDING_API_KEY not set. Embedding functionality will be disabled.")
            self.client = None
            return

        try:
            self.client = openai.OpenAI(
                base_url=config.api.embedding_api_base_url,
                api_key=config.api.embedding_api_key,
                timeout=300.0,
            )
            logging.debug(
                "Embedding client initialized for %s with model '%s'",
                config.api.embedding_api_base_url,
                self.model_name,
            )
        except Exception as e:
            logging.error(f"Error initializing embedding client: {e}")
            self.client = None

    def _lazy_load_openai(self):
        global openai
        if openai is None:
            try:
                openai = importlib.import_module("openai")
                logging.debug("Successfully lazy-loaded 'openai'.")
            except ImportError:
                logging.error("'openai' library not found. Embedding functionality is disabled. Please install it with 'pip install openai'.")
                self.client = None

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Batch gets embeddings for a list of texts with a robust retry mechanism.
        """
        if not self.client:
            # Warn once to avoid log floods when embeddings are disabled/unavailable
            global _EMBED_WARNED
            if not _EMBED_WARNED:
                logging.warning("Embedding client is not available; embedding-dependent features are disabled.")
                _EMBED_WARNED = True
            else:
                logging.debug("Embedding client unavailable; skipping embedding generation for this batch.")
            return [[] for _ in texts]

        self._lazy_load_tenacity()
        if tenacity is None:
            # Fallback to a single attempt if tenacity is not available
            return self._get_embeddings_single_attempt(texts)

        openai_mod = openai
        if openai_mod is None:
            logging.error("OpenAI library is unavailable; cannot request embeddings.")
            return [[] for _ in texts]

        all_embeddings: list[list[float]] = []
        total_batches = (len(texts) + self.config.vector.embedding_batch_size - 1) // self.config.vector.embedding_batch_size

        # 使用进度条显示 embedding 生成进度
        batch_range = range(0, len(texts), self.config.vector.embedding_batch_size)
        if PROGRESS_AVAILABLE and total_batches > 1:
            batch_iterator = create_progress_bar(
                list(batch_range),
                desc="🧠 生成 Embeddings",
                unit="批次"
            )
        else:
            batch_iterator = batch_range
            if total_batches > 1:
                logging.info(f"🧠 开始生成 embeddings: 共 {total_batches} 批次")

        for i in batch_iterator:
            batch_texts = texts[i : i + self.config.vector.embedding_batch_size]

            try:
                retryer = tenacity.Retrying(
                    wait=tenacity.wait_exponential(
                        multiplier=self.config.runtime.api_retry_wait_multiplier,
                        min=2,
                        max=self.config.runtime.api_retry_max_wait,
                    ),
                    stop=tenacity.stop_after_attempt(self.config.runtime.api_retry_max_attempts),
                    retry=(tenacity.retry_if_exception_type(openai_mod.APITimeoutError) | tenacity.retry_if_exception_type(openai_mod.APIConnectionError) | tenacity.retry_if_exception_type(openai_mod.InternalServerError) | tenacity.retry_if_exception_type(openai_mod.RateLimitError)),
                    before_sleep=tenacity.before_sleep_log(logging.getLogger(__name__), logging.WARNING),
                    reraise=True,
                )
                response = retryer(self.client.embeddings.create, model=self.model_name, input=batch_texts)
                embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings)
                logging.debug(
                    "Successfully obtained %s embeddings for batch %s",
                    len(embeddings),
                    i // self.config.vector.embedding_batch_size + 1,
                )
            except Exception as e_retry:
                logging.error(
                    f"  Failed to get embeddings for batch {i // self.config.vector.embedding_batch_size + 1} after retries: {e_retry}",
                    exc_info=True,
                )
                all_embeddings.extend([[] for _ in batch_texts])
        return all_embeddings

    def _get_embeddings_single_attempt(self, texts: list[str]) -> list[list[float]]:
        # Helper for when tenacity is not available
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.config.vector.embedding_batch_size):
            batch_texts = texts[i : i + self.config.vector.embedding_batch_size]
            try:
                response = self.client.embeddings.create(model=self.model_name, input=batch_texts)
                embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings)
            except Exception as e:
                logging.error(
                    f"  Failed to get embeddings for batch {i // self.config.vector.embedding_batch_size + 1}: {e}",
                    exc_info=True,
                )
                all_embeddings.extend([[] for _ in batch_texts])
        return all_embeddings

    def get_embedding(self, text: str) -> list[float]:
        """Gets the embedding for a single text."""
        embeddings = self.get_embeddings([text])
        return embeddings[0] if embeddings and embeddings[0] else []

    def _lazy_load_tenacity(self):
        global tenacity
        if tenacity is None:
            try:
                tenacity = importlib.import_module("tenacity")
                logging.debug("Successfully lazy-loaded 'tenacity'.")
            except ImportError:
                logging.warning("'tenacity' library not found. The robust retry mechanism is disabled. Please install it with 'pip install tenacity'.")


class VectorDBManager:
    """
    Manages interactions with the ChromaDB vector database.
    Handles data persistence and similarity search, with support for hybrid search and cross-encoder reranking.
    """

    def __init__(self, config: Config, embedding_model: EmbeddingModel):
        self.config = config
        self.embedding_model = embedding_model
        self.client: Any = None
        self.collection: Any = None
        self.bm25_index: Any = None
        self._bm25_documents: list[str] = []
        self._bm25_metadatas: list[dict[str, Any]] = []
        self._bm25_ids: list[str] = []
        self.cross_encoder: Any = None
        self._initialize_db()
        self._initialize_enhanced_search()

    def _lazy_load_chromadb(self):
        global chromadb
        if chromadb is None:
            try:
                chromadb = importlib.import_module("chromadb")
                logging.debug("Successfully lazy-loaded 'chromadb'.")
            except ImportError:
                logging.error("'chromadb' library not found. Vector database functionality is disabled. Please install it with 'pip install chromadb'.")

    def _vector_settings(self) -> Any | None:
        return getattr(self.config, "vector", None)

    def _bm25_config_enabled(self) -> bool:
        vector_settings = self._vector_settings()
        if not vector_settings:
            return True
        return bool(getattr(vector_settings, "enable_bm25_search", True))

    def _rerank_config_enabled(self) -> bool:
        vector_settings = self._vector_settings()
        if not vector_settings:
            return True
        return bool(getattr(vector_settings, "enable_rerank", True))

    def _hybrid_config_enabled(self) -> bool:
        vector_settings = self._vector_settings()
        if not vector_settings:
            return True
        return bool(getattr(vector_settings, "enable_hybrid_search", True))

    def _bm25_ready(self) -> bool:
        return self._bm25_config_enabled() and BM25_AVAILABLE and BM25Okapi is not None

    def _rerank_ready(self) -> bool:
        return self._rerank_config_enabled() and CROSS_ENCODER_AVAILABLE and CrossEncoder is not None

    def _reset_bm25_cache(self) -> None:
        self._bm25_documents = []
        self._bm25_metadatas = []
        self._bm25_ids = []

    def _refresh_bm25_index_if_needed(self) -> None:
        if not self._bm25_ready():
            return
        try:
            self._build_bm25_index()
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("Failed to refresh BM25 index after update: %s", exc)

    def _initialize_db(self):
        """Initializes the database client and collection."""
        self._lazy_load_chromadb()
        if chromadb is None:
            return

        try:
            logging.debug("Initializing ChromaDB at path: %s", self.config.vector.vector_db_path)
            self.client = chromadb.PersistentClient(path=self.config.vector.vector_db_path)
            self.collection = self.client.get_or_create_collection(name=self.config.vector.vector_db_collection_name)
            logging.debug(
                "ChromaDB collection '%s' loaded/created.",
                self.config.vector.vector_db_collection_name,
            )
            self.get_db_stats()
        except Exception as e:
            logging.error(f"Failed to initialize ChromaDB: {e}", exc_info=True)
            self.client = None
            self.collection = None

    def add_experience(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> bool:
        """Adds new experiences (texts, metadatas, etc.) to the database."""
        if not self.collection or not self.embedding_model:
            logging.info("Vector DB or embedding model not initialized. Skipping add.")
            return False
        if not texts:
            logging.warning("Attempted to add an empty list of texts to the experience store.")
            return False

        try:
            embeddings = self.embedding_model.get_embeddings(texts)

            valid_texts: list[str] = []
            valid_embeddings: list[list[float]] = []
            valid_metadatas: list[dict[str, Any]] = []
            valid_ids: list[str] = []

            for i, emb in enumerate(embeddings):
                if emb:
                    valid_texts.append(texts[i])
                    valid_embeddings.append(emb)
                    if metadatas:
                        valid_metadatas.append(metadatas[i])
                    if ids:
                        valid_ids.append(ids[i])
                    else:
                        content_hash = hashlib.md5(texts[i].encode()).hexdigest()
                        valid_ids.append(f"exp_{content_hash}_{int(time.time())}")
                else:
                    logging.warning(f"Could not generate embedding for text, skipping add: {texts[i][:100]}...")

            if not valid_texts:
                logging.warning("No valid embeddings were generated. No new experiences will be added.")
                return False

            logging.info(f"Adding {len(valid_texts)} new experiences to the vector database...")
            self.collection.add(
                embeddings=valid_embeddings,
                documents=valid_texts,
                metadatas=valid_metadatas if valid_metadatas else None,
                ids=valid_ids,
            )
            logging.info(f"Successfully added {len(valid_texts)} new experiences.")
            self._refresh_bm25_index_if_needed()
            self.get_db_stats()
            return True
        except Exception as e:
            logging.error(f"Failed to add experiences to the vector database: {e}", exc_info=True)
            return False

    def retrieve_experience(
        self,
        query_text: str,
        n_results: int = -1,
        where_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieves the most similar experiences from the database for a given query text."""
        if not self.collection or not self.embedding_model:
            logging.info("Vector DB or embedding model not initialized. Retrieval disabled.")
            return []

        if n_results == -1:
            n_results = self.config.vector.num_retrieved_experiences

        try:
            logging.info(f"Retrieving top {n_results} experiences from vector database for query: '{query_text[:100]}...'")
            query_embedding = self.embedding_model.get_embedding(query_text)
            if not query_embedding:
                logging.error("Could not generate embedding for the query text. Retrieval failed.")
                return []

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            retrieved_experiences: list[dict[str, Any]] = []
            if results and results.get("ids") and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    exp = {
                        "id": results["ids"][0][i],
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                    }
                    retrieved_experiences.append(exp)
                logging.info(f"Successfully retrieved {len(retrieved_experiences)} experiences.")
            else:
                logging.info("No relevant experiences found.")
            return retrieved_experiences
        except Exception as e:
            logging.error(f"Failed to retrieve experiences from the vector database: {e}", exc_info=True)
            return []

    def get_db_stats(self) -> dict[str, int]:
        """Gets and prints statistics about the database."""
        if not self.collection:
            logging.info("Vector database collection not initialized. Cannot get stats.")
            return {}
        try:
            count = self.collection.count()
            logging.info(f"Vector database '{self.config.vector.vector_db_collection_name}' currently contains {count} experiences.")
            return {"count": count}
        except Exception as e:
            logging.error(f"Failed to get vector database stats: {e}", exc_info=True)
            return {}

    def _initialize_enhanced_search(self):
        """Initializes enhanced search features (BM25 and cross-encoder)."""
        if self._bm25_ready():
            try:
                self._build_bm25_index()
                logging.info("BM25 index initialized successfully.")
            except Exception as e:
                logging.warning(f"Failed to initialize BM25 index: {e}")
                self.bm25_index = None
        else:
            if not self._bm25_config_enabled():
                logging.debug("BM25 search disabled via configuration; skipping index build.")
            elif not (BM25_AVAILABLE and BM25Okapi is not None):
                logging.warning("`rank_bm25` is not available; BM25 search will remain disabled.")
            self.bm25_index = None
            self._reset_bm25_cache()

        cross_encoder_cls = CrossEncoder
        if self._rerank_ready() and cross_encoder_cls is not None:
            try:
                vector_settings = self._vector_settings()
                model_name = (
                    getattr(vector_settings, "rerank_model_name", None) or "cross-encoder/ms-marco-MiniLM-L-6-v2"
                )
                self.cross_encoder = cross_encoder_cls(model_name)
                logging.info("Cross-encoder initialized successfully.")
            except Exception as e:
                logging.warning(f"Failed to initialize cross-encoder: {e}")
                self.cross_encoder = None
        else:
            if not self._rerank_config_enabled():
                logging.debug("Cross-encoder reranking disabled via configuration.")
            elif not (CROSS_ENCODER_AVAILABLE and cross_encoder_cls is not None):
                logging.warning("`sentence_transformers` is not available; cross-encoder reranking disabled.")
            self.cross_encoder = None

    def _build_bm25_index(self):
        """Builds the BM25 index."""
        if not self.collection:
            logging.warning("ChromaDB collection not initialized, cannot build BM25 index.")
            self._reset_bm25_cache()
            return

        if not BM25_AVAILABLE or BM25Okapi is None:
            logging.warning("BM25 library is not available.")
            self._reset_bm25_cache()
            return

        try:
            results = self.collection.get(include=["documents", "metadatas"], limit=10000)

            documents_raw = results.get("documents") or []
            documents: list[str] = [doc for doc in documents_raw if isinstance(doc, str)]
            if not documents:
                logging.info("No documents available to build BM25 index.")
                self._reset_bm25_cache()
                self.bm25_index = None
                return

            metadatas_raw = results.get("metadatas") or []
            ids_raw = results.get("ids") or []

            self._bm25_documents = documents
            self._bm25_metadatas = []
            self._bm25_ids = []

            for idx in range(len(documents)):
                meta = metadatas_raw[idx] if idx < len(metadatas_raw) else {}
                if not isinstance(meta, dict):
                    meta = {}
                self._bm25_metadatas.append(meta)

                doc_id = ids_raw[idx] if idx < len(ids_raw) else None
                self._bm25_ids.append(str(doc_id) if doc_id is not None else f"bm25_{idx}")

            tokenized_documents: list[list[str]] = [doc.lower().split() for doc in documents]
            self.bm25_index = BM25Okapi(tokenized_documents)
            logging.info(f"BM25 index built with {len(documents)} documents.")

        except Exception as e:
            logging.error(f"Failed to build BM25 index: {e}", exc_info=True)
            self.bm25_index = None
            self._reset_bm25_cache()

    def _bm25_search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        """Performs a search using BM25."""
        if not self.bm25_index or not self._bm25_documents:
            return []

        try:
            query_tokens = query.lower().split()
            bm25_scores = self.bm25_index.get_scores(query_tokens)

            doc_count = min(len(bm25_scores), len(self._bm25_documents))
            if doc_count == 0:
                logging.debug("BM25 index has no scored documents for the given query.")
                return []

            top_indices = sorted(range(doc_count), key=lambda i: bm25_scores[i], reverse=True)[:top_k]

            bm25_results: list[dict[str, Any]] = []
            for idx in top_indices:
                if idx >= len(self._bm25_documents):
                    continue
                document = self._bm25_documents[idx]
                metadata = self._bm25_metadatas[idx] if idx < len(self._bm25_metadatas) else {}
                doc_id = self._bm25_ids[idx] if idx < len(self._bm25_ids) else f"bm25_{idx}"
                bm25_results.append({
                    "id": doc_id,
                    "document": document,
                    "metadata": metadata,
                    "bm25_score": bm25_scores[idx],
                    "source": "bm25",
                })

            logging.info(f"BM25 search completed, returning {len(bm25_results)} results.")
            return bm25_results

        except Exception as e:
            logging.error(f"BM25 search failed: {e}", exc_info=True)
            return []

    def _rerank_results(self, query: str, candidates: list[dict[str, Any]], top_k: int = 10) -> list[dict[str, Any]]:
        """Reranks candidate results using a cross-encoder."""
        if not self.cross_encoder or not candidates:
            return candidates[:top_k]

        try:
            pairs = [(query, candidate["document"]) for candidate in candidates]
            scores = self.cross_encoder.predict(pairs)

            for i, candidate in enumerate(candidates):
                candidate["rerank_score"] = scores[i]

            reranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)

            logging.info(f"Cross-encoder reranking complete, selecting top {top_k} from {len(candidates)} candidates.")
            return reranked[:top_k]

        except Exception as e:
            logging.error(f"Cross-encoder reranking failed: {e}")
            return candidates[:top_k]

    def hybrid_retrieve_experience(
        self,
        query_text: str,
        n_results: int = -1,
        use_rerank: bool = True,
        bm25_weight: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Performs hybrid retrieval by combining vector search and BM25, with optional cross-encoder reranking.
        Respects feature toggles provided via configuration and gracefully degrades when components are unavailable.
        """
        vector_settings = self._vector_settings()
        if n_results == -1:
            default_results = getattr(vector_settings, "num_retrieved_experiences", 5) if vector_settings else 5
            n_results = default_results

        if not self._hybrid_config_enabled():
            logging.info("Hybrid retrieval disabled via configuration; falling back to primary retrieval strategy.")
            semantic_enabled = bool(getattr(vector_settings, "enable_semantic_search", True)) if vector_settings else True
            if semantic_enabled:
                return self.retrieve_experience(query_text, n_results)
            if self._bm25_ready():
                return self._bm25_search(query_text, n_results)
            logging.warning("Hybrid retrieval disabled and no viable fallback (semantic search disabled, BM25 unavailable).")
            return []

        logging.info(f"Starting hybrid retrieval for: '{query_text[:100]}...' (target: {n_results} results)")

        semantic_enabled = bool(getattr(vector_settings, "enable_semantic_search", True)) if vector_settings else True

        vector_results: list[dict[str, Any]] = []
        if semantic_enabled:
            vector_results = self.retrieve_experience(query_text, n_results * 2)
            logging.info(f"Vector search completed with {len(vector_results)} results.")
        else:
            logging.debug("Semantic vector search disabled via configuration; skipping vector retrieval.")

        bm25_results: list[dict[str, Any]] = []
        if self._bm25_ready():
            bm25_results = self._bm25_search(query_text, n_results * 2)
            logging.info(f"BM25 search completed with {len(bm25_results)} results.")
        elif self._bm25_config_enabled():
            logging.warning("BM25 search enabled in configuration but unavailable (missing library or index).")

        combined_results: dict[str, dict[str, Any]] = {}

        for result in vector_results:
            doc_id = result.get("id", "")
            if doc_id:
                result["vector_score"] = 1.0 - result.get("distance", 1.0)
                result["source"] = "vector"
                combined_results[doc_id] = result

        for result in bm25_results:
            doc_id = result.get("id", "")
            if doc_id:
                if doc_id in combined_results:
                    existing = combined_results[doc_id]
                    existing["bm25_score"] = result.get("bm25_score", 0)
                    existing["source"] = "hybrid"
                else:
                    result["vector_score"] = result.get("vector_score", 0.0)
                    result["source"] = "bm25"
                    combined_results[doc_id] = result

        if bm25_weight is None:
            configured_weight = getattr(vector_settings, "bm25_weight", 0.3) if vector_settings else 0.3
            bm25_weight = configured_weight if isinstance(configured_weight, (int, float)) else 0.3
        else:
            bm25_weight = float(bm25_weight)

        bm25_weight = max(0.0, min(1.0, float(bm25_weight)))
        has_bm25_signal = any(res.get("bm25_score") for res in combined_results.values())

        for result in combined_results.values():
            vector_score = result.get("vector_score", 0)
            bm25_score = result.get("bm25_score", 0)

            if bm25_score is not None and bm25_score > 0:
                normalized_bm25 = 1 / (1 + math.exp(-bm25_score / 10))
            else:
                normalized_bm25 = 0

            if has_bm25_signal:
                final_score = (1 - bm25_weight) * vector_score + bm25_weight * normalized_bm25
            else:
                final_score = vector_score
            result["hybrid_score"] = final_score

        hybrid_results: list[dict[str, Any]] = list(combined_results.values())
        hybrid_results.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)

        apply_rerank = use_rerank and self.cross_encoder and self._rerank_ready() and len(hybrid_results) > 5
        if apply_rerank:
            logging.info("Applying cross-encoder reranking...")
            hybrid_results = self._rerank_results(query_text, hybrid_results, n_results)

        final_results: list[dict[str, Any]] = hybrid_results[:n_results]
        logging.info(f"Hybrid retrieval complete, returning {len(final_results)} results.")

        source_stats: dict[str, int] = {}
        for result in final_results:
            source = result.get("source", "unknown")
            source_stats[source] = source_stats.get(source, 0) + 1

        logging.info(f"Result source statistics: {source_stats}")
        return final_results
