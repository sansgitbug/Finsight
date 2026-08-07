"""Cosine-similarity retrieval over FinSight's persisted dual vector store."""

from __future__ import annotations

import json
import logging
from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from src.embeddings import EmbeddingConfig, EmbeddingError, load_embedding_model
from src.vectorstore import SearchResult, VectorStore, VectorStoreConfig, VectorStoreError, load_vector_store, search

LOGGER = logging.getLogger(__name__)


class RetrievalError(Exception):
    """Raised when query embedding or vector-store retrieval fails."""


@dataclass(frozen=True, slots=True)
class RetrieverConfig:
    """Query encoding and candidate-retrieval options.

    Attributes:
        top_k: Number of cosine-similarity candidates to retrieve.
        embedding_model: SentenceTransformer name used for query encoding.
        device: Execution device, either ``"cpu"`` or ``"cuda"``.
    """

    top_k: int = 20
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Validate retrieval limits and the shared embedding model settings."""
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        EmbeddingConfig(model_name=self.embedding_model, device=self.device)


class QueryEmbeddingModel(Protocol):
    """SentenceTransformer query-encoding API used by :class:`Retriever`."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        """Encode one or more queries into a vector matrix."""


class Retriever:
    """Load FinSight retrieval resources and return cosine-ranked chunks.

    The class deliberately returns raw :class:`src.vectorstore.SearchResult`
    instances. This keeps the output lossless and directly consumable by a
    later CrossEncoder reranker without introducing scoring changes here.
    """

    def __init__(
        self,
        config: RetrieverConfig | None = None,
        vectorstore_config: VectorStoreConfig | None = None,
        model: QueryEmbeddingModel | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.config = config or RetrieverConfig()
        self.vectorstore_config = vectorstore_config or VectorStoreConfig()
        self._model = model
        self._vector_store = vector_store

    def load(self) -> Retriever:
        """Load the embedding model and persisted FAISS/ChromaDB resources."""
        if self._model is None:
            try:
                self._model = load_embedding_model(
                    EmbeddingConfig(model_name=self.config.embedding_model, device=self.config.device)
                )
            except (EmbeddingError, ValueError) as exc:
                raise RetrievalError(f"Unable to load query embedding model: {exc}") from exc
        if self._vector_store is None:
            try:
                self._vector_store = load_vector_store(self.vectorstore_config)
            except VectorStoreError as exc:
                raise RetrievalError(f"Unable to load vector store: {exc}") from exc
        LOGGER.info("Retriever loaded model %s and collection %s", self.config.embedding_model, self.vectorstore_config.collection_name)
        return self

    def embed_query(self, query: str) -> np.ndarray:
        """Encode one non-empty query using the same normalized embedding space."""
        if not query or not query.strip():
            raise ValueError("query must be non-empty")
        self.load()
        assert self._model is not None
        try:
            vectors = np.asarray(
                self._model.encode(
                    [query.strip()], batch_size=1, show_progress_bar=False, normalize_embeddings=True
                ),
                dtype=np.float32,
            )
        except Exception as exc:  # SentenceTransformer backend errors vary by environment.
            raise RetrievalError(f"Unable to embed query: {exc}") from exc
        if vectors.ndim != 2 or vectors.shape[0] != 1 or vectors.shape[1] == 0 or not np.isfinite(vectors).all():
            raise RetrievalError("Query embedding model returned an invalid vector")
        return vectors[0]

    def retrieve(self, query_embedding: Sequence[float] | np.ndarray, top_k: int | None = None) -> list[SearchResult]:
        """Retrieve cosine-ranked chunks for an already embedded query."""
        candidate_count = top_k if top_k is not None else self.config.top_k
        if candidate_count <= 0:
            raise ValueError("top_k must be greater than zero")
        self.load()
        assert self._vector_store is not None
        try:
            results = search(self._vector_store, query_embedding, candidate_count)
        except (VectorStoreError, ValueError) as exc:
            raise RetrievalError(f"Vector-store search failed: {exc}") from exc
        LOGGER.info("Retrieved %d cosine-ranked chunks", len(results))
        return results

    def retrieve_from_text(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Embed ``query`` and return its Top-K cosine-similarity candidates."""
        return self.retrieve(self.embed_query(query), top_k)


def _result_payload(rank: int, result: SearchResult) -> dict[str, Any]:
    """Serialize a retrieval result without changing its raw retrieval score."""
    return {"rank": rank, "similarity_score": result.score, "chunk": asdict(result.chunk), "metadata": result.metadata}


def _print_results(results: Sequence[SearchResult]) -> None:
    """Render a compact human-readable result list for the CLI."""
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        preview = " ".join(chunk.text.split())[:400]
        print(f"Rank: {rank}")
        print(f"Similarity: {result.score:.4f}")
        print(f"Ticker: {chunk.ticker}")
        print(f"Filing date: {chunk.filing_date}")
        print(f"Section: {chunk.section_name}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Text: {preview}\n")


def main() -> int:
    """Run cosine retrieval for a text query from the command line."""
    parser = ArgumentParser(description="Retrieve FinSight filing chunks by cosine similarity.")
    parser.add_argument("--query", required=True, help="Natural-language retrieval query")
    parser.add_argument("--vectorstore", type=Path, default=Path("vectorstore"), help="Directory containing FAISS and ChromaDB data")
    parser.add_argument("--top-k", type=int, default=20, help="Number of chunks to retrieve")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Embedding execution device")
    parser.add_argument("--json", dest="json_output", type=Path, help="Optional JSON output path")
    arguments: Namespace = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    store_directory = arguments.vectorstore
    try:
        retriever = Retriever(
            RetrieverConfig(arguments.top_k, arguments.embedding_model, arguments.device),
            VectorStoreConfig(store_directory / "faiss.index", store_directory / "chroma"),
        )
        results = retriever.retrieve_from_text(arguments.query)
        _print_results(results)
        if arguments.json_output:
            arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
            arguments.json_output.write_text(
                json.dumps([_result_payload(rank, result) for rank, result in enumerate(results, start=1)], indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    except (OSError, RetrievalError, ValueError) as exc:
        LOGGER.error("Retrieval failed: %s", exc)
        return 1
    return 0


__all__ = ["RetrievalError", "Retriever", "RetrieverConfig"]


if __name__ == "__main__":
    raise SystemExit(main())
