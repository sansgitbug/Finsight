"""
Cosine-similarity retrieval over FinSight's persisted dual vector store.
"""

from __future__ import annotations

import json
import logging
from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from src.preprocessing.embeddings import (
    EmbeddingConfig,
    EmbeddingError,
    load_embedding_model,
)
from src.retrieval.vectorstore import (
    SearchResult,
    VectorStore,
    VectorStoreConfig,
    VectorStoreError,
    load_vector_store,
    search,
)

LOGGER = logging.getLogger(__name__)


class DenseRetrievalError(Exception):
    """Raised when query embedding or vector-store retrieval fails."""


@dataclass(frozen=True, slots=True)
class DenseRetrieverConfig:
    """Query encoding and candidate-retrieval options."""

    top_k: int = 20
    embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        EmbeddingConfig(
            model_name=self.embedding_model,
            device=self.device,
        )


class QueryEmbeddingModel(Protocol):
    """SentenceTransformer query-encoding API."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        """Encode queries into a vector matrix."""


class DenseRetriever:
    """
    Load FinSight retrieval resources and return cosine-ranked chunks.

    Supports optional ticker filtering.
    """

    def __init__(
        self,
        config: DenseRetrieverConfig | None = None,
        vectorstore_config: VectorStoreConfig | None = None,
        model: QueryEmbeddingModel | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:

        self.config = (
            config or DenseRetrieverConfig()
        )

        self.vectorstore_config = (
            vectorstore_config
            or VectorStoreConfig()
        )

        self._model = model
        self._vector_store = vector_store

    def load(self) -> "DenseRetriever":
        """Load embedding model and vector store."""

        if self._model is None:
            try:
                self._model = load_embedding_model(
                    EmbeddingConfig(
                        model_name=self.config.embedding_model,
                        device=self.config.device,
                    )
                )
            except (
                EmbeddingError,
                ValueError,
            ) as exc:
                raise DenseRetrievalError(
                    f"Unable to load query embedding model: {exc}"
                ) from exc

        if self._vector_store is None:
            try:
                self._vector_store = load_vector_store(
                    self.vectorstore_config
                )
            except VectorStoreError as exc:
                raise DenseRetrievalError(
                    f"Unable to load vector store: {exc}"
                ) from exc

        LOGGER.info(
            "Retriever loaded model %s and collection %s",
            self.config.embedding_model,
            self.vectorstore_config.collection_name,
        )

        return self

    @property
    def vector_store(self) -> VectorStore:
        """Return the loaded vector store."""

        self.load()

        assert self._vector_store is not None

        return self._vector_store

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        """Encode one non-empty query."""

        if not query or not query.strip():
            raise ValueError(
                "query must be non-empty"
            )

        self.load()

        assert self._model is not None

        try:
            vectors = np.asarray(
                self._model.encode(
                    [query.strip()],
                    batch_size=1,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                ),
                dtype=np.float32,
            )

        except Exception as exc:
            raise DenseRetrievalError(
                f"Unable to embed query: {exc}"
            ) from exc

        if (
            vectors.ndim != 2
            or vectors.shape[0] != 1
            or vectors.shape[1] == 0
            or not np.isfinite(vectors).all()
        ):
            raise DenseRetrievalError(
                "Query embedding model returned an invalid vector"
            )

        return vectors[0]

    def retrieve(
        self,
        query_embedding: Sequence[float] | np.ndarray,
        top_k: int | None = None,
        ticker: str | None = None,
    ) -> list[SearchResult]:
        """
        Retrieve cosine-ranked chunks.

        If ticker is provided, only chunks belonging to
        that ticker are returned.
        """

        candidate_count = (
            top_k
            if top_k is not None
            else self.config.top_k
        )

        if candidate_count <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        self.load()

        assert self._vector_store is not None

        # FAISS itself does not apply our ticker metadata filter.
        # Retrieve a larger pool when company filtering is requested,
        # then filter before returning the final candidates.
        search_k = candidate_count

        if ticker:
            search_k = max(
                candidate_count * 20,
                200,
            )

        try:
            results = search(
                self._vector_store,
                query_embedding,
                search_k,
            )

        except (
            VectorStoreError,
            ValueError,
        ) as exc:
            raise DenseRetrievalError(
                f"Vector-store search failed: {exc}"
            ) from exc

        if ticker:
            ticker_normalized = (
                ticker.strip().upper()
            )

            results = [
                result
                for result in results
                if result.chunk.ticker.upper()
                == ticker_normalized
            ]

        LOGGER.info(
            "Retrieved %d cosine-ranked chunks%s",
            len(results),
            (
                f" for ticker {ticker.upper()}"
                if ticker
                else ""
            ),
        )

        return results[:candidate_count]

    def retrieve_from_text(
        self,
        query: str,
        top_k: int | None = None,
        ticker: str | None = None,
    ) -> list[SearchResult]:
        """
        Embed query and return cosine-similarity candidates.
        """

        return self.retrieve(
            self.embed_query(query),
            top_k=top_k,
            ticker=ticker,
        )


def _result_payload(
    rank: int,
    result: SearchResult,
) -> dict[str, Any]:
    """Serialize a retrieval result."""

    return {
        "rank": rank,
        "similarity_score": result.score,
        "chunk": asdict(result.chunk),
        "metadata": result.metadata,
    }


def _print_results(
    results: Sequence[SearchResult],
) -> None:
    """Render retrieval results for CLI."""

    for rank, result in enumerate(
        results,
        start=1,
    ):
        chunk = result.chunk

        preview = " ".join(
            chunk.text.split()
        )[:400]

        print(f"Rank: {rank}")
        print(
            f"Similarity: {result.score:.4f}"
        )
        print(
            f"Ticker: {chunk.ticker}"
        )
        print(
            f"Filing date: {chunk.filing_date}"
        )
        print(
            f"Section: {chunk.section_name}"
        )
        print(
            f"Chunk ID: {chunk.chunk_id}"
        )
        print(
            f"Text: {preview}\n"
        )


def main() -> int:
    """Run cosine retrieval from CLI."""

    parser = ArgumentParser(
        description=(
            "Retrieve FinSight filing chunks "
            "by cosine similarity."
        )
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Natural-language retrieval query",
    )

    parser.add_argument(
        "--vectorstore",
        type=Path,
        default=Path("vectorstore"),
        help=(
            "Directory containing FAISS "
            "and ChromaDB data"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of chunks to retrieve",
    )

    parser.add_argument(
        "--embedding-model",
        default=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        help="SentenceTransformer model name",
    )

    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Embedding execution device",
    )

    parser.add_argument(
        "--json",
        dest="json_output",
        type=Path,
        help="Optional JSON output path",
    )

    arguments: Namespace = (
        parser.parse_args()
    )

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s %(levelname)s "
            "%(name)s: %(message)s"
        ),
    )

    store_directory = arguments.vectorstore

    try:
        retriever = DenseRetriever(
            DenseRetrieverConfig(
                arguments.top_k,
                arguments.embedding_model,
                arguments.device,
            ),
            VectorStoreConfig(
                store_directory / "faiss.index",
                store_directory / "chroma",
            ),
        )

        results = retriever.retrieve_from_text(
            arguments.query
        )

        _print_results(results)

        if arguments.json_output:
            arguments.json_output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            arguments.json_output.write_text(
                json.dumps(
                    [
                        _result_payload(
                            rank,
                            result,
                        )
                        for rank, result in enumerate(
                            results,
                            start=1,
                        )
                    ],
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

    except (
        OSError,
        DenseRetrievalError,
        ValueError,
    ) as exc:

        LOGGER.error(
            "Retrieval failed: %s",
            exc,
        )

        return 1

    return 0


__all__ = [
    "DenseRetrievalError",
    "DenseRetriever",
    "DenseRetrieverConfig",
]


if __name__ == "__main__":
    raise SystemExit(main())