"""
Hybrid retrieval using Dense Retrieval + BM25 + Reciprocal Rank Fusion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import (
    DenseRetriever,
    DenseRetrieverConfig,
    DenseRetrievalError,
)
from src.retrieval.vectorstore import (
    SearchResult,
    VectorStore,
    VectorStoreConfig,
    get_all_chunks,
)

LOGGER = logging.getLogger(__name__)


class HybridRetrievalError(Exception):
    """Raised when hybrid retrieval fails."""


@dataclass(slots=True, frozen=True)
class HybridRetrieverConfig:
    """Configuration for hybrid retrieval."""

    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_final: int = 10
    rrf_k: int = 60

    embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.top_k_dense <= 0:
            raise ValueError("top_k_dense must be positive")

        if self.top_k_sparse <= 0:
            raise ValueError("top_k_sparse must be positive")

        if self.top_k_final <= 0:
            raise ValueError("top_k_final must be positive")

        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")


@dataclass(slots=True, frozen=True)
class RetrievalExplanation:
    """Explain how a chunk was ranked by hybrid retrieval."""

    chunk_id: str

    dense_rank: int | None
    dense_score: float

    bm25_rank: int | None
    bm25_score: float

    rrf_score: float


class HybridRetriever:
    """
    Hybrid retriever combining:

    - Dense semantic retrieval
    - Sparse BM25 retrieval
    - Reciprocal Rank Fusion
    """

    def __init__(
        self,
        config: HybridRetrieverConfig | None = None,
        vectorstore_config: VectorStoreConfig | None = None,
        dense_retriever: DenseRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:

        self.config = (
            config or HybridRetrieverConfig()
        )

        self.vectorstore_config = (
            vectorstore_config
            or VectorStoreConfig()
        )

        self._vector_store = vector_store
        self._dense = dense_retriever
        self._bm25 = bm25_retriever

    def load(self) -> "HybridRetriever":
        """Load retrieval resources."""

        if self._dense is None:
            self._dense = DenseRetriever(
                DenseRetrieverConfig(
                    top_k=self.config.top_k_dense,
                    embedding_model=self.config.embedding_model,
                    device=self.config.device,
                ),
                vectorstore_config=self.vectorstore_config,
                vector_store=self._vector_store,
            )

        self._dense.load()

        if self._vector_store is None:
            self._vector_store = self._dense.vector_store

        if self._bm25 is None:
            assert self._vector_store is not None

            chunks = get_all_chunks(
                self._vector_store
            )

            self._bm25 = BM25Retriever(chunks)

        LOGGER.info(
            "Hybrid retriever ready (%d BM25 docs)",
            len(self._bm25.chunks),
        )

        return self

    def _rrf(
        self,
        dense: Sequence[SearchResult],
        sparse,
    ) -> dict[str, float]:
        """Compute Reciprocal Rank Fusion scores."""

        scores: dict[str, float] = {}

        for rank, result in enumerate(
            dense,
            start=1,
        ):
            scores.setdefault(
                result.chunk.chunk_id,
                0.0,
            )

            scores[
                result.chunk.chunk_id
            ] += 1.0 / (
                self.config.rrf_k + rank
            )

        for rank, result in enumerate(
            sparse,
            start=1,
        ):
            scores.setdefault(
                result.chunk.chunk_id,
                0.0,
            )

            scores[
                result.chunk.chunk_id
            ] += 1.0 / (
                self.config.rrf_k + rank
            )

        return scores

    def explain(
        self,
        query: str,
        ticker: str | None = None,
    ) -> dict[str, RetrievalExplanation]:
        """
        Return dense, BM25, and RRF signals for candidates.
        """

        if not query or not query.strip():
            raise ValueError(
                "query must be non-empty"
            )

        self.load()

        assert self._dense is not None
        assert self._bm25 is not None

        normalized_ticker = (
            ticker.strip().upper()
            if ticker
            else None
        )

        try:
            query_embedding = (
                self._dense.embed_query(query)
            )

            dense_results = (
                self._dense.retrieve(
                    query_embedding,
                    top_k=self.config.top_k_dense,
                    ticker=normalized_ticker,
                )
            )

            sparse_results = (
                self._bm25.search(
                    query,
                    top_k=self.config.top_k_sparse,
                    ticker=normalized_ticker,
                )
            )

        except DenseRetrievalError as exc:
            raise HybridRetrievalError(
                str(exc)
            ) from exc

        rrf_scores = self._rrf(
            dense_results,
            sparse_results,
        )

        dense_info = {
            result.chunk.chunk_id: (
                rank,
                result.score,
            )
            for rank, result in enumerate(
                dense_results,
                start=1,
            )
        }

        sparse_info = {
            result.chunk.chunk_id: (
                rank,
                result.score,
            )
            for rank, result in enumerate(
                sparse_results,
                start=1,
            )
        }

        chunk_ids = (
            set(dense_info)
            | set(sparse_info)
        )

        explanations: dict[
            str,
            RetrievalExplanation,
        ] = {}

        for chunk_id in chunk_ids:

            dense_rank, dense_score = (
                dense_info.get(
                    chunk_id,
                    (None, 0.0),
                )
            )

            bm25_rank, bm25_score = (
                sparse_info.get(
                    chunk_id,
                    (None, 0.0),
                )
            )

            explanations[chunk_id] = (
                RetrievalExplanation(
                    chunk_id=chunk_id,
                    dense_rank=dense_rank,
                    dense_score=float(
                        dense_score
                    ),
                    bm25_rank=bm25_rank,
                    bm25_score=float(
                        bm25_score
                    ),
                    rrf_score=float(
                        rrf_scores.get(
                            chunk_id,
                            0.0,
                        )
                    ),
                )
            )

        return explanations

    def retrieve_from_text(
        self,
        query: str,
        ticker: str | None = None,
    ) -> list[SearchResult]:
        """
        Retrieve using dense + BM25 + RRF.

        If ticker is provided, only chunks belonging to
        that company are considered.
        """

        if not query or not query.strip():
            raise ValueError(
                "query must be non-empty"
            )

        self.load()

        assert self._dense is not None
        assert self._bm25 is not None

        normalized_ticker = (
            ticker.strip().upper()
            if ticker
            else None
        )

        try:
            query_embedding = (
                self._dense.embed_query(query)
            )

            dense_results = (
                self._dense.retrieve(
                    query_embedding,
                    top_k=self.config.top_k_dense,
                    ticker=normalized_ticker,
                )
            )

            sparse_results = (
                self._bm25.search(
                    query,
                    top_k=self.config.top_k_sparse,
                    ticker=normalized_ticker,
                )
            )

        except DenseRetrievalError as exc:
            raise HybridRetrievalError(
                str(exc)
            ) from exc

        rrf_scores = self._rrf(
            dense_results,
            sparse_results,
        )

        dense_lookup = {
            result.chunk.chunk_id: result
            for result in dense_results
        }

        for result in sparse_results:

            if result.chunk.chunk_id not in dense_lookup:
                dense_lookup[result.chunk.chunk_id] = SearchResult(
                    chunk=result.chunk,
                    score=0.0,
                    metadata=result.chunk.metadata,
                )

        merged: list[SearchResult] = []

        for chunk_id, result in (
            dense_lookup.items()
        ):
            merged.append(
                SearchResult(
                    chunk=result.chunk,
                    score=rrf_scores.get(
                        chunk_id,
                        0.0,
                    ),
                    metadata=result.metadata,
                )
            )

        merged.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        LOGGER.info(
            "Hybrid retrieval produced %d "
            "merged candidates%s",
            len(merged),
            (
                f" for ticker {normalized_ticker}"
                if normalized_ticker
                else ""
            ),
        )

        return merged[
            : self.config.top_k_final
        ]


__all__ = [
    "HybridRetriever",
    "HybridRetrieverConfig",
    "HybridRetrievalError",
    "RetrievalExplanation",
]