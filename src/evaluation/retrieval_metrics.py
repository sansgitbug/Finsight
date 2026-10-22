
"""
Evaluation metrics for FinSight retrieval.

Supports Recall@K, Precision@K, and Mean Reciprocal Rank (MRR).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.retrieval.vectorstore import SearchResult


class EvaluationError(Exception):
    """Raised when retrieval evaluation fails."""


@dataclass(frozen=True, slots=True)
class RetrievalEvaluation:
    """Evaluation results for one retrieval query."""

    query: str
    k: int
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float


def _validate_inputs(
    results: Sequence[SearchResult],
    relevant_chunk_ids: set[str],
    k: int,
) -> None:
    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant_chunk_ids:
        raise EvaluationError(
            "relevant_chunk_ids cannot be empty"
        )


def recall_at_k(
    results: Sequence[SearchResult],
    relevant_chunk_ids: set[str],
    k: int,
) -> float:
    """Calculate Recall@K."""

    _validate_inputs(results, relevant_chunk_ids, k)

    retrieved_ids = {
        result.chunk.chunk_id
        for result in results[:k]
    }

    hits = len(retrieved_ids & relevant_chunk_ids)

    return hits / len(relevant_chunk_ids)


def precision_at_k(
    results: Sequence[SearchResult],
    relevant_chunk_ids: set[str],
    k: int,
) -> float:
    """Calculate Precision@K."""

    _validate_inputs(results, relevant_chunk_ids, k)

    retrieved = results[:k]

    hits = sum(
        result.chunk.chunk_id in relevant_chunk_ids
        for result in retrieved
    )

    return hits / k


def reciprocal_rank(
    results: Sequence[SearchResult],
    relevant_chunk_ids: set[str],
    k: int = 5,
) -> float:
    """Calculate reciprocal rank within the top K results."""

    _validate_inputs(results, relevant_chunk_ids, k)

    for rank, result in enumerate(results[:k], start=1):
        if result.chunk.chunk_id in relevant_chunk_ids:
            return 1.0 / rank

    return 0.0


def evaluate_retrieval(
    query: str,
    results: Sequence[SearchResult],
    relevant_chunk_ids: set[str],
    k: int = 5,
) -> RetrievalEvaluation:
    """Evaluate one query using Recall@K, Precision@K, and MRR@K."""

    if not query or not query.strip():
        raise ValueError("query must be non-empty")

    return RetrievalEvaluation(
        query=query,
        k=k,
        recall_at_k=recall_at_k(
            results,
            relevant_chunk_ids,
            k,
        ),
        precision_at_k=precision_at_k(
            results,
            relevant_chunk_ids,
            k,
        ),
        reciprocal_rank=reciprocal_rank(
            results,
            relevant_chunk_ids,
            k,
        ),
    )


__all__ = [
    "EvaluationError",
    "RetrievalEvaluation",
    "evaluate_retrieval",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
