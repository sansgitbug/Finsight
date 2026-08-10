"""
Cross-encoder reranking for FinSight.

This module reranks retrieval candidates produced by the hybrid retriever
using a CrossEncoder model.
"""

from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from sentence_transformers import CrossEncoder

from src.retrieval.hybrid import RetrievalExplanation
from src.retrieval.vectorstore import SearchResult

LOGGER = logging.getLogger(__name__)


class RerankerError(Exception):
    """Raised when reranking fails."""


@dataclass(slots=True, frozen=True)
class RerankerConfig:
    """Configuration for the CrossEncoder reranker."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cpu"
    top_k_after_reranking: int = 5

    def __post_init__(self) -> None:
        if self.top_k_after_reranking <= 0:
            raise ValueError(
                "top_k_after_reranking must be positive"
            )


@dataclass(slots=True, frozen=True)
class RankedResult:
    """Search result after CrossEncoder reranking."""

    result: SearchResult

    retrieval_score: float
    reranker_score: float

    rank_before: int
    rank_after: int

    dense_rank: int | None = None
    dense_score: float = 0.0

    bm25_rank: int | None = None
    bm25_score: float = 0.0

    rrf_score: float = 0.0


class Reranker:
    """
    Rerank retrieval candidates using a CrossEncoder model.

    The default behavior remains top-5 reranking for the normal FinSight
    pipeline. Callers such as temporal analysis may override top_k when
    they need access to the complete candidate set.
    """

    def __init__(
        self,
        config: RerankerConfig | None = None,
    ) -> None:
        self.config = config or RerankerConfig()
        self._model: CrossEncoder | None = None

    def load(self) -> None:
        """Lazily load the CrossEncoder model."""

        if self._model is not None:
            return

        LOGGER.info(
            "Loading CrossEncoder model %s",
            self.config.model_name,
        )

        try:
            self._model = CrossEncoder(
                self.config.model_name,
                device=self.config.device,
            )
        except Exception as exc:
            raise RerankerError(
                f"Unable to load CrossEncoder model: {exc}"
            ) from exc

    @property
    def model(self) -> CrossEncoder:
        """Return the loaded CrossEncoder model."""

        self.load()

        assert self._model is not None

        return self._model

    def score(
        self,
        query: str,
        results: Sequence[SearchResult],
    ) -> list[float]:
        """Score retrieval candidates using the CrossEncoder."""

        if not query.strip():
            raise ValueError("query must be non-empty")

        if not results:
            return []

        pairs = [
            (
                query,
                result.chunk.text,
            )
            for result in results
        ]

        try:
            scores = self.model.predict(
                pairs,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise RerankerError(
                f"Unable to score retrieval candidates: {exc}"
            ) from exc

        return [float(score) for score in scores]

    def rerank(
        self,
        query: str,
        results: Sequence[SearchResult],
        explanations: dict[str, RetrievalExplanation] | None = None,
        top_k: int | None = None,
    ) -> list[RankedResult]:
        """
        Rerank retrieval candidates using the CrossEncoder.

        Parameters
        ----------
        query:
            User query.

        results:
            Retrieval candidates.

        explanations:
            Optional hybrid retrieval explanations containing dense,
            BM25, and RRF signals.

        top_k:
            Optional override for the number of results returned.
            If omitted, ``RerankerConfig.top_k_after_reranking`` is used.
        """

        if not query.strip():
            raise ValueError("query must be non-empty")

        if not results:
            return []

        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")

        reranker_scores = self.score(
            query,
            results,
        )

        ranked: list[RankedResult] = []

        for original_rank, (
            result,
            reranker_score,
        ) in enumerate(
            zip(
                results,
                reranker_scores,
                strict=True,
            ),
            start=1,
        ):
            explanation = (
                explanations.get(result.chunk.chunk_id)
                if explanations is not None
                else None
            )

            ranked.append(
                RankedResult(
                    result=result,
                    retrieval_score=result.score,
                    reranker_score=reranker_score,
                    rank_before=original_rank,
                    rank_after=0,
                    dense_rank=(
                        explanation.dense_rank
                        if explanation is not None
                        else None
                    ),
                    dense_score=(
                        explanation.dense_score
                        if explanation is not None
                        else 0.0
                    ),
                    bm25_rank=(
                        explanation.bm25_rank
                        if explanation is not None
                        else None
                    ),
                    bm25_score=(
                        explanation.bm25_score
                        if explanation is not None
                        else 0.0
                    ),
                    rrf_score=(
                        explanation.rrf_score
                        if explanation is not None
                        else result.score
                    ),
                )
            )

        ranked.sort(
            key=lambda candidate: candidate.reranker_score,
            reverse=True,
        )

        candidate_limit = (
            top_k
            if top_k is not None
            else self.config.top_k_after_reranking
        )

        final_results: list[RankedResult] = []

        for new_rank, candidate in enumerate(
            ranked[:candidate_limit],
            start=1,
        ):
            final_results.append(
                RankedResult(
                    result=candidate.result,
                    retrieval_score=candidate.retrieval_score,
                    reranker_score=candidate.reranker_score,
                    rank_before=candidate.rank_before,
                    rank_after=new_rank,
                    dense_rank=candidate.dense_rank,
                    dense_score=candidate.dense_score,
                    bm25_rank=candidate.bm25_rank,
                    bm25_score=candidate.bm25_score,
                    rrf_score=candidate.rrf_score,
                )
            )

        LOGGER.info(
            "Reranked %d candidates into top %d results",
            len(results),
            len(final_results),
        )

        return final_results


def print_results(
    results: Sequence[RankedResult],
) -> None:
    """Print reranked retrieval results with explanations."""

    for result in results:
        chunk = result.result.chunk

        preview = " ".join(
            chunk.text.split()
        )[:400]

        print("-" * 80)
        print(f"Final Rank       : {result.rank_after}")
        print(f"Previous Rank    : {result.rank_before}")
        print()
        print(f"Dense Rank       : {result.dense_rank}")
        print(f"Dense Score      : {result.dense_score:.4f}")
        print()
        print(f"BM25 Rank        : {result.bm25_rank}")
        print(f"BM25 Score       : {result.bm25_score:.4f}")
        print()
        print(f"RRF Score        : {result.rrf_score:.4f}")
        print(f"CrossEncoder     : {result.reranker_score:.4f}")
        print()
        print(f"Ticker           : {chunk.ticker}")
        print(f"Filing Date      : {chunk.filing_date}")
        print(f"Section          : {chunk.section_name}")
        print(f"Chunk ID          : {chunk.chunk_id}")
        print()
        print(preview)
        print()


def save_results(
    results: Sequence[RankedResult],
    output: Path,
) -> None:
    """Save reranked results to JSON."""

    payload = []

    for result in results:
        payload.append(
            {
                "rank_before": result.rank_before,
                "rank_after": result.rank_after,
                "dense_rank": result.dense_rank,
                "dense_score": result.dense_score,
                "bm25_rank": result.bm25_rank,
                "bm25_score": result.bm25_score,
                "rrf_score": result.rrf_score,
                "retrieval_score": result.retrieval_score,
                "reranker_score": result.reranker_score,
                "chunk": asdict(result.result.chunk),
                "metadata": result.result.metadata,
            }
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


__all__ = [
    "RankedResult",
    "Reranker",
    "RerankerConfig",
    "RerankerError",
    "print_results",
    "save_results",
]