"""
Cross-encoder reranking for FinSight.

This module reranks semantic retrieval candidates produced by the Retriever
using a CrossEncoder model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from sentence_transformers import CrossEncoder

from src.retrieval.vectorstore import SearchResult

from dataclasses import asdict
import json
from pathlib import Path
from argparse import ArgumentParser

LOGGER = logging.getLogger(__name__)


class RerankerError(Exception):
    """Raised when reranking fails."""


@dataclass(slots=True, frozen=True)
class RerankerConfig:
    """Configuration for the CrossEncoder reranker."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cpu"
    top_k_after_reranking: int = 5


@dataclass(slots=True, frozen=True)
class RankedResult:
    """
    Search result after reranking.
    """

    result: SearchResult

    retrieval_score: float
    reranker_score: float

    rank_before: int
    rank_after: int


class Reranker:
    """
    Rerank retrieval candidates using a CrossEncoder model.

    Unlike the Retriever, which compares embeddings, the CrossEncoder
    jointly reads the query and the retrieved chunk to estimate
    semantic relevance.
    """

    def __init__(
        self,
        config: RerankerConfig | None = None,
    ) -> None:
        self.config = config or RerankerConfig()
        self._model: CrossEncoder | None = None

    def load(self) -> None:
        """
        Lazily load the CrossEncoder model.
        """
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
        """
        Return the loaded CrossEncoder model.
        """
        self.load()
        assert self._model is not None
        return self._model

    def score(
        self,
        query: str,
        results: Sequence[SearchResult],
    ) -> list[float]:
        """
        Score retrieval candidates using the CrossEncoder.

        Parameters
        ----------
        query
            User query.

        results
            Retrieval candidates returned by the Retriever.

        Returns
        -------
        list[float]
            CrossEncoder relevance scores.
        """

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
    def rerank(self, query: str, results: Sequence[SearchResult],) -> list[RankedResult]:
        """
        Rerank retrieval candidates using the CrossEncoder.

        Parameters
        ----------
        query
            User query.

        results
            Top-K retrieval candidates from the Retriever.

        Returns
        -------
        list[RankedResult]
            Results sorted by CrossEncoder score.
        """

        if not results:
            return []

        reranker_scores = self.score(query, results)

        ranked: list[RankedResult] = []

        for original_rank, (result, reranker_score) in enumerate(
            zip(results, reranker_scores, strict=True),
            start=1,
        ):
            ranked.append(
                RankedResult(
                    result=result,
                    retrieval_score=result.score,
                    reranker_score=reranker_score,
                    rank_before=original_rank,
                    rank_after=0,
                )
            )

        ranked.sort(
            key=lambda candidate: candidate.reranker_score,
            reverse=True,
        )

        final_results: list[RankedResult] = []

        for new_rank, candidate in enumerate(
            ranked[: self.config.top_k_after_reranking],
            start=1,
        ):
            final_results.append(
                RankedResult(
                    result=candidate.result,
                    retrieval_score=candidate.retrieval_score,
                    reranker_score=candidate.reranker_score,
                    rank_before=candidate.rank_before,
                    rank_after=new_rank,
                )
            )

        LOGGER.info(
            "Reranked %d candidates into top %d results",
            len(results),
            len(final_results),
        )

        return final_results

def print_results(results: Sequence[RankedResult]) -> None:
    """
    Print reranked retrieval results.
    """
    for result in results:
        chunk = result.result.chunk

        preview = " ".join(chunk.text.split())[:400]

        print("-" * 80)
        print(f"Old Rank        : {result.rank_before}")
        print(f"New Rank        : {result.rank_after}")
        print(f"Retrieval Score : {result.retrieval_score:.4f}")
        print(f"Reranker Score  : {result.reranker_score:.4f}")
        print(f"Ticker          : {chunk.ticker}")
        print(f"Filing Date     : {chunk.filing_date}")
        print(f"Section         : {chunk.section_name}")
        print(f"Chunk ID        : {chunk.chunk_id}")
        print()
        print(preview)
        print()

def save_results(
    results: Sequence[RankedResult],
    output: Path,
) -> None:
    """
    Save reranked results to JSON.
    """

    payload = []

    for result in results:
        payload.append(
            {
                "rank_before": result.rank_before,
                "rank_after": result.rank_after,
                "retrieval_score": result.retrieval_score,
                "reranker_score": result.reranker_score,
                "chunk": asdict(result.result.chunk),
                "metadata": result.result.metadata,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )