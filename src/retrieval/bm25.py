"""
Sparse BM25 retrieval for FinSight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from rank_bm25 import BM25Okapi

from src.preprocessing.chunker import Chunk


@dataclass(slots=True, frozen=True)
class BM25Result:
    """One BM25 retrieval result."""

    chunk: Chunk
    score: float


class BM25Retriever:
    """
    Sparse lexical retriever using BM25.

    Supports optional ticker filtering so retrieval can be
    restricted to a single company's filings.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
    ) -> None:
        self.chunks = list(chunks)

        if not self.chunks:
            raise ValueError(
                "BM25Retriever requires at least one chunk"
            )

        self.corpus = [
            chunk.text.lower().split()
            for chunk in self.chunks
        ]

        self.index = BM25Okapi(self.corpus)

    def search(
        self,
        query: str,
        top_k: int = 10,
        ticker: str | None = None,
    ) -> list[BM25Result]:
        """
        Search the BM25 index.

        If ``ticker`` is provided, only chunks belonging to
        that company's filings are considered.
        """

        if not query or not query.strip():
            raise ValueError(
                "query must be non-empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be positive"
            )

        tokens = query.strip().lower().split()

        # Determine which documents are eligible.
        if ticker:
            ticker_normalized = (
                ticker.strip().upper()
            )

            candidate_indices = [
                index
                for index, chunk in enumerate(
                    self.chunks
                )
                if chunk.ticker.strip().upper()
                == ticker_normalized
            ]
        else:
            candidate_indices = list(
                range(len(self.chunks))
            )

        if not candidate_indices:
            return []

        # BM25 scores are calculated against the complete
        # corpus, then only eligible company documents are
        # considered for the final ranking.
        scores = self.index.get_scores(tokens)

        ranked = sorted(
            (
                (
                    self.chunks[index],
                    float(scores[index]),
                )
                for index in candidate_indices
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )

        return [
            BM25Result(
                chunk=chunk,
                score=score,
            )
            for chunk, score in ranked[:top_k]
        ]


__all__ = [
    "BM25Result",
    "BM25Retriever",
]