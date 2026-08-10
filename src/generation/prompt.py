"""
Prompt construction for grounded FinSight financial question answering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.retrieval.reranker import RankedResult


class PromptError(Exception):
    """Raised when prompt construction fails."""


@dataclass(frozen=True, slots=True)
class PromptConfig:
    """Configuration for grounded financial prompts."""

    max_context_chunks: int = 5
    max_chunk_characters: int = 5000

    def __post_init__(self) -> None:
        if self.max_context_chunks <= 0:
            raise ValueError("max_context_chunks must be greater than zero")

        if self.max_chunk_characters <= 0:
            raise ValueError(
                "max_chunk_characters must be greater than zero"
            )


class PromptBuilder:
    """
    Builds grounded prompts from reranked SEC filing chunks.

    The model is explicitly instructed to answer only from the supplied
    context and to identify when the context does not contain enough evidence.
    """

    def __init__(
        self,
        config: PromptConfig | None = None,
    ) -> None:
        self.config = config or PromptConfig()

    def build(
        self,
        query: str,
        results: Sequence[RankedResult],
    ) -> str:
        """
        Build a grounded financial QA prompt.
        """

        if not query or not query.strip():
            raise ValueError("query must be non-empty")

        if not results:
            raise PromptError(
                "Cannot build a grounded prompt without retrieval results"
            )

        selected = results[: self.config.max_context_chunks]

        context_blocks: list[str] = []

        for index, ranked in enumerate(selected, start=1):
            chunk = ranked.result.chunk

            text = chunk.text.strip()

            if not text:
                continue

            text = text[: self.config.max_chunk_characters]

            context_blocks.append(
                f"""SOURCE {index}
Ticker: {chunk.ticker}
Filing date: {chunk.filing_date}
Filing type: {chunk.filing_type}
Section: {chunk.section_name}
Chunk ID: {chunk.chunk_id}
Reranker score: {ranked.reranker_score:.4f}

{text}
"""
            )

        if not context_blocks:
            raise PromptError(
                "Retrieved results contained no usable text"
            )

        context = "\n".join(context_blocks)

        return f"""You are FinSight, a financial research assistant.

Answer the user's question using ONLY the financial filing context
provided below.

Rules:
1. Do not invent facts, numbers, dates, or explanations.
2. Do not use outside knowledge.
3. If the supplied context does not contain enough information to answer,
   say that the available filing context is insufficient.
4. Distinguish clearly between facts stated in the filings and conclusions
   that can reasonably be drawn from those facts.
5. When using information from a source, cite it using its chunk ID.
6. Prefer precise financial terminology.
7. Keep the answer concise but sufficiently detailed to be useful.

FINANCIAL FILING CONTEXT
========================

{context}

USER QUESTION
=============

{query.strip()}

ANSWER
======
"""


__all__ = [
    "PromptBuilder",
    "PromptConfig",
    "PromptError",
]