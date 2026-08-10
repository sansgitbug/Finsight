"""
End-to-end grounded financial question answering for FinSight.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from src.generation.llm import LLM, LLMConfig, LLMError
from src.generation.prompt import PromptBuilder, PromptConfig, PromptError
from src.retrieval.dense import DenseRetriever, DenseRetrieverConfig
from src.retrieval.hybrid import (
    HybridRetriever,
    HybridRetrieverConfig,
    HybridRetrievalError,
)
from src.retrieval.reranker import (
    RankedResult,
    Reranker,
    RerankerConfig,
    RerankerError,
)

LOGGER = logging.getLogger(__name__)


class GenerationError(Exception):
    """Raised when end-to-end answer generation fails."""


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    """Configuration for the complete FinSight QA pipeline."""

    dense_top_k: int = 20
    sparse_top_k: int = 20
    hybrid_top_k: int = 10
    reranker_top_k: int = 5

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_device: str = "cpu"

    llm_model: str = "qwen2.5:7b-instruct"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    max_context_chunks: int = 5

    def __post_init__(self) -> None:
        if self.dense_top_k <= 0:
            raise ValueError("dense_top_k must be positive")

        if self.sparse_top_k <= 0:
            raise ValueError("sparse_top_k must be positive")

        if self.hybrid_top_k <= 0:
            raise ValueError("hybrid_top_k must be positive")

        if self.reranker_top_k <= 0:
            raise ValueError("reranker_top_k must be positive")

        if self.max_context_chunks <= 0:
            raise ValueError("max_context_chunks must be positive")


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """Final answer together with the evidence used to generate it."""

    answer: str
    sources: tuple[RankedResult, ...]


class FinSightGenerator:
    """
    End-to-end FinSight financial QA pipeline.

    Pipeline:

        Query
          ↓
        Hybrid retrieval
          ↓
        CrossEncoder reranking
          ↓
        Grounded prompt construction
          ↓
        Local Qwen LLM
          ↓
        Answer + source evidence
    """

    def __init__(
        self,
        config: GeneratorConfig | None = None,
        hybrid_retriever: HybridRetriever | None = None,
        reranker: Reranker | None = None,
        prompt_builder: PromptBuilder | None = None,
        llm: LLM | None = None,
    ) -> None:
        self.config = config or GeneratorConfig()

        self._hybrid = hybrid_retriever
        self._reranker = reranker
        self._prompt_builder = prompt_builder
        self._llm = llm

    def load(self) -> "FinSightGenerator":
        """Load all generation pipeline components."""

        if self._hybrid is None:
            self._hybrid = HybridRetriever(
                HybridRetrieverConfig(
                    top_k_dense=self.config.dense_top_k,
                    top_k_sparse=self.config.sparse_top_k,
                    top_k_final=self.config.hybrid_top_k,
                    embedding_model=self.config.embedding_model,
                    device=self.config.embedding_device,
                )
            )

        self._hybrid.load()

        if self._reranker is None:
            self._reranker = Reranker(
                RerankerConfig(
                    model_name=self.config.reranker_model,
                    device=self.config.reranker_device,
                    top_k_after_reranking=self.config.reranker_top_k,
                )
            )

        if self._prompt_builder is None:
            self._prompt_builder = PromptBuilder(
                PromptConfig(
                    max_context_chunks=self.config.max_context_chunks,
                )
            )

        if self._llm is None:
            self._llm = LLM(
                LLMConfig(
                    model_name=self.config.llm_model,
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.llm_max_tokens,
                )
            )

        LOGGER.info("FinSight generation pipeline loaded")

        return self

    def retrieve_and_rerank(
        self,
        query: str,
        ticker: str | None = None,
    ) -> list[RankedResult]:
            """Retrieve and rerank evidence for a query."""

            if not query or not query.strip():
                raise ValueError("query must be non-empty")

            self.load()

            assert self._hybrid is not None
            assert self._reranker is not None

            normalized_ticker = (
                ticker.strip().upper()
                if ticker
                else None
            )

            try:
                candidates = self._hybrid.retrieve_from_text(
                    query,
                    ticker=ticker,
                )

                explanations = self._hybrid.explain(
                    query,
                    ticker=normalized_ticker,
                )

                ranked = self._reranker.rerank(
                    query,
                    candidates,
                    explanations=explanations,
                )

            except (
                HybridRetrievalError,
                RerankerError,
            ) as exc:
                raise GenerationError(
                    f"Retrieval/reranking failed: {exc}"
                ) from exc

            LOGGER.info(
                "Retrieved %d candidates and retained %d reranked sources%s",
                len(candidates),
                len(ranked),
                f" for {normalized_ticker}"
                if normalized_ticker
                else "",
            )

            return ranked

    def generate(
        self,
        query: str,
        ticker: str | None = None,
    ) -> GeneratedAnswer:
        """
        Run the complete FinSight pipeline for one user query.
        """

        if not query or not query.strip():
            raise ValueError("query must be non-empty")

        self.load()

        assert self._prompt_builder is not None
        assert self._llm is not None

        ranked = self.retrieve_and_rerank(
            query,
            ticker=ticker,
        )

        if not ranked:
            company_text = (
                f" for ticker {ticker.upper()}"
                if ticker
                else ""
            )

            raise GenerationError(
                f"No relevant financial evidence was retrieved{company_text}"
            )

        try:
            prompt = self._prompt_builder.build(
                query,
                ranked,
            )

            answer = self._llm.generate(prompt)

        except (
            PromptError,
            LLMError,
        ) as exc:
            raise GenerationError(
                f"Answer generation failed: {exc}"
            ) from exc

        LOGGER.info(
            "Generated grounded answer using %d sources%s",
            len(ranked),
            f" for {ticker.upper()}"
            if ticker
            else "",
        )

        return GeneratedAnswer(
            answer=answer,
            sources=tuple(ranked),
        )

__all__ = [
    "FinSightGenerator",
    "GeneratedAnswer",
    "GenerationError",
    "GeneratorConfig",
]