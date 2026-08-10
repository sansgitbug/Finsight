"""
Cross-filing temporal analysis for FinSight.

Uses the existing hybrid retrieval and CrossEncoder reranking pipeline,
then selects topic-relevant evidence across the requested filing years.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from src.generation.llm import LLM, LLMError
from src.retrieval.hybrid import (
    HybridRetriever,
    HybridRetrievalError,
)
from src.retrieval.reranker import (
    RankedResult,
    Reranker,
    RerankerError,
)


class TemporalAnalysisError(Exception):
    """Raised when cross-filing analysis fails."""


@dataclass(frozen=True, slots=True)
class FilingEvidence:
    """Reranked evidence associated with one filing date."""

    filing_date: str
    sources: tuple[RankedResult, ...]


@dataclass(frozen=True, slots=True)
class TemporalAnalysis:
    """Result of a cross-filing comparison."""

    query: str
    evidence: tuple[FilingEvidence, ...]
    answer: str


class TemporalAnalyzer:
    """Compare financial evidence across filing periods."""

    def __init__(
        self,
        hybrid: HybridRetriever,
        reranker: Reranker,
        llm: LLM,
        max_sources_per_filing: int = 3,
    ) -> None:
        if max_sources_per_filing <= 0:
            raise ValueError(
                "max_sources_per_filing must be positive"
            )

        self._hybrid = hybrid
        self._reranker = reranker
        self._llm = llm
        self.max_sources_per_filing = max_sources_per_filing

    def analyze(
        self,
        query: str,
        top_k: int = 10,
    ) -> TemporalAnalysis:
        """Retrieve evidence and compare it across filing periods."""

        if not query or not query.strip():
            raise ValueError("query must be non-empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        try:
            candidates = self._hybrid.retrieve_from_text(query)

            explanations = self._hybrid.explain(query)

            # IMPORTANT:
            # Temporal analysis needs the entire hybrid candidate set.
            # Normal FinSight still defaults to the reranker's configured
            # top-k (currently 5).
            ranked = self._reranker.rerank(
                query,
                candidates,
                explanations=explanations,
                top_k=len(candidates),
            )

        except (
            HybridRetrievalError,
            RerankerError,
        ) as exc:
            raise TemporalAnalysisError(
                f"Temporal retrieval failed: {exc}"
            ) from exc

        if not ranked:
            raise TemporalAnalysisError(
                "No relevant filing evidence was retrieved"
            )

        selected = self._select_temporal_evidence(
            query,
            ranked,
            top_k,
        )

        grouped: dict[str, list[RankedResult]] = {}

        for result in selected:
            filing_date = result.result.chunk.filing_date

            grouped.setdefault(
                filing_date,
                [],
            ).append(result)

        evidence = tuple(
            FilingEvidence(
                filing_date=filing_date,
                sources=tuple(
                    sources[: self.max_sources_per_filing]
                ),
            )
            for filing_date, sources in sorted(
                grouped.items()
            )
        )

        prompt = self._build_comparison_prompt(
            query,
            evidence,
        )

        try:
            answer = self._llm.generate(prompt)
        except LLMError as exc:
            raise TemporalAnalysisError(
                f"Temporal answer generation failed: {exc}"
            ) from exc

        return TemporalAnalysis(
            query=query,
            evidence=evidence,
            answer=answer,
        )

    @staticmethod
    def _topic_sections(
        query: str,
    ) -> set[str]:
        """Determine filing sections relevant to the question."""

        query_lower = query.lower()

        sections: set[str] = set()

        if any(
            term in query_lower
            for term in (
                "risk",
                "risks",
                "cybersecurity",
                "cyber",
                "security",
            )
        ):
            sections.add("Risk Factors")

        if any(
            term in query_lower
            for term in (
                "legal",
                "lawsuit",
                "litigation",
                "regulatory",
            )
        ):
            sections.add("Legal Proceedings")

        if any(
            term in query_lower
            for term in (
                "revenue",
                "sales",
                "profit",
                "margin",
                "income",
                "expenses",
            )
        ):
            sections.add(
                "Management's Discussion and Analysis"
            )

        return sections

    @classmethod
    def _select_temporal_evidence(
        cls,
        query: str,
        ranked: Sequence[RankedResult],
        top_k: int,
    ) -> list[RankedResult]:
        """
        Select topic-relevant evidence while guaranteeing coverage
        of explicitly requested years when evidence exists.
        """

        topic_sections = cls._topic_sections(query)
        requested_years = cls._extract_requested_years(query)

        if topic_sections:
            candidates = [
                result
                for result in ranked
                if result.result.chunk.section_name
                in topic_sections
            ]
        else:
            candidates = list(ranked)

        # If section filtering finds nothing, retain the full candidate set.
        if not candidates:
            candidates = list(ranked)

        candidates = sorted(
            candidates,
            key=lambda result: result.reranker_score,
            reverse=True,
        )

        selected: list[RankedResult] = []
        selected_ids: set[str] = set()

        # First: guarantee evidence from every requested year.
        for year in requested_years:
            year_candidates = [
                result
                for result in candidates
                if result.result.chunk.filing_date.startswith(year)
            ]

            for result in year_candidates[
                : cls._sources_per_year()
            ]:
                chunk_id = result.result.chunk.chunk_id

                if chunk_id in selected_ids:
                    continue

                selected.append(result)
                selected_ids.add(chunk_id)

        # Second: fill remaining slots with strongest relevant evidence.
        for result in candidates:
            if len(selected) >= top_k:
                break

            chunk_id = result.result.chunk.chunk_id

            if chunk_id in selected_ids:
                continue

            selected.append(result)
            selected_ids.add(chunk_id)

        return selected[:top_k]

    @staticmethod
    def _sources_per_year() -> int:
        """Maximum initial evidence items selected for each year."""

        return 3

    @staticmethod
    def _extract_requested_years(
        query: str,
    ) -> tuple[str, ...]:
        """Extract explicitly mentioned four-digit years."""

        return tuple(
            dict.fromkeys(
                re.findall(
                    r"\b20\d{2}\b",
                    query,
                )
            )
        )

    @classmethod
    def _build_comparison_prompt(
        cls,
        query: str,
        evidence: Sequence[FilingEvidence],
    ) -> str:
        """Build a grounded temporal comparison prompt."""

        requested_years = cls._extract_requested_years(query)

        comparison_period = (
            " vs ".join(requested_years)
            if requested_years
            else "the available filing periods"
        )

        sections: list[str] = []

        for filing in evidence:
            chunks: list[str] = []

            for source in filing.sources:
                chunk = source.result.chunk

                chunks.append(
                    f"""
Chunk ID: {chunk.chunk_id}
Ticker: {chunk.ticker}
Filing date: {chunk.filing_date}
Filing type: {chunk.filing_type}
Section: {chunk.section_name}
CrossEncoder score: {source.reranker_score:.4f}

{chunk.text.strip()}
"""
                )

            sections.append(
                f"""
===== FILING DATE: {filing.filing_date} =====
{"".join(chunks)}
"""
            )

        context = "\n".join(sections)

        return f"""
You are FinSight, a financial research assistant.

The user is asking for a temporal comparison.

USER QUESTION:
{query.strip()}

REQUESTED COMPARISON:
{comparison_period}

Use ONLY the supplied filing evidence.

RULES:

1. Directly answer the user's question.
2. Compare the requested periods.
3. Prioritize evidence directly related to the subject of the question.
4. Do not substitute an unrelated financial metric or topic.
5. Do not invent facts.
6. Every important claim must identify its filing date and chunk ID.
7. Clearly distinguish explicit filing statements from inference.
8. If evidence is insufficient for a particular period, say so.
9. Do not answer with a summary of only the latest filing.
10. If the supplied evidence supports a genuine change, explain what
    changed and cite the evidence supporting the change.

FILING EVIDENCE:
{context}

TEMPORAL COMPARISON:
"""

__all__ = [
    "FilingEvidence",
    "TemporalAnalysis",
    "TemporalAnalysisError",
    "TemporalAnalyzer",
]