from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.explainability.shap_explainer import (
    RerankerSHAPExplainer,
    SHAPExplainerError,
)
from src.generation.generator import FinSightGenerator, GenerationError
from src.generation.temporal import TemporalAnalyzer, TemporalAnalysisError


app = FastAPI(
    title="FinSight API",
    version="1.0.0",
    description="Financial RAG API for SEC filing research.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)


class TemporalRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, gt=0)


_generator: FinSightGenerator | None = None
_shap_explainer: RerankerSHAPExplainer | None = None


def get_generator() -> FinSightGenerator:
    global _generator

    if _generator is None:
        _generator = FinSightGenerator()
        _generator.load()

    return _generator


def get_shap_explainer() -> RerankerSHAPExplainer:
    global _shap_explainer

    if _shap_explainer is None:
        generator = get_generator()

        assert generator._reranker is not None

        _shap_explainer = RerankerSHAPExplainer(
            generator._reranker,
            max_tokens=40,
            max_evals=300,
        )

    return _shap_explainer


def merge_wordpiece_tokens(
    attributions: list[dict[str, float]],
) -> list[dict[str, float]]:
    """
    Merge WordPiece fragments such as:

        cyber + ##security

    into:

        cybersecurity
    """

    merged: list[dict[str, float]] = []

    for item in attributions:
        token = item["token"]
        value = item["value"]

        if token in {"[CLS]", "[SEP]", "[PAD]"}:
            continue

        if token.startswith("##") and merged:
            merged[-1]["token"] += token[2:]
            merged[-1]["value"] += value
        else:
            merged.append(
                {
                    "token": token,
                    "value": value,
                }
            )

    return merged


def serialize_shap(explanation: Any) -> dict[str, Any]:
    raw = [
        {
            "token": attribution.token,
            "value": float(attribution.value),
        }
        for attribution in explanation.attributions
    ]

    merged = merge_wordpiece_tokens(raw)

    positive = sorted(
        [x for x in merged if x["value"] > 0],
        key=lambda x: x["value"],
        reverse=True,
    )[:8]

    negative = sorted(
        [x for x in merged if x["value"] < 0],
        key=lambda x: x["value"],
    )[:4]

    return {
        "base_value": explanation.base_value,
        "predicted_score": explanation.predicted_score,
        "positive": positive,
        "negative": negative,
    }


def serialize_ranked_result(
    result: Any,
) -> dict[str, Any]:
    chunk = result.result.chunk

    return {
        "chunk_id": chunk.chunk_id,
        "ticker": chunk.ticker,
        "filing_date": chunk.filing_date,
        "filing_type": chunk.filing_type,
        "section_name": chunk.section_name,

        # Full evidence remains available when the source is expanded.
        "text": chunk.text,

        "reranker_score": result.reranker_score,
        "dense_rank": result.dense_rank,
        "dense_score": result.dense_score,
        "bm25_rank": result.bm25_rank,
        "bm25_score": result.bm25_score,
        "rrf_score": result.rrf_score,
        "rank_before": result.rank_before,
        "rank_after": result.rank_after,
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "finsight",
    }


@app.post("/api/query")
def query_financial_filings(
    request: QueryRequest,
) -> dict[str, Any]:
    try:
        generator = get_generator()

        result = generator.generate(request.query)

        sources = []

        for source in result.sources:
            serialized = serialize_ranked_result(source)

            try:
                explanation = get_shap_explainer().explain(
                    query=request.query,
                    chunk_id=source.result.chunk.chunk_id,
                    chunk_text=source.result.chunk.text,
                )

                serialized["shap"] = serialize_shap(
                    explanation
                )

            except SHAPExplainerError as exc:
                serialized["shap"] = {
                    "base_value": None,
                    "predicted_score": None,
                    "positive": [],
                    "negative": [],
                    "error": str(exc),
                }

            sources.append(serialized)

        return {
            "query": request.query,
            "answer": result.answer,
            "sources": sources,
            "retrieval": {
                "method": "hybrid + CrossEncoder reranker",
                "candidate_count": len(sources),
            },
        }

    except GenerationError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.post("/api/temporal")
def temporal_analysis(
    request: TemporalRequest,
) -> dict[str, Any]:
    try:
        generator = get_generator()

        assert generator._hybrid is not None
        assert generator._reranker is not None
        assert generator._llm is not None

        analyzer = TemporalAnalyzer(
            generator._hybrid,
            generator._reranker,
            generator._llm,
        )

        result = analyzer.analyze(
            request.query,
            top_k=request.top_k,
        )

        evidence = []

        for filing in result.evidence:
            evidence.append(
                {
                    "filing_date": filing.filing_date,
                    "sources": [
                        serialize_ranked_result(source)
                        for source in filing.sources
                    ],
                }
            )

        return {
            "query": result.query,
            "answer": result.answer,
            "evidence": evidence,
        }

    except TemporalAnalysisError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.get("/api/benchmark")
def benchmark() -> dict[str, Any]:
    benchmark_path = Path(
        "results/retrieval_benchmark.json"
    )

    if not benchmark_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Benchmark report not found.",
        )

    try:
        return json.loads(
            benchmark_path.read_text(
                encoding="utf-8"
            )
        )

    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read benchmark report: {exc}",
        ) from exc