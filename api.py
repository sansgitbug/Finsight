from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from src.retrieval.vectorstore import get_all_chunks
from src.ingestion.ingest import ingest_company

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.explainability.shap_explainer import (
    RerankerSHAPExplainer,
    SHAPExplainerError,
)
from src.generation.generator import (
    FinSightGenerator,
    GenerationError,
)
from src.generation.temporal import (
    TemporalAnalyzer,
    TemporalAnalysisError,
)
from pathlib import Path

from src.ingestion.ingest import ingest_company
from src.preprocessing.chunker import chunk_directory
from src.preprocessing.embeddings import embed_chunks, load_chunks
from src.retrieval.vectorstore import (
    VectorStoreConfig,
    add_embeddings,
    load_vector_store,
)

# ---------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------

class QueryRequest(BaseModel):
    ticker: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)

class TemporalRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, gt=0)


# ---------------------------------------------------------------------
# Lazy-loaded components
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# SHAP processing
# ---------------------------------------------------------------------

def merge_wordpiece_tokens(
    attributions: list[dict[str, float]],
) -> list[dict[str, float]]:
    """
    Merge WordPiece fragments.

    Example:
        cyber + ##security
        ->
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


def serialize_shap(
    explanation: Any,
) -> dict[str, Any]:
    raw = [
        {
            "token": attribution.token,
            "value": float(attribution.value),
        }
        for attribution in explanation.attributions
    ]

    merged = merge_wordpiece_tokens(raw)

    positive = sorted(
        [item for item in merged if item["value"] > 0],
        key=lambda item: item["value"],
        reverse=True,
    )[:8]

    negative = sorted(
        [item for item in merged if item["value"] < 0],
        key=lambda item: item["value"],
    )[:4]

    return {
        "base_value": explanation.base_value,
        "predicted_score": explanation.predicted_score,
        "positive": positive,
        "negative": negative,
    }


# ---------------------------------------------------------------------
# Retrieval result serialization
# ---------------------------------------------------------------------

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

        # Full evidence is retained so the frontend can expand it.
        "text": chunk.text,

        "reranker_score": float(result.reranker_score),

        "dense_rank": result.dense_rank,
        "dense_score": float(result.dense_score),

        "bm25_rank": result.bm25_rank,
        "bm25_score": float(result.bm25_score),

        "rrf_score": float(result.rrf_score),

        "rank_before": result.rank_before,
        "rank_after": result.rank_after,
    }


# ---------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "finsight",
    }


# ---------------------------------------------------------------------
# Main financial QA endpoint
# ---------------------------------------------------------------------

@app.post("/api/query")
def query_financial_filings(
    request: QueryRequest,
) -> dict[str, Any]:

    try:
        generator = get_generator()
        result = generator.generate(
            request.query,
            ticker=request.ticker.strip().upper(),
        )
            
            
        

        sources: list[dict[str, Any]] = []

        for source in result.sources:

            serialized = serialize_ranked_result(
                source
            )

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
                "ticker": request.ticker.upper(),
                "method": "hybrid + CrossEncoder reranker",
                "candidate_count": len(sources),
            },
        }

    except GenerationError as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ---------------------------------------------------------------------
# Temporal analysis endpoint
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Benchmark endpoint
# ---------------------------------------------------------------------

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

    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to read benchmark report: {exc}",
        ) from exc


@app.get("/api/companies")
def companies() -> dict[str, Any]:
    generator = get_generator()

    assert generator._hybrid is not None
    assert generator._hybrid._vector_store is not None

    chunks = get_all_chunks(
        generator._hybrid._vector_store
    )

    companies: dict[str, dict[str, Any]] = {}

    for chunk in chunks:
        ticker = chunk.ticker.upper()

        if ticker not in companies:
            companies[ticker] = {
                "ticker": ticker,
                "filings": set(),
                "latest_filing": chunk.filing_date,
            }

        companies[ticker]["filings"].add(
            chunk.filing_date
        )

        if chunk.filing_date > companies[ticker]["latest_filing"]:
            companies[ticker]["latest_filing"] = chunk.filing_date

    result = []

    for company in companies.values():
        result.append(
            {
                "ticker": company["ticker"],
                "filings": len(company["filings"]),
                "latest_filing": company["latest_filing"],
            }
        )

    result.sort(key=lambda x: x["ticker"])

    return {"companies": result}

class IngestRequest(BaseModel):
    ticker: str = Field(..., min_length=1)
@app.post("/api/ingest")
def ingest_ticker(
    request: IngestRequest,
) -> dict[str, Any]:

    ticker = request.ticker.strip().upper()

    try:
        # --------------------------------------------------
        # 1. Download SEC filings
        # --------------------------------------------------

        filings = ingest_company(ticker)

        # --------------------------------------------------
        # 2. Chunk the newly downloaded filings
        # --------------------------------------------------

        data_dir = Path("data")
        chunks_dir = Path("chunks")

        chunk_files = chunk_directory(
            data_dir=data_dir,
            chunks_dir=chunks_dir,
        )

        # --------------------------------------------------
        # 3. Load the generated chunks
        # --------------------------------------------------

        chunks = []

        for chunk_file in chunk_files:
            chunks.extend(
                load_chunks(chunk_file)
            )

        if not chunks:
            raise RuntimeError(
                f"No chunks were produced for {ticker}"
            )

        # --------------------------------------------------
        # 4. Generate embeddings
        # --------------------------------------------------

        embeddings = embed_chunks(chunks)

        if not embeddings:
            raise RuntimeError(
                f"No embeddings were produced for {ticker}"
            )

        # --------------------------------------------------
        # 5. Add embeddings to existing vector store
        # --------------------------------------------------

        store = load_vector_store(
            VectorStoreConfig()
        )

        add_embeddings(
            store,
            embeddings,
        )

        # --------------------------------------------------
        # 6. Reset cached pipeline components
        # --------------------------------------------------

        global _generator

        _generator = None

        return {
            "ticker": ticker,
            "filings_ingested": len(filings),
            "chunks_created": len(chunks),
            "embeddings_created": len(embeddings),
            "filings": [
                {
                    "accession_number": filing.accession_number,
                    "filing_type": filing.filing_type,
                    "filing_date": filing.filing_date,
                }
                for filing in filings
            ],
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to ingest and index "
                f"{ticker}: {exc}"
            ),
        ) from exc