"""
FinSight retrieval benchmark.

Compares:
    1. Dense retrieval
    2. BM25 retrieval
    3. Hybrid retrieval
    4. Hybrid + CrossEncoder reranking
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Metrics for one retrieval strategy."""

    strategy: str
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float


def _evaluate(
    query: str,
    results: list,
    relevant_ids: set[str],
    k: int,
) -> tuple[float, float, float]:

    normalized_results = []

    for result in results:
        if hasattr(result, "result"):
            normalized_results.append(result.result)
        else:
            normalized_results.append(result)

    metrics = evaluate_retrieval(
        query=query,
        results=normalized_results,
        relevant_chunk_ids=relevant_ids,
        k=k,
    )

    return (
        metrics.recall_at_k,
        metrics.precision_at_k,
        metrics.reciprocal_rank,
    )

def run_benchmark(
    benchmark_path: Path,
    k: int = 5,
) -> list[BenchmarkResult]:
    """Run the retrieval benchmark across all evaluation queries."""

    queries: list[dict[str, Any]] = json.loads(
        benchmark_path.read_text(encoding="utf-8")
    )

    dense = DenseRetriever()
    hybrid = HybridRetriever()
    reranker = Reranker()

    hybrid.load()

    # Build BM25 over the same corpus used by the vector store.
    from src.retrieval.vectorstore import get_all_chunks

    assert hybrid._vector_store is not None

    chunks = get_all_chunks(hybrid._vector_store)
    bm25 = BM25Retriever(chunks)

    accumulated: dict[str, list[float]] = {
        "Dense": [],
        "BM25": [],
        "Hybrid": [],
        "Hybrid + Reranker": [],
    }

    results: list[BenchmarkResult] = []

    for item in queries:
        query = item["query"]
        relevant_ids = set(item["relevant_chunk_ids"])

        dense_results = dense.retrieve_from_text(query, top_k=k)
        bm25_results = [
            result
            for result in bm25.search(query, top_k=k)
        ]

        # BM25Result does not use the same SearchResult type,
        # so only evaluate its shared chunk identity.
        bm25_search_results = [
            type(dense_results[0])(
                chunk=result.chunk,
                score=result.score,
                metadata=dict(result.chunk.metadata),
            )
            for result in bm25_results
        ] if dense_results else []

        hybrid_results = hybrid.retrieve_from_text(query)

        reranked_results = reranker.rerank(
            query,
            hybrid_results,
        )

        strategies = {
            "Dense": dense_results,
            "BM25": bm25_search_results,
            "Hybrid": hybrid_results,
            "Hybrid + Reranker": reranked_results,
        }

        for strategy, strategy_results in strategies.items():
            recall, precision, mrr = _evaluate(
                query,
                strategy_results,
                relevant_ids,
                k,
            )

            results.append(
                BenchmarkResult(
                    strategy=strategy,
                    recall_at_k=recall,
                    precision_at_k=precision,
                    reciprocal_rank=mrr,
                )
            )

    return results


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print average retrieval metrics grouped by strategy."""

    strategies = [
        "Dense",
        "BM25",
        "Hybrid",
        "Hybrid + Reranker",
    ]

    print("\nFinSight Retrieval Benchmark")
    print("=" * 72)

    for strategy in strategies:
        rows = [
            result
            for result in results
            if result.strategy == strategy
        ]

        if not rows:
            continue

        recall = sum(r.recall_at_k for r in rows) / len(rows)
        precision = sum(r.precision_at_k for r in rows) / len(rows)
        mrr = sum(r.reciprocal_rank for r in rows) / len(rows)

        print(
            f"{strategy:<22}"
            f"Recall@K: {recall:.3f}   "
            f"Precision@K: {precision:.3f}   "
            f"MRR: {mrr:.3f}"
        )

def save_report(
    results: list[BenchmarkResult],
    output: Path,
) -> None:
    """Save aggregate retrieval metrics as JSON."""

    strategies = [
        "Dense",
        "BM25",
        "Hybrid",
        "Hybrid + Reranker",
    ]

    report = {
        "query_count": len(
            {
                # Each strategy is evaluated once per query.
                # The total number of rows is therefore divided by
                # the number of strategies.
                result.strategy
                for result in results
            }
        ),
        "k": 5,
        "strategies": {},
    }

    # Number of benchmark queries.
    if strategies:
        report["query_count"] = len(results) // len(strategies)

    for strategy in strategies:
        rows = [
            result
            for result in results
            if result.strategy == strategy
        ]

        if not rows:
            continue

        report["strategies"][strategy] = {
            "recall_at_k": sum(
                row.recall_at_k for row in rows
            ) / len(rows),
            "precision_at_k": sum(
                row.precision_at_k for row in rows
            ) / len(rows),
            "mrr": sum(
                row.reciprocal_rank for row in rows
            ) / len(rows),
        }

    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    benchmark_results = run_benchmark(
        Path("data/evaluation_queries.json"),
        k=5,
    )

    print_summary(benchmark_results)

    save_report(
        benchmark_results,
        Path("results/retrieval_benchmark.json"),
    )

    print(
        "\nSaved benchmark report to "
        "results/retrieval_benchmark.json"
    )