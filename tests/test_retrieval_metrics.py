
import pytest
from types import SimpleNamespace

from src.evaluation.retrieval_metrics import (
    EvaluationError,
    evaluate_retrieval,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def make_result(chunk_id):
    """Create a minimal result with the structure metrics expect."""
    return SimpleNamespace(
        chunk=SimpleNamespace(chunk_id=chunk_id)
    )


@pytest.fixture
def results():
    return [
        make_result("A"),  # relevant, rank 1
        make_result("B"),  # irrelevant, rank 2
        make_result("C"),  # relevant, rank 3
        make_result("D"),  # irrelevant, rank 4
        make_result("E"),  # irrelevant, rank 5
        make_result("F"),  # relevant, rank 6
    ]


def test_recall_at_k(results):
    assert recall_at_k(results, {"A", "C", "F"}, 5) == pytest.approx(2 / 3)


def test_precision_at_k(results):
    assert precision_at_k(results, {"A", "C", "F"}, 5) == pytest.approx(2 / 5)


def test_reciprocal_rank_uses_k_cutoff(results):
    assert reciprocal_rank(results, {"F"}, 5) == 0.0
    assert reciprocal_rank(results, {"F"}, 6) == pytest.approx(1 / 6)


def test_empty_results_score_zero():
    assert recall_at_k([], {"A"}, 5) == 0.0
    assert precision_at_k([], {"A"}, 5) == 0.0
    assert reciprocal_rank([], {"A"}, 5) == 0.0


def test_empty_relevant_ids_raise_error(results):
    with pytest.raises(EvaluationError):
        recall_at_k(results, set(), 5)


def test_invalid_k_raises_error(results):
    with pytest.raises(ValueError):
        precision_at_k(results, {"A"}, 0)


def test_evaluate_retrieval_uses_same_cutoff(results):
    evaluation = evaluate_retrieval(
        query="test query",
        results=results,
        relevant_chunk_ids={"A", "C", "F"},
        k=5,
    )

    assert evaluation.k == 5
    assert evaluation.recall_at_k == pytest.approx(2 / 3)
    assert evaluation.precision_at_k == pytest.approx(2 / 5)
    assert evaluation.reciprocal_rank == 1.0
