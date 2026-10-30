"""Behavioral tests for graded-relevance NDCG and cutoff semantics."""

from math import log2

import pytest

from ai_engineering.evaluations import ndcg_at_k

GRADES = {"a": 3, "b": 2, "c": 1}


@pytest.mark.parametrize(
    ("relevance", "retrieved", "k", "expected"),
    [
        (GRADES, ["a", "b", "c"], 3, 1.0),
        (GRADES, ["c", "b", "a"], 3, (1 + 2 / log2(3) + 3 / 2) / (3 + 2 / log2(3) + 1 / 2)),
        (GRADES, ["b"], 1, 2 / 3),
        ({"a": 1}, ["x", "a"], 2, 1 / log2(3)),
        ({"a": 1, "b": 1}, ["a", "a", "b"], 3, (1 + 1 / 2) / (1 + 1 / log2(3))),
        ({"a": 1, "b": 1, "c": 1}, ["a"], 1, 1.0),
        ({"a": 1}, ["x", "y"], 2, 0.0),
        ({"a": 1}, [], 3, 0.0),
        ({"a": 0, "b": 0}, ["a", "b"], 2, 0.0),
        ({}, ["a"], 1, 0.0),
        ({"a": 2.5, "b": 0.5}, ["a", "b"], 2, 1.0),
    ],
)
def test_ndcg_scores(relevance, retrieved, k, expected):
    assert ndcg_at_k(relevance, retrieved, k) == pytest.approx(expected)


def test_exponential_gain_rewards_highly_relevant_documents_more():
    expected = (1 + 3 / log2(3) + 7 / 2) / (7 + 3 / log2(3) + 1 / 2)
    assert ndcg_at_k(GRADES, ["c", "b", "a"], 3, gain="exponential") == pytest.approx(expected)
    assert ndcg_at_k(GRADES, ["c", "b", "a"], 3, gain="exponential") < ndcg_at_k(
        GRADES, ["c", "b", "a"], 3
    )


def test_unknown_gain_is_rejected():
    with pytest.raises(ValueError, match="gain"):
        ndcg_at_k(GRADES, ["a"], 1, gain="quadratic")


@pytest.mark.parametrize("grade", [-1, float("nan"), float("inf")])
def test_invalid_grade_values_are_rejected(grade):
    with pytest.raises(ValueError, match="grade"):
        ndcg_at_k({"a": grade}, ["a"], 1)


@pytest.mark.parametrize("grade", [True, "2", None])
def test_nonnumeric_grades_are_rejected(grade):
    with pytest.raises(TypeError, match="real number"):
        ndcg_at_k({"a": grade}, ["a"], 1)


@pytest.mark.parametrize("k", [0, -1])
def test_nonpositive_cutoff_is_rejected(k):
    with pytest.raises(ValueError, match="greater than zero"):
        ndcg_at_k({}, [], k)


@pytest.mark.parametrize("k", [True, 1.5, "2", None])
def test_noninteger_cutoff_is_rejected(k):
    with pytest.raises(TypeError, match="integer"):
        ndcg_at_k({}, [], k)


def test_only_consumes_top_k_results():
    def retrieval():
        yield "a"
        yield "b"
        raise AssertionError("retrieval was consumed past the cutoff")

    assert ndcg_at_k({"a": 1}, retrieval(), 2) == 1.0
