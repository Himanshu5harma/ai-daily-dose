"""Behavioral tests for retrieval coverage and cutoff semantics."""

import pytest

from ai_engineering.evaluations import recall_at_k


@pytest.mark.parametrize(
    ("relevant", "retrieved", "k", "expected"),
    [
        ({"a", "c"}, ["a", "b", "c"], 3, 1.0),
        ({"a", "c"}, ["a", "b", "c"], 2, 0.5),
        ({"a"}, ["b", "c"], 2, 0.0),
        ({"a"}, [], 3, 0.0),
        (set(), ["a"], 1, 0.0),
        (set(), [], 1, 0.0),
        ({"a", "c"}, ["a"], 10, 0.5),
        ({"a"}, ["b", "a"], 1, 0.0),
        (["a", "a", "c"], ["a", "c"], 2, 1.0),
        ({"a", "c"}, ["a", "a", "c"], 2, 0.5),
        ({"a", "c"}, ["a", "a", "c"], 3, 1.0),
        ({1, 3}, [1, 2, 3], 2, 0.5),
        ({"a", "b", "c"}, ["a"], 1, 1 / 3),
    ],
)
def test_recall_scores(relevant, retrieved, k, expected):
    assert recall_at_k(relevant, retrieved, k) == pytest.approx(expected)


@pytest.mark.parametrize("k", [0, -1, -10])
def test_nonpositive_cutoff_is_rejected(k):
    with pytest.raises(ValueError, match="greater than zero"):
        recall_at_k([], [], k)


@pytest.mark.parametrize("k", [True, False, 1.5, "2", None])
def test_noninteger_cutoff_is_rejected(k):
    with pytest.raises(TypeError, match="integer"):
        recall_at_k([], [], k)


def test_accepts_generators():
    relevant = (doc for doc in ["a", "c"])
    retrieved = (doc for doc in ["a", "b", "c"])
    assert recall_at_k(relevant, retrieved, 2) == 0.5


def test_only_consumes_top_k_results():
    def retrieval():
        yield "a"
        yield "b"
        raise AssertionError("retrieval was consumed past the cutoff")

    assert recall_at_k({"a", "c"}, retrieval(), 2) == 0.5


def test_empty_relevance_does_not_consume_retrieval():
    retrieved = iter(["a"])
    assert recall_at_k([], retrieved, 1) == 0.0
    assert next(retrieved) == "a"
