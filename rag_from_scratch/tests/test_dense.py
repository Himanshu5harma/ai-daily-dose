import numpy as np
import pytest

from rag_from_scratch.dense import cosine_top_k

M = np.array(
    [
        [1.0, 0.0],  # 0: same direction as query
        [10.0, 10.0],  # 1: 45 degrees, long vector
        [0.0, 1.0],  # 2: orthogonal
        [-3.0, 0.0],  # 3: opposite
        [0.0, 0.0],  # 4: zero vector
    ]
)
Q = np.array([2.0, 0.0])  # not unit length on purpose


def test_ranks_by_cosine_not_dot_product():
    result = cosine_top_k(Q, M, 5)
    assert [i for i, _ in result] == [0, 1, 2, 4, 3]
    sims = dict(result)
    assert sims[0] == pytest.approx(1.0)
    assert sims[1] == pytest.approx(1 / np.sqrt(2))
    assert sims[2] == pytest.approx(0.0)
    assert sims[3] == pytest.approx(-1.0)


def test_zero_vectors_score_zero_without_nan():
    result = cosine_top_k(Q, M, 5)
    assert dict(result)[4] == 0.0
    assert all(np.isfinite(s) for _, s in cosine_top_k(np.zeros(2), M, 5))


def test_returns_python_types_and_k_truncates():
    result = cosine_top_k(Q, M, 2)
    assert len(result) == 2
    assert all(type(i) is int and type(s) is float for i, s in result)


def test_k_larger_than_n_returns_all():
    assert len(cosine_top_k(Q, M, 50)) == 5


def test_empty_matrix():
    assert cosine_top_k(Q, np.zeros((0, 2)), 3) == []


def test_ties_go_to_lower_index():
    matrix = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [5.0, 0.0]])
    assert [i for i, _ in cosine_top_k(np.array([1.0, 0.0]), matrix, 3)] == [1, 3, 4]


def test_ties_on_larger_matrix_with_partial_k():
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(1000, 8))
    matrix[[17, 400, 999]] = [1.0] * 8  # three identical best rows
    result = cosine_top_k(np.ones(8), matrix, 2)
    assert [i for i, _ in result] == [17, 400]


@pytest.mark.parametrize(
    "query,matrix,k",
    [
        (np.ones(3), np.ones((4, 3)), 0),
        (np.ones((1, 3)), np.ones((4, 3)), 1),
        (np.ones(3), np.ones(3), 1),
        (np.ones(2), np.ones((4, 3)), 1),
    ],
)
def test_rejects_bad_inputs(query, matrix, k):
    with pytest.raises(ValueError):
        cosine_top_k(query, matrix, k)
