import pytest

from rag_from_scratch.fusion import reciprocal_rank_fusion as rrf


def test_basic_fusion_scores_and_order():
    result = rrf([["a", "b", "c"], ["b", "c", "d"]], k=60)
    assert [d for d, _ in result] == ["b", "c", "a", "d"]
    scores = dict(result)
    assert scores["b"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["d"] == pytest.approx(1 / 63)


def test_agreement_beats_single_first_place():
    # "x" is 2nd in both lists, "a" and "b" are each 1st in only one.
    result = rrf([["a", "x"], ["b", "x"]], k=60)
    assert result[0][0] == "x"


def test_k_controls_top_rank_vs_agreement():
    # "a" is 1st in one list; "x" is 3rd in both.
    rankings = [["a", "y", "x"], ["b", "z", "x"]]
    assert rrf(rankings, k=0)[0][0] == "a"  # 1/1 beats 1/3 + 1/3
    assert rrf(rankings, k=60)[0][0] == "x"  # 2/63 beats 1/61


def test_ties_follow_first_appearance():
    assert [d for d, _ in rrf([["a", "b"], ["b", "a"]])] == ["a", "b"]
    assert [d for d, _ in rrf([["p"], ["q"], ["r"]])] == ["p", "q", "r"]


def test_duplicates_within_a_ranking_count_once():
    result = dict(rrf([["a", "a", "b"]], k=60))
    assert result["a"] == pytest.approx(1 / 61)
    assert result["b"] == pytest.approx(1 / 63)  # the duplicate still occupied rank 2


def test_works_with_any_hashable_ids():
    assert [d for d, _ in rrf([[3, 1], [1, 2]])] == [1, 3, 2]


def test_top_n_and_empty_inputs():
    assert len(rrf([["a", "b", "c"]], top_n=2)) == 2
    assert rrf([]) == []
    assert rrf([[], []]) == []


@pytest.mark.parametrize("kwargs", [{"k": -1}, {"top_n": 0}, {"top_n": -3}])
def test_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        rrf([["a"]], **kwargs)
