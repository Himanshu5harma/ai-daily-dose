"""Behavioral tests for bootstrap intervals and paired significance tests."""

import pytest

from ai_engineering.evaluations import bootstrap_ci, evaluate, paired_randomization_test

SCORES = [0.0, 0.2, 0.5, 0.5, 0.7, 1.0, 1.0, 0.3, 0.8, 0.6]


def test_bootstrap_interval_contains_the_mean():
    interval = bootstrap_ci(SCORES, n_resamples=2_000, seed=7)

    assert interval.mean == pytest.approx(0.56)
    assert interval.low < interval.mean < interval.high
    assert interval.confidence == 0.95


def test_bootstrap_interval_is_reproducible_with_a_seed():
    first = bootstrap_ci(SCORES, n_resamples=500, seed=1)
    second = bootstrap_ci(SCORES, n_resamples=500, seed=1)
    assert first == second


def test_higher_confidence_gives_a_wider_interval():
    narrow = bootstrap_ci(SCORES, confidence=0.8, n_resamples=2_000, seed=3)
    wide = bootstrap_ci(SCORES, confidence=0.99, n_resamples=2_000, seed=3)
    assert wide.low <= narrow.low and narrow.high <= wide.high


def test_constant_scores_give_a_zero_width_interval():
    interval = bootstrap_ci([0.4] * 5, n_resamples=100, seed=0)
    assert (interval.low, interval.mean, interval.high) == pytest.approx((0.4, 0.4, 0.4))


@pytest.mark.parametrize(
    ("scores", "kwargs", "error"),
    [
        ([], {}, ValueError),
        ([float("nan")], {}, ValueError),
        (["0.5"], {}, TypeError),
        ([0.5], {"confidence": 0}, ValueError),
        ([0.5], {"confidence": 1}, ValueError),
        ([0.5], {"n_resamples": 0}, ValueError),
        ([0.5], {"n_resamples": True}, TypeError),
    ],
)
def test_bootstrap_rejects_invalid_input(scores, kwargs, error):
    with pytest.raises(error):
        bootstrap_ci(scores, **kwargs)


def test_identical_systems_are_not_significantly_different():
    scores = dict(enumerate(SCORES))
    result = paired_randomization_test(scores, scores, n_resamples=500, seed=0)

    assert result.mean_difference == 0.0
    assert result.p_value == 1.0


def test_consistently_better_system_is_significant():
    better = {q: 1.0 for q in range(30)}
    worse = {q: 0.0 for q in range(30)}

    result = paired_randomization_test(better, worse, n_resamples=2_000, seed=0)

    assert result.mean_difference == 1.0
    assert result.p_value < 0.01


def test_small_noisy_difference_is_not_significant():
    a = {0: 0.5, 1: 0.6, 2: 0.4, 3: 0.7}
    b = {0: 0.6, 1: 0.5, 2: 0.5, 3: 0.6}

    result = paired_randomization_test(a, b, n_resamples=2_000, seed=0)

    assert result.p_value > 0.5


def test_systems_must_be_scored_on_the_same_queries():
    with pytest.raises(ValueError, match="same queries"):
        paired_randomization_test({"q1": 1.0}, {"q2": 1.0})


def test_empty_paired_scores_are_rejected():
    with pytest.raises(ValueError, match="empty"):
        paired_randomization_test({}, {})


def test_works_on_per_query_scores_from_a_report():
    qrels = {q: {"good": 1} for q in range(20)}
    baseline = evaluate(qrels, {q: ["bad", "good"] for q in range(20)}, ["mrr@2"])
    candidate = evaluate(qrels, {q: ["good", "bad"] for q in range(20)}, ["mrr@2"])

    result = paired_randomization_test(
        candidate.per_query["mrr@2"], baseline.per_query["mrr@2"], n_resamples=2_000, seed=0
    )
    interval = bootstrap_ci(candidate.per_query["mrr@2"].values(), n_resamples=200, seed=0)

    assert result.mean_difference == pytest.approx(0.5)
    assert result.p_value < 0.01
    assert interval.mean == 1.0
