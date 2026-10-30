"""Statistical checks for comparing retrieval evaluation results."""

import math
import random
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass

from ._validation import check_finite


@dataclass(frozen=True)
class ConfidenceInterval:
    mean: float
    low: float
    high: float
    confidence: float


@dataclass(frozen=True)
class SignificanceResult:
    mean_difference: float
    p_value: float


def bootstrap_ci(
    scores: Iterable[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int | None = None,
) -> ConfidenceInterval:
    """Return a percentile bootstrap confidence interval for the mean score.

    ``scores`` are per-query scores, for example ``report.per_query["ndcg@10"].values()``.
    Pass ``seed`` for reproducible intervals. Cost is O(n_resamples * len(scores)).

    Raises:
        TypeError: If a score is not a real number.
        ValueError: If scores are empty or non-finite, or the settings are invalid.
    """
    values = [check_finite(score, "score") for score in scores]
    if not values:
        raise ValueError("scores must not be empty")
    _check_settings(n_resamples)
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    rng = random.Random(seed)
    n = len(values)
    means = sorted(math.fsum(rng.choices(values, k=n)) / n for _ in range(n_resamples))
    tail = (1 - confidence) / 2
    return ConfidenceInterval(
        mean=math.fsum(values) / n,
        low=_percentile(means, tail),
        high=_percentile(means, 1 - tail),
        confidence=confidence,
    )


def paired_randomization_test(
    scores_a: Mapping[Hashable, float],
    scores_b: Mapping[Hashable, float],
    *,
    n_resamples: int = 10_000,
    seed: int | None = None,
) -> SignificanceResult:
    """Return the mean per-query difference (a - b) and a two-sided p-value.

    Under the null hypothesis the two systems are interchangeable, so the sign of
    each per-query difference is flipped at random. The p-value is the share of
    resamples whose absolute mean difference is at least the observed one, with
    the +1 correction so it is never zero.

    Raises:
        TypeError: If a score is not a real number.
        ValueError: If the systems were scored on different or no queries, a score
            is non-finite, or n_resamples is invalid.
    """
    if scores_a.keys() != scores_b.keys():
        raise ValueError("both systems must be scored on the same queries")
    if not scores_a:
        raise ValueError("scores must not be empty")
    _check_settings(n_resamples)

    diffs = [
        check_finite(scores_a[query], "score") - check_finite(scores_b[query], "score")
        for query in scores_a
    ]
    observed = abs(math.fsum(diffs))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_resamples):
        total = math.fsum(diff if rng.random() < 0.5 else -diff for diff in diffs)
        if abs(total) >= observed - 1e-12:
            extreme += 1
    return SignificanceResult(
        mean_difference=math.fsum(diffs) / len(diffs),
        p_value=(extreme + 1) / (n_resamples + 1),
    )


def _check_settings(n_resamples: int) -> None:
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int):
        raise TypeError("n_resamples must be an integer")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be greater than zero")


def _percentile(sorted_values: list[float], q: float) -> float:
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction
