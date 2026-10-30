"""Shared input checks for evaluation metrics."""

import math
from numbers import Real


def check_cutoff(k: int) -> None:
    """Raise if k is not a positive integer (booleans are rejected)."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    if k <= 0:
        raise ValueError("k must be greater than zero")


def check_finite(value: object, name: str) -> float:
    """Return value as a float, raising if it is not a finite real number."""
    # Exact float/int is the common case; testing it first avoids the slow ABC check
    # in isinstance(value, Real), which dominates when scoring millions of results.
    # bool is a different exact type, so it still falls through to the rejection.
    if type(value) is float or type(value) is int:
        number = float(value)
    elif isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    else:
        number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def check_grade(value: object) -> float:
    """Return a relevance grade, raising if it is not a finite non-negative number."""
    grade = check_finite(value, "relevance grade")
    if grade < 0:
        raise ValueError("relevance grade must not be negative")
    return grade
