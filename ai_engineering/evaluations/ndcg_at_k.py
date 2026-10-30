"""Normalized discounted cumulative gain at a retrieval cutoff for a single query."""

import heapq
import math
from collections.abc import Callable, Hashable, Iterable, Mapping
from itertools import islice
from typing import Literal, TypeVar

from ._validation import check_cutoff, check_grade

DocumentId = TypeVar("DocumentId", bound=Hashable)
Gain = Literal["linear", "exponential"]

_GAINS: dict[str, Callable[[float], float]] = {
    "linear": lambda grade: grade,
    "exponential": lambda grade: 2.0**grade - 1.0,
}


def ndcg_at_k(
    relevance: Mapping[DocumentId, float],
    retrieved_ids: Iterable[DocumentId],
    k: int,
    *,
    gain: Gain = "linear",
) -> float:
    """Return DCG@k of the retrieval divided by the DCG@k of an ideal ranking.

    ``relevance`` maps judged IDs to graded relevance (for example 0-3). Unjudged
    IDs have zero gain. Position i (1-based) is discounted by log2(i + 1).
    ``gain="linear"`` uses the grade itself (trec_eval); ``"exponential"`` uses
    2**grade - 1, which rewards highly relevant documents more strongly.
    Duplicates occupy ranking positions but earn gain only at their first
    occurrence. Returns 0.0 if no judged ID has a positive grade. Only the first
    k results are consumed.

    Raises:
        TypeError: If k is not an integer or a grade is not a real number.
        ValueError: If k is not positive, a grade is negative or non-finite, or
            gain is unknown.
    """
    check_cutoff(k)
    if gain not in _GAINS:
        raise ValueError(f"gain must be one of {sorted(_GAINS)}")
    gain_of = _GAINS[gain]

    grades = {}
    for doc_id, value in relevance.items():
        grade = check_grade(value)
        if grade > 0:
            grades[doc_id] = grade
    if not grades:
        return 0.0

    ideal_dcg = _dcg(heapq.nlargest(k, grades.values()), gain_of)

    seen = set()
    ranked_grades = []
    for doc_id in islice(retrieved_ids, k):
        ranked_grades.append(0.0 if doc_id in seen else grades.get(doc_id, 0.0))
        seen.add(doc_id)
    return _dcg(ranked_grades, gain_of) / ideal_dcg


def _dcg(grades: Iterable[float], gain_of: Callable[[float], float]) -> float:
    return math.fsum(
        gain_of(grade) / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1)
    )
