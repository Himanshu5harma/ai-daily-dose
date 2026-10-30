"""Reciprocal rank at a retrieval cutoff for a single query."""

from collections.abc import Hashable, Iterable
from itertools import islice
from typing import TypeVar

DocumentId = TypeVar("DocumentId", bound=Hashable)


def reciprocal_rank_at_k(
    relevant_ids: Iterable[DocumentId],
    retrieved_ids: Iterable[DocumentId],
    k: int,
) -> float:
    """Return 1 / rank of the first relevant ID within the first k results.

    Ranks are 1-based. IDs must be hashable. Retrieved IDs must be ordered from
    highest to lowest relevance. A duplicate is placed by its first occurrence,
    and every occurrence still occupies a ranking position. Returns 0.0 if no
    relevant ID appears in the first k results, or if the relevance judgments or
    the retrieval are empty. Only the results up to the first hit (at most k) are
    consumed.

    Raises:
        TypeError: If k is not an integer (booleans are rejected).
        ValueError: If k is zero or negative.
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    if k <= 0:
        raise ValueError("k must be greater than zero")

    relevant = set(relevant_ids)
    if not relevant:
        return 0.0

    for rank, doc_id in enumerate(islice(retrieved_ids, k), start=1):
        if doc_id in relevant:
            return 1 / rank
    return 0.0
