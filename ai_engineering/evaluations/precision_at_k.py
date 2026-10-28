"""Precision at a retrieval cutoff for a single query."""

from collections.abc import Hashable, Iterable
from itertools import islice
from typing import TypeVar

DocumentId = TypeVar("DocumentId", bound=Hashable)


def precision_at_k(
    relevant_ids: Iterable[DocumentId],
    retrieved_ids: Iterable[DocumentId],
    k: int,
) -> float:
    """Return the number of unique relevant IDs in the first k results, divided by k.

    IDs must be hashable. Retrieved IDs must be ordered from highest to lowest
    relevance. The denominator is always k, so a retrieval shorter than k is
    penalized for the missing slots. Duplicates occupy ranking positions but
    count toward precision once. Empty relevance judgments or retrievals return
    0.0. Only the first k results are consumed.

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

    retrieved = set(islice(retrieved_ids, k))
    return len(relevant & retrieved) / k
