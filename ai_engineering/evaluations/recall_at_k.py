"""Recall at a retrieval cutoff for a single query."""

from collections.abc import Hashable, Iterable
from itertools import islice
from typing import TypeVar

DocumentId = TypeVar("DocumentId", bound=Hashable)


def recall_at_k(
    relevant_ids: Iterable[DocumentId],
    retrieved_ids: Iterable[DocumentId],
    k: int,
) -> float:
    """Return the fraction of unique relevant IDs found in the first k results.

    IDs must be hashable. Retrieved IDs must be ordered from highest to lowest
    relevance. Duplicates occupy ranking positions but count toward recall once.
    Empty relevance judgments or retrievals return 0.0. A cutoff larger than the
    retrieval uses all available results. Only the first k results are consumed.

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
    return len(relevant & retrieved) / len(relevant)
