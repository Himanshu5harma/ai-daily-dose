"""Hybrid retrieval by reciprocal rank fusion. YOU implement this (checkpoint 5)."""

from collections.abc import Hashable, Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hashable]],
    k: int = 60,
    top_n: int | None = None,
) -> list[tuple[Hashable, float]]:
    """Fuse several best-first rankings of IDs into one.

    ``score(d) = sum over rankings that contain d of 1 / (k + rank(d))``, with
    ``rank`` 1-based within that ranking.

    Contract:
    - Within one ranking only the first occurrence of an ID counts.
    - IDs missing from a ranking get nothing from it (no penalty term).
    - Returns ``[(id, score), ...]`` sorted by score descending. Ties go to the ID
      that appears first when scanning ``rankings[0]`` fully, then ``rankings[1]``,
      and so on.
    - ``top_n`` truncates the result; ``None`` returns every ID.
    - No rankings, or only empty ones, returns ``[]``.

    Raises:
        ValueError: if ``k < 0`` or ``top_n`` is given and ``<= 0``.
    """

    if k < 0:
        raise ValueError("k must be >= 0")

    if top_n is not None and top_n <= 0:
        raise ValueError("top_n must be greater than 0")


    scores: dict[Hashable, float] = {}
    first_seen: dict[Hashable, int] = {}

    # Tracks the order in which IDs are first seen across all rankings
    order = 0

    for ranking in rankings:

        # Avoid counting duplicates in the same ranking
        seen_in_ranking: set[Hashable] = set()

        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen_in_ranking:
                continue  # Skip duplicates in the same ranking

            seen_in_ranking.add(doc_id)

            # Update score for this doc_id
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + rank)

            # Record the first time we see this doc_id across all rankings
            if doc_id not in first_seen:
                first_seen[doc_id] = order
                order += 1

    # Sort by score descending, then by first seen order
    result = sorted(scores.items(), key=lambda x: (-x[1], first_seen.get(x[0], float('inf'))))

    # Truncate if top_n is specified
    if top_n is not None:
        result = result[:top_n]

    return result