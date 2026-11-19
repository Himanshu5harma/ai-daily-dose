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
    raise NotImplementedError
