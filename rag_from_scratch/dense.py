"""Dense retrieval. YOU implement ``cosine_top_k`` (checkpoint 3)."""

import numpy as np


def cosine_top_k(query: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return the ``k`` rows of ``matrix`` most cosine-similar to ``query``.

    Contract:
    - ``query`` has shape ``(d,)``; ``matrix`` has shape ``(n, d)``.
    - Do not assume the inputs are normalized: return the true cosine similarity.
    - Returns ``[(row_index, similarity), ...]`` as Python ``int``/``float``,
      sorted by similarity descending; ties go to the lower row index.
    - ``k > n`` returns all ``n`` rows; ``n == 0`` returns ``[]``.
    - A zero vector (query or row) has similarity ``0.0`` with everything; never NaN.
    - Vectorized with NumPy: no Python loop over the rows.

    Raises:
        ValueError: if ``k <= 0``, ``query`` is not 1-D, ``matrix`` is not 2-D, or
            the dimensions differ.
    """
    raise NotImplementedError
