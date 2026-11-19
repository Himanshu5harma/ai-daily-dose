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
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if query.ndim != 1:
        raise ValueError("query must be 1-dimensional")

    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")

    if query.shape[0] != matrix.shape[1]:
        raise ValueError("query and matrix dimensions must match")

    n = matrix.shape[0]
    if n == 0:
        return []

    # Calculate dot products between query and every row.
    dot_products = matrix @ query

    # Calculate vector magnitudes.
    query_norm = np.linalg.norm(query)
    row_norms = np.linalg.norm(matrix, axis=1)

    # Calculate cosine similarity safely.
    denominators = query_norm * row_norms
    
    similarities = np.divide(
        dot_products,
        denominators,
        out=np.zeros_like(dot_products, dtype=float),
        where=denominators != 0,
    )

    # Sort by similarity descending, then row index ascending.
    indices = np.lexsort((np.arange(n), -similarities))
    top_indices = indices[:k]

    return [(int(i), float(similarities[i])) for i in top_indices]
