"""BM25 lexical retrieval. YOU implement this from scratch (checkpoint 4)."""

from collections.abc import Sequence


def tokenize(text: str) -> list[str]:
    """Lowercase ``text`` and return its runs of word characters, in order.

    Word characters are letters, digits and underscore (Unicode-aware, like
    ``\\w`` in ``re``). Everything else separates tokens. No stemming, no stopwords.

    Example: ``"shutil.copytree(dirs_exist_ok=True)"`` ->
    ``["shutil", "copytree", "dirs_exist_ok", "true"]``.
    """
    raise NotImplementedError


class BM25:
    """Okapi BM25 over a fixed corpus of pre-tokenized documents.

    Uses the standard Okapi term-frequency saturation (``k1``) and document-length
    normalization (``b``, relative to the average document length), with the
    Lucene-style IDF that is never negative::

        idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

    where ``N`` is the number of documents and ``df(t)`` how many contain ``t``.

    Contract:
    - A document's score is the sum over the *query tokens as given* (a repeated
      query token counts each time) of that token's BM25 term score.
    - Tokens absent from a document contribute 0.
    - If every document is empty (average length 0), every score is 0.0.

    Raises (in ``__init__``):
        ValueError: if ``k1 < 0`` or ``b`` is outside ``[0, 1]``.
    """

    def __init__(self, corpus_tokens: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        raise NotImplementedError

    def idf(self, term: str) -> float:
        """IDF of ``term`` (defined for unseen terms too, with ``df = 0``)."""
        raise NotImplementedError

    def scores(self, query_tokens: Sequence[str]) -> list[float]:
        """BM25 score of every document, in corpus order."""
        raise NotImplementedError

    def top_k(self, query_tokens: Sequence[str], k: int) -> list[tuple[int, float]]:
        """Up to ``k`` ``(doc_index, score)`` pairs with ``score > 0``.

        Sorted by score descending, ties to the lower index. Documents scoring
        exactly 0 are left out, so fewer than ``k`` results can come back.

        Raises:
            ValueError: if ``k <= 0``.
        """
        raise NotImplementedError
