"""BM25 lexical retrieval. YOU implement this from scratch (checkpoint 4)."""

import re
import math
from collections import Counter

from collections.abc import Sequence


def tokenize(text: str) -> list[str]:
    """Lowercase ``text`` and return its runs of word characters, in order.

    Word characters are letters, digits and underscore (Unicode-aware, like
    ``\\w`` in ``re``). Everything else separates tokens. No stemming, no stopwords.

    Example: ``"shutil.copytree(dirs_exist_ok=True)"`` ->
    ``["shutil", "copytree", "dirs_exist_ok", "true"]``.
    """
    return re.findall(r"\w+", text.lower())


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
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not (0 <= b <= 1):
            raise ValueError("b must be in [0, 1]")

        self.k1 = k1
        self.b = b

        # Store document tokens
        self.corpus = list(corpus_tokens)

        # Number of documents
        self.N = len(self.corpus)

        # Document lengths
        self.doc_lengths = [len(doc) for doc in self.corpus]

        # Average document length
        self.avg_doc_length = sum(self.doc_lengths) / self.N if self.N > 0 else 0.0

        # Term frequencies for each document
        self.term_frequencies = [Counter(doc) for doc in self.corpus]

        # Document frequencies for each term
        self.doc_frequencies = Counter()

        for doc in self.corpus:
            self.doc_frequencies.update(set(doc))  # Count each term once per document

        

    def idf(self, term: str) -> float:
        """IDF of ``term`` (defined for unseen terms too, with ``df = 0``)."""
        df = self.doc_frequencies.get(term, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5)) if self.N > 0 else 0.0

    def scores(self, query_tokens: Sequence[str]) -> list[float]:
        """BM25 score of every document, in corpus order."""
        if self.avg_doc_length == 0:
            return [0.0] * self.N

        scores = []
        for i, doc in enumerate(self.corpus):
            score = 0.0
            doc_len = self.doc_lengths[i]
            tf = self.term_frequencies[i]

            for query_token in query_tokens:
                current_tf = tf.get(query_token, 0)

                if current_tf == 0:
                    continue

                idf = self.idf(query_token)

                # BM25 document length normalization
                normalization = (1 - self.b) + self.b * (doc_len / self.avg_doc_length)

                # term frequency saturation

                numerator = current_tf * (self.k1 + 1)
                denominator = current_tf + self.k1 * normalization

                score += idf * (numerator / denominator)

            scores.append(score)

        return scores
    

                

    def top_k(self, query_tokens: Sequence[str], k: int) -> list[tuple[int, float]]:
        """Up to ``k`` ``(doc_index, score)`` pairs with ``score > 0``.

        Sorted by score descending, ties to the lower index. Documents scoring
        exactly 0 are left out, so fewer than ``k`` results can come back.

        Raises:
            ValueError: if ``k <= 0``.
        """
        if k <= 0:
            raise ValueError("k must be greater than 0")

        scores = self.scores(query_tokens)
        scored_docs = [(i, score) for i, score in enumerate(scores) if score > 0]

        # Sort by score descending, then index ascending
        scored_docs.sort(key=lambda x: (-x[1], x[0]))

        return scored_docs[:k]
