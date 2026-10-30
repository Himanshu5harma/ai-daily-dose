"""Batch evaluation of a retrieval run against relevance judgments (qrels)."""

import math
import re
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from ._validation import check_finite, check_grade
from .ndcg_at_k import Gain, ndcg_at_k
from .precision_at_k import precision_at_k
from .recall_at_k import recall_at_k
from .reciprocal_rank_at_k import reciprocal_rank_at_k

QueryId = Hashable
DocumentId = Hashable
Qrels = Mapping[QueryId, Mapping[DocumentId, float]]
Ranking = Mapping[DocumentId, float] | Iterable[DocumentId]
Run = Mapping[QueryId, Ranking]
EmptyQueryPolicy = Literal["skip", "zero", "raise"]
Backend = Literal["python", "ranx"]
# (query ID, judgments, IDs counted as relevant, ranked document IDs)
_Query = tuple[QueryId, Mapping[DocumentId, float], set[DocumentId], list[DocumentId]]

_METRIC = re.compile(r"(recall|precision|mrr|ndcg|judged)@([1-9][0-9]*)")
_POLICIES = ("skip", "zero", "raise")
_BACKENDS = ("python", "ranx")


@dataclass(frozen=True)
class EvaluationReport:
    """Macro-averaged and per-query scores for one run."""

    means: dict[str, float]
    per_query: dict[str, dict[QueryId, float]]
    evaluated_queries: tuple[QueryId, ...]
    skipped_queries: tuple[QueryId, ...]
    unjudged_queries: tuple[QueryId, ...]


def evaluate(
    qrels: Qrels,
    run: Run,
    metrics: Iterable[str],
    *,
    relevance_threshold: float = 1.0,
    empty_queries: EmptyQueryPolicy = "skip",
    gain: Gain = "linear",
    backend: Backend = "python",
) -> EvaluationReport:
    """Score every judged query in ``run`` and return per-query and mean scores.

    ``metrics`` are names such as ``"recall@10"``, ``"precision@5"``,
    ``"mrr@10"``, ``"ndcg@10"`` and ``"judged@10"`` (share of the top k that has
    a judgment). A run entry is either an ordered sequence of IDs or a mapping of
    ID to score, sorted by descending score with ties broken by ascending ID.

    ``backend="python"`` (default) uses the dependency-free reference functions.
    ``backend="ranx"`` computes the scores with the ranx library, an independent and
    widely used implementation, so it is the way to cross-check the reference
    numbers. It is not faster: both backends spend most of their time preparing data
    in Python. It needs ``pip install 'ai-daily-dose[ranx]'``, an integer
    ``relevance_threshold``, and does not support ``judged@k``. Both backends share
    the validation, query selection and ordering rules above, and return the same
    report; a repeated document keeps only its first position with ranx.

    - Queries in ``qrels`` but missing from ``run`` score 0.0.
    - Queries in ``run`` but missing from ``qrels`` are reported in
      ``unjudged_queries`` and not scored.
    - Unjudged documents count as non-relevant.
    - Binary metrics treat grades >= ``relevance_threshold`` as relevant; NDCG
      uses every positive grade.
    - Queries with no relevant documents are skipped, scored as 0.0, or raise,
      depending on ``empty_queries``.

    Raises:
        TypeError: If a ranking is a string or a grade or score is not a number.
        ValueError: On unknown metrics or policy, invalid grades or scores, a
            non-positive threshold, or when no query is left to evaluate.
    """
    parsed = {}
    for metric in metrics:
        name = metric.strip().lower()
        match = _METRIC.fullmatch(name)
        if match is None:
            raise ValueError(f"unknown metric {metric!r}")
        parsed[name] = (match.group(1), int(match.group(2)))
    if not parsed:
        raise ValueError("at least one metric is required")
    if empty_queries not in _POLICIES:
        raise ValueError(f"empty_queries must be one of {_POLICIES}")
    if check_finite(relevance_threshold, "relevance_threshold") <= 0:
        raise ValueError("relevance_threshold must be greater than zero")

    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}")

    queries: list[_Query] = []
    skipped = []
    for query_id, judgments in qrels.items():
        relevant = {
            doc for doc, grade in judgments.items() if check_grade(grade) >= relevance_threshold
        }
        if not relevant:
            if empty_queries == "raise":
                raise ValueError(f"query {query_id!r} has no relevant documents")
            if empty_queries == "skip":
                skipped.append(query_id)
                continue
        queries.append((query_id, judgments, relevant, _ranked_ids(run.get(query_id, ()))))

    if not queries:
        raise ValueError("no queries left to evaluate")

    if backend == "ranx":
        from .ranx_backend import score_queries
    else:
        score_queries = _score_queries
    per_query = score_queries(queries, parsed, relevance_threshold=relevance_threshold, gain=gain)
    evaluated = [query[0] for query in queries]

    means = {
        name: math.fsum(scores.values()) / len(evaluated) for name, scores in per_query.items()
    }
    return EvaluationReport(
        means=means,
        per_query=per_query,
        evaluated_queries=tuple(evaluated),
        skipped_queries=tuple(skipped),
        unjudged_queries=tuple(query_id for query_id in run if query_id not in qrels),
    )


def _ranked_ids(ranking: Ranking) -> list[DocumentId]:
    if isinstance(ranking, Mapping):
        scores = {doc: check_finite(score, "retrieval score") for doc, score in ranking.items()}
        return sorted(scores, key=lambda doc: (-scores[doc], doc))
    if isinstance(ranking, (str, bytes)):
        raise TypeError("ranking must be a sequence of document IDs, not a string")
    return list(ranking)


def _score_queries(
    queries: list[_Query],
    parsed: Mapping[str, tuple[str, int]],
    *,
    relevance_threshold: float,
    gain: Gain,
) -> dict[str, dict[QueryId, float]]:
    del relevance_threshold  # already applied when each query's relevant set was built
    return {
        name: {
            query_id: _score(kind, k, judgments, relevant, ranking, gain)
            for query_id, judgments, relevant, ranking in queries
        }
        for name, (kind, k) in parsed.items()
    }


def _score(
    kind: str,
    k: int,
    judgments: Mapping[DocumentId, float],
    relevant: set[DocumentId],
    ranking: list[DocumentId],
    gain: Gain,
) -> float:
    if kind == "recall":
        return recall_at_k(relevant, ranking, k)
    if kind == "precision":
        return precision_at_k(relevant, ranking, k)
    if kind == "mrr":
        return reciprocal_rank_at_k(relevant, ranking, k)
    if kind == "ndcg":
        return ndcg_at_k(judgments, ranking, k, gain=gain)
    top = ranking[:k]
    return sum(doc in judgments for doc in top) / len(top) if top else 0.0
