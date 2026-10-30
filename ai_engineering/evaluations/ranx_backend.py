"""Scoring backend that delegates metric computation to the ranx library.

Use it through ``evaluate(..., backend="ranx")``. ranx is an optional dependency
(``pip install 'ai-daily-dose[ranx]'``) and an independent implementation of the
metrics, which makes it a cross-check for the reference backend. This module only
converts data to and from ranx's format, so validation, query selection and ranking
order stay identical to the reference backend. Measured on 20,000-60,000 queries
with 100 results each, it is slower than the reference backend, because converting
dicts to ranx's format costs more than the metric math it saves.
"""

from collections.abc import Hashable, Mapping
from typing import Any

from .ndcg_at_k import Gain

QueryId = Hashable
DocumentId = Hashable


def score_queries(
    queries: list[tuple[QueryId, Mapping[DocumentId, float], set[DocumentId], list[DocumentId]]],
    parsed: Mapping[str, tuple[str, int]],
    *,
    relevance_threshold: float,
    gain: Gain,
) -> dict[str, dict[QueryId, float]]:
    """Return ``{metric name: {query ID: score}}`` computed by ranx.

    Each query is ``(query ID, judgments, relevant IDs, ranked IDs)``. Queries with
    an empty ranking or empty judgments are scored 0.0 here, because ranx rejects
    empty inputs. A repeated document keeps only its first (highest) position,
    since a ranx run is a mapping of ID to score.

    Raises:
        ImportError: If ranx is not installed or cannot be imported.
        ValueError: For ``judged@k``, a non-integer threshold, or IDs that collide
            once converted to strings (ranx identifiers are strings).
    """
    if any(kind == "judged" for kind, _ in parsed.values()):
        raise ValueError("judged@k is not available with backend='ranx'")
    if relevance_threshold != int(relevance_threshold):
        raise ValueError("backend='ranx' needs an integer relevance_threshold")
    threshold = int(relevance_threshold)
    ranx = _import_ranx()

    ranx_names = {name: _ranx_name(kind, k, threshold, gain) for name, (kind, k) in parsed.items()}
    scores: dict[str, dict[QueryId, float]] = {name: {} for name in parsed}
    query_ids: dict[str, QueryId] = {}
    qrels: dict[str, dict[str, float]] = {}
    run: dict[str, dict[str, float]] = {}
    for query_id, judgments, _relevant, ranking in queries:
        key = str(query_id)
        if key in query_ids:
            raise ValueError(f"query IDs collide once converted to strings: {key!r}")
        query_ids[key] = query_id
        if judgments and ranking:
            qrels[key] = _string_keys(judgments)
            run[key] = _rank_scores(ranking)
        else:
            for name in parsed:
                scores[name][query_id] = 0.0

    if run:
        unique = list(dict.fromkeys(ranx_names.values()))
        # ranx raises KeyError for a single metric with return_mean=False; repeating
        # the name avoids that and does not change the result.
        results = ranx.evaluate(qrels, run, unique * 2 if len(unique) == 1 else unique, False)
        for name, ranx_name in ranx_names.items():
            # For plain dicts ranx scores queries in dict insertion order and does not
            # check that qrels and run were built in the same order. Both dicts above
            # are filled in the same loop, so their order is identical.
            for key, value in zip(run, results[ranx_name], strict=True):
                scores[name][query_ids[key]] = float(value)

    return {name: {query[0]: scores[name][query[0]] for query in queries} for name in parsed}


def _import_ranx() -> Any:
    try:
        import ranx
    except ImportError as exc:
        raise ImportError(
            "backend='ranx' needs the optional dependency: pip install 'ai-daily-dose[ranx]'"
        ) from exc
    return ranx


def _ranx_name(kind: str, k: int, threshold: int, gain: Gain) -> str:
    if kind == "ndcg":
        return f"{'ndcg_burges' if gain == 'exponential' else 'ndcg'}@{k}"
    return f"{kind}@{k}-l{threshold}"


def _string_keys(judgments: Mapping[DocumentId, float]) -> dict[str, float]:
    converted = {str(doc): float(grade) for doc, grade in judgments.items()}
    if len(converted) != len(judgments):
        raise ValueError("document IDs collide once converted to strings")
    return converted


def _rank_scores(ranking: list[DocumentId]) -> dict[str, float]:
    # Turn the ranking into strictly decreasing scores, so ranx reproduces our order
    # exactly and never has to break ties itself.
    unique = list(dict.fromkeys(ranking))
    ids = [str(doc) for doc in unique]
    if len(set(ids)) != len(ids):
        raise ValueError("document IDs collide once converted to strings")
    return {doc: float(len(ids) - position) for position, doc in enumerate(ids)}
