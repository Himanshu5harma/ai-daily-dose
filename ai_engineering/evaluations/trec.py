"""Readers for the TREC qrels and run file formats used by search benchmarks.

qrels line: ``query_id  iteration  document_id  grade``
run line:   ``query_id  Q0  document_id  rank  score  system_name``

Fields are whitespace separated. Files are read line by line, so large files are
not loaded as text first. Blank lines and lines starting with ``#`` are ignored.
"""

import math
from collections.abc import Mapping
from os import PathLike
from pathlib import Path


def read_qrels(path: str | PathLike[str]) -> dict[str, dict[str, float]]:
    """Return ``{query ID: {document ID: grade}}`` from a TREC qrels file.

    Raises:
        ValueError: On a malformed line, a non-finite or negative grade, or a
            conflicting duplicate judgment for the same query and document.
    """
    qrels: dict[str, dict[str, float]] = {}
    for number, fields in _lines(path, expected=4):
        query_id, _iteration, doc_id, raw_grade = fields
        grade = _number(raw_grade, number, "grade")
        if grade < 0:
            raise ValueError(f"line {number}: grade must not be negative")
        judged = qrels.setdefault(query_id, {})
        if judged.setdefault(doc_id, grade) != grade:
            raise ValueError(f"line {number}: conflicting judgments for {query_id} {doc_id}")
    return qrels


def read_run(path: str | PathLike[str]) -> dict[str, dict[str, float]]:
    """Return ``{query ID: {document ID: score}}`` from a TREC run file.

    The score column is used, not the rank column. If a document appears twice for
    a query, the higher score is kept.

    Raises:
        ValueError: On a malformed line or a non-finite score.
    """
    run: dict[str, dict[str, float]] = {}
    for number, fields in _lines(path, expected=6):
        query_id, _q0, doc_id, _rank, raw_score, _system = fields
        score = _number(raw_score, number, "score")
        ranked = run.setdefault(query_id, {})
        ranked[doc_id] = max(score, ranked.get(doc_id, score))
    return run


def write_run(
    path: str | PathLike[str], run: Mapping[str, Mapping[str, float]], system: str = "run"
) -> None:
    """Write ``{query ID: {document ID: score}}`` as a TREC run file, best first."""
    with Path(path).open("w", encoding="utf-8") as handle:
        for query_id, scores in run.items():
            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                handle.write(f"{query_id} Q0 {doc_id} {rank} {score!r} {system}\n")


def _lines(path: str | PathLike[str], expected: int):
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            fields = text.split()
            if len(fields) != expected:
                raise ValueError(f"line {number}: expected {expected} fields, got {len(fields)}")
            yield number, fields


def _number(raw: str, line: int, name: str) -> float:
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"line {line}: {name} {raw!r} is not a number") from None
    if not math.isfinite(value):
        raise ValueError(f"line {line}: {name} must be finite")
    return value
