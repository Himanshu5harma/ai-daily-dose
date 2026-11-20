"""Experiment runner: dense vs BM25 vs hybrid (RRF), by query type and chunk size.

Usage (from the repo root)::

    python -m rag_from_scratch.run_experiments                  # all 4 configs, all 3 methods
    python -m rag_from_scratch.run_experiments --no-dense       # BM25 only, no model needed
    python -m rag_from_scratch.run_experiments --configs section-2000,fixed-2000

Configs are ``<chunker>-<size>``: ``fixed`` (with ``--overlap``) or ``section``, at 500
or 2000 characters.

How a query is scored (both metrics come from ``ai_engineering.evaluations``):

- A question has one or more evidence *groups*; a chunk satisfies a group when it
  covers one of the group's gold ``(source_id, anchor)`` references
  (see ``corpus.anchors_in_span``).
- ``recall@5``: share of the question's evidence groups that appear in the top 5 chunks.
- ``MRR``: 1 / rank of the first chunk (within the top 5) that satisfies any group.
- By default, candidates are limited to pages of the release line(s) the question names,
  because the 3.12 and 3.13 pages are near-duplicates and would otherwise tie.
  ``--no-version-filter`` turns that off.
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ai_engineering.evaluations import recall_at_k, reciprocal_rank_at_k

from .bm25 import BM25, tokenize
from .chunking import fixed_size_spans, section_spans
from .corpus import Document, anchors_in_span, load_corpus
from .dense import cosine_top_k
from .embeddings import Embedder
from .fusion import reciprocal_rank_fusion
from .questions import Question, load_questions

K = 5  # cutoff for recall@K and MRR@K
DEPTH = 50  # how deep each retriever's list is before fusion
RRF_K = 60
QUERY_TYPES = ("exact", "conceptual", "multi")
RESULTS_DIR = Path(__file__).resolve().parent / "results"
ALL_CONFIGS = "fixed-500,section-500,fixed-2000,section-2000"


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    version: str
    start: int
    end: int
    text: str
    evidence: frozenset[tuple[str, str]]  # (source_id, anchor) pairs this chunk covers


def build_chunks(docs: list[Document], strategy: str, size: int, overlap: int) -> list[Chunk]:
    chunks = []
    for doc in docs:
        if strategy == "fixed":
            spans = fixed_size_spans(doc.text, size, overlap)
        else:
            spans = section_spans(doc, size)
        for start, end in spans:
            evidence = frozenset((doc.doc_id, a) for a in anchors_in_span(doc, start, end))
            chunks.append(Chunk(doc.doc_id, doc.version, start, end, doc.text[start:end], evidence))
    return chunks


def satisfies(chunk: Chunk, group) -> bool:
    return any((r.source_id, r.anchor) in chunk.evidence for r in group.references)


def score_query(question: Question, ranked: list[int], chunks: list[Chunk]) -> dict:
    """Recall@K over evidence groups, MRR@K over chunks, and the first relevant rank."""
    relevant = {
        i for i, c in enumerate(chunks) if any(satisfies(c, g) for g in question.evidence_groups)
    }
    emitted: list[str] = []  # group ids covered by the top-K chunks
    for i in ranked[:K]:
        for group in question.evidence_groups:
            if group.group_id not in emitted and satisfies(chunks[i], group):
                emitted.append(group.group_id)
    group_ids = [g.group_id for g in question.evidence_groups]
    first_rank = next((r for r, i in enumerate(ranked, 1) if i in relevant), None)
    return {
        "recall": recall_at_k(group_ids, emitted, max(1, len(emitted))),
        "rr": reciprocal_rank_at_k(relevant, ranked, K),
        "first_rank": first_rank,
    }


def run_config(
    chunks: list[Chunk],
    questions: list[Question],
    embedder: Embedder | None,
    version_filter: bool,
) -> dict[str, dict[str, dict]]:
    """Return ``{method: {qid: score dict}}`` for one chunking configuration."""
    bm25 = BM25([tokenize(c.text) for c in chunks])
    matrix = qvecs = None
    if embedder is not None:
        matrix = embedder.embed_documents([c.text for c in chunks])
        qvecs = embedder.embed_queries([q.query for q in questions])

    out: dict[str, dict[str, dict]] = {"bm25": {}}
    if embedder is not None:
        out = {"dense": {}, "bm25": {}, "hybrid": {}}

    for qi, q in enumerate(questions):
        allowed = [
            i
            for i, c in enumerate(chunks)
            if not (version_filter and q.versions) or c.version in q.versions
        ]
        allowed_set = set(allowed)
        bm_hits = bm25.top_k(tokenize(q.query), len(chunks))
        rankings = {"bm25": [i for i, _ in bm_hits if i in allowed_set][:DEPTH]}
        if embedder is not None:
            hits = cosine_top_k(qvecs[qi], matrix[allowed], DEPTH)
            rankings["dense"] = [allowed[i] for i, _ in hits]
            fused = reciprocal_rank_fusion([rankings["dense"], rankings["bm25"]], RRF_K, DEPTH)
            rankings["hybrid"] = [doc_id for doc_id, _ in fused]
        for method, ranked in rankings.items():
            out[method][q.qid] = score_query(q, ranked, chunks)
    return out


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def print_chunk_stats(stats: dict[str, dict]) -> None:
    print("\nChunk statistics")
    print(f"{'config':<14}{'chunks':>8}{'total chars':>14}{'median':>9}{'<100 chars':>12}")
    for name, s in stats.items():
        print(f"{name:<14}{s['chunks']:>8}{s['total_chars']:>14}{s['median']:>9}{s['tiny']:>12}")


def print_table(results: dict, questions: list[Question]) -> None:
    by_type = {t: [q for q in questions if q.query_type == t] for t in QUERY_TYPES}
    header = f"{'config / method':<22}" + "".join(
        f"{t + ' (n=' + str(len(by_type[t])) + ')':>20}" for t in QUERY_TYPES
    )
    print(f"\nrecall@{K} / MRR@{K}  (mean over queries)")
    print(header + f"{'ALL (n=' + str(len(questions)) + ')':>20}")
    for config, methods in results.items():
        print("-" * len(header))
        for method, per_query in methods.items():
            cells = []
            for group in [*(by_type[t] for t in QUERY_TYPES), questions]:
                r = mean(per_query[q.qid]["recall"] for q in group)
                m = mean(per_query[q.qid]["rr"] for q in group)
                cells.append(f"{r:.2f} / {m:.2f}")
            print(f"{config + ' ' + method:<22}" + "".join(f"{c:>20}" for c in cells))


def print_disagreements(results: dict, questions: list[Question], top: int = 5) -> None:
    print(f"\nTop {top} queries where methods disagree most (rank of first relevant chunk)")
    for config, methods in results.items():
        if len(methods) < 2:
            continue
        rows = []
        for q in questions:
            rr = [methods[m][q.qid]["rr"] for m in methods]
            rc = [methods[m][q.qid]["recall"] for m in methods]
            spread = (max(rr) - min(rr), max(rc) - min(rc))
            ranks = {m: methods[m][q.qid]["first_rank"] for m in methods}
            rows.append((spread, q, ranks))
        rows.sort(key=lambda r: r[0], reverse=True)
        print(f"\n[{config}]")
        print(f"  {'qid':<8}{'type':<11}" + "".join(f"{m:>8}" for m in methods) + "  query")
        for _, q, ranks in rows[:top]:
            cells = "".join(f"{ranks[m] if ranks[m] else '-':>8}" for m in methods)
            print(f"  {q.qid:<8}{q.query_type:<11}{cells}  {q.query[:60]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", default=ALL_CONFIGS)
    parser.add_argument("--overlap", type=int, default=0, help="overlap for fixed chunking")
    parser.add_argument("--no-dense", action="store_true", help="BM25 only; skips the model")
    parser.add_argument("--no-version-filter", action="store_true")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "results.json")
    args = parser.parse_args()

    docs = load_corpus()
    questions = load_questions()
    embedder = None if args.no_dense else Embedder()

    results, stats = {}, {}
    for config in args.configs.split(","):
        strategy, size = config.split("-")
        start = time.time()
        chunks = build_chunks(docs, strategy, int(size), args.overlap)
        lengths = sorted(len(c.text) for c in chunks)
        name = config + (f"+ov{args.overlap}" if strategy == "fixed" and args.overlap else "")
        stats[name] = {
            "chunks": len(chunks),
            "total_chars": sum(lengths),
            "median": lengths[len(lengths) // 2],
            "tiny": sum(n < 100 for n in lengths),
        }
        print(f"{name}: {len(chunks)} chunks, running...", flush=True)
        results[name] = run_config(chunks, questions, embedder, not args.no_version_filter)
        print(f"{name}: done in {time.time() - start:.0f}s", flush=True)

    print_chunk_stats(stats)
    print_table(results, questions)
    print_disagreements(results, questions)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "k": K,
        "rrf_k": RRF_K,
        "overlap": args.overlap,
        "version_filter": not args.no_version_filter,
        "model": None if embedder is None else embedder.model_name,
        "questions": {q.qid: {"query": q.query, "type": q.query_type} for q in questions},
        "stats": stats,
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
