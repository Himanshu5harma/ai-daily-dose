"""Evaluate two retrieval systems against shared qrels, the way a search team would.

Workflow
1. Judgments (qrels) and each system's ranked results (a run) live in TREC files.
2. Both runs are scored on the same queries with the ranx backend.
3. A paired randomization test and a bootstrap interval say whether the candidate's
   improvement is real or noise.
4. The reference Python backend scores the baseline too. Its means must match ranx,
   which is the point of having two independent implementations. Compare the timings:
   with this wrapper ranx is not faster, because converting the data dominates.

The data is synthetic, so the script runs anywhere. Replace the generated files with
your own TREC qrels and run files for a real evaluation.

    pip install -e '.[ranx]'
    python examples/evaluate_at_scale.py --queries 5000
"""

import argparse
import random
import tempfile
import time
from pathlib import Path

from ai_engineering.evaluations import (
    bootstrap_ci,
    evaluate,
    paired_randomization_test,
    read_qrels,
    read_run,
    write_run,
)

METRICS = ["recall@100", "mrr@10", "ndcg@10"]
CORPUS_SIZE = 50_000
DEPTH = 100  # documents retrieved per query


def make_benchmark(queries: int, seed: int) -> tuple[dict, dict, dict]:
    """Return graded qrels plus a baseline run and a better candidate run."""
    rng = random.Random(seed)
    qrels, baseline, candidate = {}, {}, {}
    for number in range(queries):
        query_id = f"q{number}"
        relevant = {f"d{d}": rng.choice([1, 1, 2, 3]) for d in rng.sample(range(CORPUS_SIZE), 5)}
        qrels[query_id] = relevant
        for run, boost in ((baseline, 0.8), (candidate, 1.1)):
            scores = {f"d{d}": rng.gauss(0, 1) for d in rng.sample(range(CORPUS_SIZE), DEPTH)}
            for doc, grade in relevant.items():  # a stronger system scores relevant docs higher
                if rng.random() < 0.85:
                    scores[doc] = rng.gauss(boost * (0.5 + grade / 3), 1)
            run[query_id] = dict(sorted(scores.items(), key=lambda item: -item[1])[:DEPTH])
    judgments = {q: {d: float(g) for d, g in docs.items()} for q, docs in qrels.items()}
    return judgments, baseline, candidate


def write_qrels(path: Path, qrels: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for query_id, judged in qrels.items():
            for doc_id, grade in judged.items():
                handle.write(f"{query_id} 0 {doc_id} {int(grade)}\n")


def timed(label: str, qrels: dict, run: dict, backend: str):
    start = time.perf_counter()
    report = evaluate(qrels, run, METRICS, backend=backend)
    print(f"  {label:<14}{time.perf_counter() - start:7.2f}s")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queries", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as directory:
        workdir = Path(directory)
        qrels_data, baseline_data, candidate_data = make_benchmark(args.queries, args.seed)
        write_qrels(workdir / "qrels.txt", qrels_data)
        write_run(workdir / "baseline.run", baseline_data, "baseline")
        write_run(workdir / "candidate.run", candidate_data, "candidate")

        qrels = read_qrels(workdir / "qrels.txt")
        baseline = read_run(workdir / "baseline.run")
        candidate = read_run(workdir / "candidate.run")

    print(f"{len(qrels):,} queries, {DEPTH} results each\n\nScoring time:")
    base = timed("baseline/ranx", qrels, baseline, "ranx")
    cand = timed("candidate/ranx", qrels, candidate, "ranx")
    reference = timed("baseline/python", qrels, baseline, "python")

    worst = max(abs(base.means[m] - reference.means[m]) for m in METRICS)
    print(f"\nBackends agree on every mean to within {worst:.1e}\n")

    header = f"{'metric':<12}{'baseline':>10}{'candidate':>11}{'delta':>9}{'p-value':>9}"
    print(f"{header}   95% CI of delta")
    for metric in METRICS:
        a, b = cand.per_query[metric], base.per_query[metric]
        test = paired_randomization_test(a, b, n_resamples=1_000, seed=0)
        interval = bootstrap_ci([a[q] - b[q] for q in a], n_resamples=1_000, seed=0)
        print(
            f"{metric:<12}{base.means[metric]:>10.4f}{cand.means[metric]:>11.4f}"
            f"{test.mean_difference:>+9.4f}{test.p_value:>9.4f}   "
            f"[{interval.low:+.4f}, {interval.high:+.4f}]"
        )
    print("\nShip only if the interval excludes 0 and latency and cost are acceptable.")


if __name__ == "__main__":
    main()
