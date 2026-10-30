# Retrieval evaluation

## Recall@K

For one query, Recall@K measures how much of the known relevant document set
appears in the first K retrieved results:

```text
Recall@K = unique relevant IDs in the first K results / unique relevant IDs
```

For relevant IDs `{a, c}` and retrieved IDs `[a, b, c]`, Recall@2 is `0.5`
and Recall@3 is `1.0`.

### Input contract and edge cases

- Document IDs can be any hashable type. Use a consistent ID type across inputs.
- Retrieved results must be ordered; duplicate IDs still occupy ranking positions.
- Duplicate relevant or retrieved IDs never inflate the score.
- A positive integer K is required. Booleans and non-integers raise `TypeError`;
  zero and negative integers raise `ValueError`.
- If fewer than K results exist, all available results are used.
- Empty retrievals return `0.0`.
- With no relevance judgments, recall is mathematically undefined. This API
  returns `0.0` as an explicit convention. Track or exclude unjudged queries
  separately when computing aggregate evaluation results.
- Iterables, including generators, are supported. At most K retrieved IDs are
  consumed; an empty relevant set does not consume retrievals.

### Limitations

Recall@K does not measure result ordering or irrelevant results, so it should be
paired with precision and rank-based metrics.

## Batch evaluation

`evaluate(qrels, run, metrics)` scores many queries at once and returns an
`EvaluationReport` with per-query scores, macro-averaged means, and the
evaluated, skipped and unjudged queries.

```python
from ai_engineering.evaluations import bootstrap_ci, evaluate, paired_randomization_test

qrels = {"q1": {"doc-1": 3, "doc-2": 0, "doc-3": 1}}  # graded judgments
run = {"q1": {"doc-1": 0.9, "doc-3": 0.7}}  # scores or an ordered list
report = evaluate(qrels, run, ["recall@10", "precision@5", "mrr@10", "ndcg@10", "judged@10"])
report.means["ndcg@10"]
```

- Metrics: `recall@k`, `precision@k`, `mrr@k`, `ndcg@k`, `judged@k`.
- Binary metrics count grades `>= relevance_threshold` (default 1) as relevant.
  NDCG uses every positive grade, with `gain="linear"` (trec_eval) or
  `"exponential"` (`2**grade - 1`).
- Unjudged documents count as non-relevant. Watch `judged@k` when comparing
  systems; a drop means the judgments no longer cover what the system returns.
- Queries missing from the run score 0.0. Queries without relevant documents
  follow `empty_queries="skip" | "zero" | "raise"`.
- Scored runs are sorted by descending score, then by ascending document ID.

`bootstrap_ci(scores)` gives a confidence interval for a mean score, and
`paired_randomization_test(scores_a, scores_b)` gives a p-value for the
per-query difference between two systems. Both accept a `seed`.

The default backend is pure Python: O(judgments + k) per query, and
O(n_resamples × queries) for the statistics.

### ranx backend

`evaluate(..., backend="ranx")` computes the same metrics with the
[ranx](https://amenra.github.io/ranx/) library, an independent and widely used
implementation. Install it with `pip install -e '.[ranx]'`.

- Use it to **cross-check** the reference numbers. The test suite compares both
  backends on 300 random graded queries, including ties, missing queries and
  empty judgments, and they agree to within 1e-6.
- It is **not faster** here. On 20,000-60,000 queries with 100 results each,
  the reference backend is quicker (for example 1.9 s against 4.1 s for 60,000
  queries and 3 metrics), because converting dicts to ranx's format costs more
  than the metric math. The first ranx call also pays about 2 s for numba
  compilation. If you need speed, build ranx `Qrels` and `Run` objects once and
  call `ranx.evaluate` directly.
- Differences: `judged@k` is not supported, `relevance_threshold` must be an
  integer, and a repeated document keeps only its first position.
- Document and query IDs are converted to strings for ranx; IDs that collide
  after conversion raise `ValueError`.

`read_qrels`, `read_run` and `write_run` read and write the TREC file formats
that search benchmarks use. `examples/evaluate_at_scale.py` runs the whole
workflow: TREC files, batch scoring, a baseline against candidate comparison with
a paired test and bootstrap interval, and a check that both backends agree.
