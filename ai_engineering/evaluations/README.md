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
