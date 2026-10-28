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

### Why it matters and trade-offs

Low recall can identify missing context before evaluating an LLM's generated
answer. High recall alone does not measure result ordering, irrelevant context,
answer correctness or latency. Future precision and reciprocal rank metrics will
complement it.

The implementation uses sets for membership and deduplication. For R relevance
judgments and T = min(K, number of retrieved results), expected time and auxiliary
space are O(R + T), assuming constant-time hashing. It does not call a model,
vector database or network service.

Run `python -m pytest tests/test_recall_at_k.py` after installing the development
dependencies from the repository root.
