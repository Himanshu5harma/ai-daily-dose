# Policy-aware retrieval

`POST /retrieve` takes a query, the caller's `user_context` and scored `candidate_documents`.
It drops every candidate that fails policy, then returns the top 3 remaining by score.

## Policy

A document is eligible only if all three checks pass:

| Check | Rule |
| --- | --- |
| Authorization | Document `access_level` is at or below the user's (`public` < `internal` < `confidential` < `restricted`) |
| Region | Document `region` equals the user's region (case-insensitive), or is `global` |
| Product version | Document `product_version` is in `allowed_product_versions` (exact match) |

Eligible documents are sorted by `score` (descending, ties by `id`) and cut to 3.

## API contract

| Status | When |
| --- | --- |
| `200` | Valid request. `results` may be empty when nothing passes policy |
| `422` | Invalid payload: missing or unknown fields, unknown access level, non-numeric or non-finite score, more than 1000 candidates |

Validation fails closed: a document without complete policy metadata is rejected, not
treated as public. Error bodies list the location and reason only, never the rejected value.
An empty result is a `200`, not a `404`. "Nothing you may see matched" is a normal outcome,
and a different status would reveal that hidden matches exist.

```json
{
  "query": "how do I reset my password?",
  "top_k": 3,
  "count": 1,
  "results": [
    {"id": "kb-12", "rank": 1, "score": 0.91, "content": "...",
     "metadata": {"access_level": "internal", "region": "eu", "product_version": "v3"}}
  ]
}
```

## Why authorization filtering must happen before reranking and context construction

1. **Top-k correctness.** If you rank first and filter second, forbidden documents with high
   scores take top-k slots and are then removed. The user gets fewer than k results, or none,
   even when eligible evidence exists. `test_filtering_happens_before_top_k_selection` covers
   this case.
2. **Rerankers and LLMs are not security boundaries.** A cross-encoder or LLM reranker reads
   the full document text. Passing restricted text to it already discloses that text to
   another system, its logs and possibly a third-party API. Also, once text is in a prompt,
   nothing reliably stops the model from quoting, paraphrasing or summarizing it. Prompt
   instructions such as "ignore restricted documents" can be bypassed by prompt injection.
3. **Side channels.** Scores, result counts, latency, citations and "I found something but
   can't show it" replies can all reveal that restricted documents exist. Filtering first
   makes every later stage behave as if the forbidden documents were never in the corpus.
4. **One enforcement point.** Applying policy once, at the start of the pipeline, means
   rerankers, caches, context builders and evaluation tools can be added or swapped without
   each one having to enforce access rules again.
5. **Stale and wrong-region evidence counts as a failure too.** Outdated product-version or
   wrong-region documents in the context produce answers that are confidently wrong or
   non-compliant, for example with data-residency rules. They need the same early exclusion.

In production, apply the same filter inside the vector-store query (metadata pre-filtering),
so ineligible documents are never fetched. Use this endpoint's check as defense in depth. The
`user_context` here comes from the request body for the exercise. A real service must derive
it from an authenticated identity (for example, token claims), never from client input.
