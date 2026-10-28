# Interview questions

Questions an interviewer could ask about the code in this repo, with answers worded the way
you would say them out loud. Add new ones under the matching heading as you add features.

## Retrieval metrics (`ai_engineering/evaluations/`)

**Q1. What is the difference between Recall@K and Precision@K?**

**A:** Both look at the top K results. Recall@K asks how many of the relevant documents I
found, so I divide the relevant hits by the total number of relevant documents. Precision@K
asks how much of what I returned is useful, so I divide the relevant hits by K. For example,
with relevant docs `{a, c}` and results `[a, b, c]`, at K=2 recall is 1/2 and precision is
1/2. At K=3 recall is 2/2 but precision drops to 2/3.

**Q2. In a RAG system, which one matters more?**

**A:** It depends on the stage. For the first-stage retriever I care about recall, because a
document that isn't retrieved can never reach the answer, and a reranker can clean up a noisy
list afterwards. For the chunks I put into the LLM prompt I care about precision, because
irrelevant chunks cost tokens, add noise and can distract the model.

**Q3. Why is the denominator of Precision@K always K, and not the number of results returned?**

**A:** Because dividing by the returned count rewards a system for returning very little. If
it returned one correct result out of a requested ten, it would score 1.0 and hide the fact
that it found almost nothing. With K as the denominator, missing slots count as misses: one
hit in `["a"]` at K=10 scores 0.1.

**Q4. What is the highest Precision@K a query can get?**

**A:** `min(number of relevant docs, K) / K`. If a query has only 2 relevant documents and
K=5, the best possible score is 0.4. So a low precision is not always a sign of a bad
retriever, and I should compare the score to that ceiling.

**Q5. How does your implementation handle duplicate IDs in the retrieved list?**

**A:** A duplicate still uses up a ranking position, but the relevant ID counts only once. I
build a set from the top K results, so duplicates collapse. Without that, a retriever could
raise its score just by repeating one relevant document.

**Q6. What do Recall@K and Precision@K not tell you?**

**A:** They ignore the order inside the top K, so a hit at rank 1 scores the same as a hit at
rank K. They also treat relevance as yes or no, with no grades. If order matters I'd use MRR,
and if I have graded relevance I'd use NDCG.

**Q7. How would you report these metrics over a whole evaluation set?**

**A:** I compute the metric for each query and then take the mean, which is a macro average.
I also have to decide what to do with queries that have no relevant documents, since recall is
undefined for them. In this repo they return 0.0, and that choice is documented.

**Q8. Why did you use `itertools.islice` instead of `retrieved_ids[:k]`?**

**A:** Slicing needs a list, and it fails on generators. `islice` works on any iterable and
reads at most K items, so a lazy or even infinite retrieval stream is never fully consumed.
There is a test that raises an error if the function reads past the cutoff.

**Q9. Why do you reject `True` and `False` for `k`?**

**A:** `bool` is a subclass of `int` in Python, so `isinstance(True, int)` is `True`, and
`k=True` would silently behave like `k=1`. I check `isinstance(k, bool)` first so that this
mistake raises a `TypeError` instead of giving a misleading score.

**Q10. What is the time and space complexity?**

**A:** Both are O(R + K), where R is the number of relevant IDs. I build one set of the
relevant IDs and one set of at most K retrieved IDs, and the intersection is bounded by the
smaller of the two.

**Q11. Why return 0.0 for an empty relevant set instead of raising an error?**

**A:** A query with no judged relevant documents is a normal case in an evaluation dataset,
and raising would stop a batch run halfway through. Returning 0.0 keeps the run going. It is
a convention rather than an exact mathematical answer, so I documented it.

## Policy-aware retrieval (`ai_engineering/retrieval/`)

**Q12. Why must authorization filtering happen before reranking and LLM context building?**

**A:** There are four reasons.
1. If I rank first and filter later, forbidden documents with high scores take the top-K
   slots and are then removed, so the user gets too few results or none.
2. Rerankers and LLMs read the full text. Once restricted text reaches them it is already
   disclosed, and a prompt instruction like "ignore restricted documents" can be bypassed by
   prompt injection.
3. Scores, result counts, citations and latency can reveal that hidden documents exist.
4. Filtering once at the start gives me a single enforcement point, so every later stage
   behaves as if the forbidden documents were never in the corpus.

**Q13. Would you filter before or after the vector search?**

**A:** Before, as a metadata filter inside the vector query, so ineligible documents are
never fetched. If I filter after a top-K search, I can end up with fewer than K results even
though eligible documents exist. I would still keep a check in the application as a second
layer, in case the database filter is misconfigured.

**Q14. When no document passes the policy, why return 200 with an empty list instead of 404?**

**A:** "Nothing you are allowed to see matched" is a normal outcome, not an error. If I
returned a different status when matches exist but are hidden, the status itself would tell
the caller that restricted documents exist.

**Q15. What does "fail closed" mean in your request validation?**

**A:** If a document has missing or unknown policy metadata, I reject the request with a 422
instead of treating the document as public. When the system can't tell whether something is
allowed, the answer is no.

**Q16. Why did you add a custom handler for validation errors?**

**A:** A test with a `NaN` score showed a real bug. FastAPI's default 422 handler copies the
rejected value into the response, and `NaN` can't be written as valid JSON, so the server
returned a 500. My handler returns only where the error is and why, never the value. That
also stops document content from being echoed back in error messages.

**Q17. Where should `user_context` come from in a real service?**

**A:** From the authenticated identity, for example verified token claims or a session
lookup, and never from the request body. If the client sends its own access level, any caller
can simply claim `restricted`. In this repo it is in the body only to keep the exercise
simple.

**Q18. Why do you break score ties by document ID?**

**A:** So the same request always returns the same order. Without a tie-breaker the result
depends on the order of the input list, which makes the API unpredictable and the tests
flaky.
