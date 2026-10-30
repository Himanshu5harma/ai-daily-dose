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

**Q12. What is reciprocal rank, and when would you use it instead of Recall@K or Precision@K?**

**A:** Reciprocal rank is 1 divided by the position of the first relevant result, counting
from 1. A hit at rank 1 scores 1.0, rank 2 scores 0.5, rank 3 scores 0.33, and no hit within
the top K scores 0.0. I use it when the user only needs one good answer, such as a question
answering or "top result" scenario, because it rewards putting that answer as high as
possible. Recall@K and Precision@K give the same score whether the hit is at rank 1 or rank K.
Averaged over many queries it is called MRR (mean reciprocal rank).

**Q13. What are the weaknesses of reciprocal rank, and how does your implementation handle duplicates?**

**A:** It only looks at the first relevant result, so it ignores every other relevant
document. A query with one hit at rank 1 and a query with five hits starting at rank 1 get the
same score. The 1/rank curve is also steep: going from rank 1 to 2 costs 0.5, but going from
rank 9 to 10 costs about 0.01. For duplicates, every occurrence takes a ranking position, so
`["a", "a", "c"]` with relevant `{c}` gives 1/3, not 1/2. I stop reading the retrieval as soon
as I find the first hit, and never past K.

## Policy-aware retrieval (`ai_engineering/retrieval/`)

**Q14. Why must authorization filtering happen before reranking and LLM context building?**

**A:** There are four reasons.
1. If I rank first and filter later, forbidden documents with high scores take the top-K
   slots and are then removed, so the user gets too few results or none.
2. Rerankers and LLMs read the full text. Once restricted text reaches them it is already
   disclosed, and a prompt instruction like "ignore restricted documents" can be bypassed by
   prompt injection.
3. Scores, result counts, citations and latency can reveal that hidden documents exist.
4. Filtering once at the start gives me a single enforcement point, so every later stage
   behaves as if the forbidden documents were never in the corpus.

**Q15. Would you filter before or after the vector search?**

**A:** Before, as a metadata filter inside the vector query, so ineligible documents are
never fetched. If I filter after a top-K search, I can end up with fewer than K results even
though eligible documents exist. I would still keep a check in the application as a second
layer, in case the database filter is misconfigured.

**Q16. When no document passes the policy, why return 200 with an empty list instead of 404?**

**A:** "Nothing you are allowed to see matched" is a normal outcome, not an error. If I
returned a different status when matches exist but are hidden, the status itself would tell
the caller that restricted documents exist.

**Q17. What does "fail closed" mean in your request validation?**

**A:** If a document has missing or unknown policy metadata, I reject the request with a 422
instead of treating the document as public. When the system can't tell whether something is
allowed, the answer is no.

**Q18. Why did you add a custom handler for validation errors?**

**A:** A test with a `NaN` score showed a real bug. FastAPI's default 422 handler copies the
rejected value into the response, and `NaN` can't be written as valid JSON, so the server
returned a 500. My handler returns only where the error is and why, never the value. That
also stops document content from being echoed back in error messages.

**Q19. Where should `user_context` come from in a real service?**

**A:** From the authenticated identity, for example verified token claims or a session
lookup, and never from the request body. If the client sends its own access level, any caller
can simply claim `restricted`. In this repo it is in the body only to keep the exercise
simple.

**Q20. Why do you break score ties by document ID?**

**A:** So the same request always returns the same order. Without a tie-breaker the result
depends on the order of the input list, which makes the API unpredictable and the tests
flaky.

## Production search evaluation (`ai_engineering/evaluations/batch.py`, `statistics.py`)

**Q21. How is production evaluation different from calling `recall_at_k` on one query?**

**A:** In production I never trust one query. I take a fixed set of judged queries, called
qrels, run the system to get a ranking per query, called a run, compute each metric per query,
and report the mean. I also report how many queries were scored, skipped or missing, because a
mean over 300 queries and a mean over 30 are not the same claim. That is what `evaluate`
returns: per-query scores, means, and the lists of evaluated, skipped and unjudged queries.

**Q22. What do you do with queries that have no relevant documents?**

**A:** Recall, MRR and NDCG are undefined for them, so I have to choose a policy and state it.
Skipping them is the common default. Scoring them as 0.0 punishes the system for something it
can't fix. Raising an error is useful in CI when every query should have a judgment. My
`evaluate` function supports all three and reports which queries were skipped, so the choice
is visible instead of hidden in the average.

**Q23. Why NDCG instead of recall or precision?**

**A:** Real judgments are usually graded, for example 0 = irrelevant up to 3 = perfect.
Recall and precision throw that away. NDCG gives each result a gain from its grade, divides it
by log2(rank + 1) so lower positions count less, and normalizes by the best possible ranking,
so a score of 1.0 means the ideal order. Linear gain uses the grade directly, as trec_eval does.
Exponential gain, 2^grade - 1, makes a perfect document worth much more than a partial one.
I have to report which one I used, because the numbers aren't comparable.

**Q24. How do you handle documents the annotators never judged?**

**A:** The standard choice is to treat them as non-relevant, which is what my code does. The
risk is that a new system finds good documents nobody judged, and gets penalized for it. So I
track `judged@k`, the share of the top k that has a judgment. If it drops when I compare a new
system to the old one, the metric is unreliable and I need more judgments before I trust it.

**Q25. A new reranker improves NDCG@10 from 0.61 to 0.63. Do you ship it?**

**A:** Not on that number alone. First I check whether the difference is real or noise. I run a
paired test on the same queries, because the per-query differences are much less noisy than
comparing two means. I use a randomization test: if the systems were equal, flipping the sign
of each query's difference wouldn't matter, so the p-value is how often a random flip gives a
difference as large as the observed one. I also report a bootstrap confidence interval on the
mean. Then I check latency, cost and online signals before shipping.

**Q26. Why a paired test and not two independent intervals?**

**A:** Query difficulty varies far more than the gap between two systems. Some queries are easy
for every system and some are hard for every system. Pairing cancels that difficulty out and
only looks at the per-query difference, so it detects real improvements with far fewer queries.
Two overlapping confidence intervals don't prove there is no difference.

**Q27. What goes wrong when you evaluate many metrics or many experiments?**

**A:** With a 0.05 threshold, one in twenty comparisons looks significant by chance. If I test
ten metrics over many experiments, I will find false wins. I pick one primary metric before the
experiment, and I correct for multiple comparisons, for example with Holm-Bonferroni, when I
really need several.

**Q28. Why do run scores need a tie-breaking rule in evaluation?**

**A:** Many retrievers return equal scores, especially after quantization or with BM25 on short
documents. If ties are ordered by input order, the same run can give different metrics on
different machines. My `evaluate` sorts by descending score and then by document ID, so results
are reproducible. Libraries such as trec_eval also use a fixed rule, though not always the same
one, so I note the rule when comparing numbers across tools.

**Q29. How does this code differ from libraries like `ranx` or `pytrec_eval`?**

**A:** The definitions match, and checking against them on the same qrels and run is the best
way to validate my numbers. That is why `evaluate` has a `backend="ranx"` option, and a test
that compares both backends on 300 random graded queries with ties and missing queries. The
differences are scope and how the data is held. Those libraries offer many more metrics such as
MAP and bpref, and statistical tests with multiple-comparison correction. Mine is pure Python,
so each query costs O(judgments + k) and the statistics cost O(resamples × queries).

**Q30. You added ranx as a backend. Is it faster, and what went wrong while integrating it?**

**A:** It was not faster, and I measured that instead of assuming it. On 20,000 to 60,000 queries
with 100 results each, my plain Python backend was quicker, for example 1.9 seconds against 4.1
for 60,000 queries. A profile showed why: the metric math is cheap, and the time goes into
validating and converting Python dicts, which ranx has to redo into its own format. ranx pays off
when you keep its `Qrels` and `Run` objects and score many times, for example in a parameter
sweep. So I kept it as a cross-check, not a speed-up. Four things bit me during integration:
1. With plain dicts, ranx scores queries in dict insertion order and never checks that qrels and
   run have the same order. I first assumed sorted order and got wrong scores on about half the
   queries, which the parity test caught. Never trust a library's ordering without a test.
2. It raises an error on an empty ranking, so I score those queries as 0 myself.
3. With a single metric and `return_mean=False` it fails with a `KeyError`, a bug in its loop
   variable, so I pass the metric name twice.
4. It hashes document IDs and its IDs are strings, so I reject IDs that collide as strings.
