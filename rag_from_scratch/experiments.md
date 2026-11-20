# Experiments

Corpus: 20 Python 3.12/3.13 docs pages (~931k characters, 450 sections with content).
Questions: 21 answerable DocLens dev questions (6 exact, 8 conceptual, 7 multi).
Metrics: evidence recall@5 and MRR@5 from `ai_engineering.evaluations`.

<!-- Template for each entry:

## E<n>: <question>

- **Question:**
- **Setup:** (chunker + size/overlap, retriever, model + revision, k, date)
- **Numbers:** (table)
- **What surprised me:** (my words)
- **Decision:** (my words)
-->

## E1: Does section-based chunking beat fixed-size chunking, and does chunk size change that?

- **Question:** For Python docs, does splitting along sections (never crossing a section
  boundary, whole blocks packed up to a limit) retrieve evidence better than slicing text
  every N characters, at 500 vs 2000 characters?
- **Setup:** 20 pages (3.12/3.13), 21 answerable dev questions, recall@5 over evidence
  groups and MRR@5. Fixed chunker: overlap 0. Section chunker: greedy block packing.
  Candidates limited to the release line(s) named in the question. Dense =
  `nomic-ai/modernbert-embed-base` @ d556a88e, normalized, cosine. Run 2026-11-19,
  `results/results.json`.
- **Numbers (overall recall@5 / MRR@5):**

  | chunker | size | chunks | median chars | dense | BM25 | hybrid |
  |---|---|---|---|---|---|---|
  | fixed | 500 | 1872 | 500 | 0.93 / 0.85 | 0.83 / 0.75 | 0.90 / 0.81 |
  | section | 500 | 2541 | 405 | 0.90 / 0.90 | 0.81 / 0.71 | 0.95 / 0.79 |
  | fixed | 2000 | 475 | 2000 | 0.88 / 0.80 | 0.95 / 0.83 | 0.95 / 0.85 |
  | section | 2000 | 753 | 1370 | 0.98 / 0.90 | 0.98 / 0.86 | 0.98 / 0.90 |

  Section-500 produced 143 chunks under 100 characters (fixed-500: 2). Total indexed
  characters are about the same for both chunkers (about 930k) because overlap is 0.
- **Caveats:** one query is about 0.05 on the overall column. Recall is near ceiling.
- **What surprised me:** _(your words)_
- **Decision:** _(your words)_

## E2: Dense vs BM25 vs hybrid (RRF), by query type

- **Question:** Does hybrid retrieval beat the better of dense and BM25, and does the
  winner depend on query type (exact-name vs conceptual vs multi-evidence)?
- **Setup:** Same corpus, questions, metrics and embedding model as E1. BM25: own
  implementation, `k1=1.5`, `b=0.75`, tokenizer `\w+` lowercased. Hybrid: RRF `k=60` over
  the top 50 of each retriever. Version filter on.
- **Numbers (section-2000, the strongest chunking; recall@5 / MRR@5):**

  | method | exact (6) | conceptual (8) | multi (7) | ALL (21) |
  |---|---|---|---|---|
  | dense | 1.00 / 1.00 | 1.00 / 0.88 | 0.93 / 0.86 | 0.98 / 0.90 |
  | BM25 | 1.00 / 0.83 | 1.00 / 0.88 | 0.93 / 0.86 | 0.98 / 0.86 |
  | hybrid | 1.00 / 1.00 | 1.00 / 0.88 | 0.93 / 0.83 | 0.98 / 0.90 |

  At 500-character chunks the gap is larger: BM25 recall on exact queries is 0.83 vs 1.00
  for dense (fixed and section), and overall 0.83 / 0.81 vs 0.93 / 0.90.
- **Caveats:** the "exact" questions are full sentences containing API names, not bare
  identifiers. n=6 for exact. Version filter makes retrieval easier than unfiltered.
- **What surprised me:** _(your words)_
- **Decision:** _(your words)_
