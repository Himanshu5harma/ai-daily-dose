# RAG retrieval from scratch

A one-day exercise: chunking, dense, BM25 and hybrid (RRF) retrieval over 20 Python
3.12/3.13 docs pages, measured with recall@5 and MRR from `ai_engineering.evaluations`.

## Who writes what

| File | Status | Author |
| --- | --- | --- |
| `chunking.py` (`fixed_size_spans`, `section_spans`) | stub | me |
| `dense.py` (`cosine_top_k`) | stub | me |
| `bm25.py` (`tokenize`, `BM25`) | stub | me |
| `fusion.py` (`reciprocal_rank_fusion`) | stub | me |
| `corpus.py`, `questions.py`, `embeddings.py`, `copy_corpus.py`, runner | plumbing | Claude |

## Setup

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -e ".[dev,rag]"
python -m rag_from_scratch.copy_corpus --doclens ../doclens
python -m pytest rag_from_scratch/tests -q
```

The tests live here, not in `tests/`, so the main suite stays green while the stubs
are unimplemented.

## Data

- `data/corpus/`: parsed pages copied from DocLens (`data/parsed/python`); Python docs
  content is © Python Software Foundation, under the PSF documentation license.
- `data/dev.json`: 24 DocLens dev questions. 21 are answerable: 6 exact, 8 conceptual
  (paraphrase), 7 multi-evidence. DocLens's `heldout.json` is deliberately not copied.
- Gold labels are `(source_id, anchor)` pairs. A chunk counts as evidence for an anchor
  when it overlaps at least 50 characters of a block with that anchor (or the whole
  block, if shorter); see `corpus.anchors_in_span`.

## Embeddings

`nomic-ai/modernbert-embed-base` at a pinned revision, loaded from DocLens's model cache
(`HF_HUB_OFFLINE=1` works). Queries get the `search_query: ` prefix and chunks get
`search_document: `. Vectors are L2-normalized, and chunk embeddings are cached in
`.cache/`.
