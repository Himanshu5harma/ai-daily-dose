# AI Daily Dose

Daily practice: LeetCode problems and small, tested AI engineering building blocks
(retrieval evaluation, RAG, agents). Python with FastAPI.

| Path | Contents |
| --- | --- |
| `ai_engineering/evaluations/` | Retrieval metrics |
| `ai_engineering/retrieval/` | Policy-aware retrieval API (FastAPI) |
| `ai_engineering/support/` | Tenant-aware support assistant, bounded tools, and persistent chat |
| `rag_from_scratch/` | Chunking, dense, BM25 and hybrid retrieval exercise (in progress) |
| `build_prod_rag/` | LangChain production RAG sandbox (in progress, separate uv package) |
| `examples/` | Runnable examples |
| `leetcode/` | Daily DSA practice |
| `tests/` | Pytest tests |
| `journal/` | Daily notes |

## Setup

Requires Python 3.10+. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

The RAG work needs extra dependencies. See the [rag_from_scratch notes](rag_from_scratch/README.md)
(`pip install -e '.[dev,rag]'`) and the [build_prod_rag notes](build_prod_rag/README.md), which
uses uv (`uv sync --package build-prod-rag`).

## Example

```python
from ai_engineering.evaluations import recall_at_k

score = recall_at_k(
    relevant_ids={"doc-1", "doc-3"},
    retrieved_ids=["doc-1", "doc-2", "doc-3"],
    k=2,
)
print(score)  # 0.5
```

See the [evaluation notes](ai_engineering/evaluations/README.md) for the metric definition and edge cases, and the [retrieval notes](ai_engineering/retrieval/README.md) for the `POST /retrieve` policy and API contract. Revision questions are in [interview_questions.md](interview_questions.md). CI runs tests on Python 3.10, 3.12 and 3.14, plus Ruff lint and format checks.

## Roadmap

- [x] `recall_at_k()`
- [x] Policy-aware `POST /retrieve` API
- [x] [Support assistant](ai_engineering/support/README.md) with tenant isolation, deadlines, tool authorization, and failure recovery
- [x] `precision_at_k()`
- [x] `reciprocal_rank_at_k()`
- [x] `ndcg_at_k()`, batch `evaluate()` and significance tests
- [x] Optional ranx backend, TREC file readers and an [evaluation example](examples/evaluate_at_scale.py)
- [ ] RAG from scratch: chunking, dense, BM25 and RRF hybrid retrieval, measured on DocLens dev questions
- [ ] Production RAG with LangChain: loaders, vector store, generation

## License

[MIT](LICENSE)
