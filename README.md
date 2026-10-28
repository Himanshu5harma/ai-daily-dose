# AI Daily Dose

Daily practice: LeetCode problems and small, tested AI engineering building blocks
(retrieval evaluation, RAG, agents). Python with FastAPI.

| Path | Contents |
| --- | --- |
| `ai_engineering/evaluations/` | Retrieval metrics |
| `ai_engineering/retrieval/` | Policy-aware retrieval API (FastAPI) |
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
- [x] `precision_at_k()`
- [ ] `reciprocal_rank_at_k()`
- [ ] First LeetCode solution

## License

[MIT](LICENSE)
