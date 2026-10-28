# 🚀 AI Daily Dose

### One Problem. One Build. Every Day.

My daily engineering practice repository, focused on becoming a stronger **Senior Full-Stack AI Engineer / Applied AI Engineer**.

The goal is simple: improve algorithmic problem-solving skills while building real, production-oriented AI engineering capabilities.

## 🎯 What I'm Practicing

**LeetCode & DSA**

- Arrays, Hash Maps, Trees, Graphs, Dynamic Programming
- Algorithm optimization and complexity analysis
- Technical interview problem-solving

**Applied AI Engineering**

- Python and FastAPI
- Retrieval-Augmented Generation (RAG)
- Embeddings and vector search
- LLM evaluation and testing
- AI agents and tool calling
- AI system design, reliability and observability

## 🛠 Tech Stack

Python · FastAPI · Pytest · React · LLM APIs · Vector Databases · Docker · GitHub Actions

Tools and frameworks will be introduced as projects require them. The current project uses Python, Pytest, Ruff and GitHub Actions; it has no runtime dependencies.

## 📂 Repository Organization

| Directory | Purpose |
| --- | --- |
| `leetcode/` | Algorithm solutions organized by pattern |
| `ai_engineering/` | Practical AI engineering implementations |
| `tests/` | Automated tests and edge cases |
| `journal/` | Daily engineering notes and learnings |

Add topic directories as challenges require them.

## Getting Started

Requires Python 3.10 or newer. Run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

### First engineering build: Recall@K

```python
from ai_engineering.evaluations import recall_at_k

score = recall_at_k(
    relevant_ids={"doc-1", "doc-3"},
    retrieved_ids=["doc-1", "doc-2", "doc-3"],
    k=2,
)
print(score)  # 0.5
```

See the [evaluation notes](ai_engineering/evaluations/README.md) for the metric definition, edge cases and trade-offs. CI runs tests on Python 3.10, 3.12 and 3.14 and checks lint and formatting.

## 📅 Daily Practice

Every day, I aim to:

1. Solve one LeetCode problem.
2. Implement one meaningful engineering improvement.
3. Write tests and understand edge cases.
4. Document one learning or technical trade-off.
5. Push meaningful progress to GitHub.

Use the [journal template](journal/TEMPLATE.md) for notes. Check [LeetCode](https://leetcode.com/problemset/) for the official daily problem.

## 🧠 Engineering Principles

- Understand before implementing.
- Prefer reliable, tested code over clever shortcuts.
- Measure AI quality instead of relying on intuition.
- Consider performance, security, cost and maintainability.
- Document failures and lessons learned.
- Never commit API keys, tokens, `.env` files, proprietary code or confidential data.

Use descriptive commits, such as `feat(evals): implement recall at k`.

## 📈 Progress

| Engineering challenge | Status |
| --- | --- |
| Implement `recall_at_k()` with tests | Implemented |
| Implement `precision_at_k()` with tests | Planned |
| Implement `reciprocal_rank_at_k()` with tests | Planned |

These are engineering assignments, not official LeetCode daily problems or claims of completed daily practice. No LeetCode submissions are recorded yet.

This repository is an ongoing learning journey. Implementations will evolve as my engineering skills improve.

**Consistency over intensity. Quality over commit count.**

## License

[MIT](LICENSE)
