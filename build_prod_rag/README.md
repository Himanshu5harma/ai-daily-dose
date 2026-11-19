# Production RAG (work in progress)

A LangChain sandbox for building a production-style RAG pipeline step by step: document
loading, chunking, a vector store, and generation. It's a separate uv workspace member with
its own dependencies, so it doesn't add LangChain to the core `ai_engineering` package.

| File | Contents |
| --- | --- |
| `main.py` | LLM smoke test (Groq) |
| `document_loaders.py` | Text and PDF loading with `TextLoader` and `PyPDFLoader` |

## Setup

From the repository root:

```bash
uv sync --package build-prod-rag
uv run --package build-prod-rag python build_prod_rag/main.py
```

Put API keys in a `.env` file (it's gitignored): `GROQ_API_KEY` for `main.py`, and
`OPENAI_API_KEY` for the OpenAI models and embeddings.

## Documents

`docs/` is gitignored. Drop your own PDFs there. Don't commit copyrighted documents such as
ISO standards. `document_loaders.py` expects `docs/iso27001.pdf` by default; point it at
any PDF you have.
