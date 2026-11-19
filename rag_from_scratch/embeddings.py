"""Thin wrapper around a local sentence-transformers model, with an on-disk cache.

Default model: ``nomic-ai/modernbert-embed-base`` (already downloaded by DocLens).
It is trained with task prefixes, so queries get ``"search_query: "`` and chunks get
``"search_document: "``. ``all-MiniLM-L6-v2`` needs no prefixes.

Set ``RAG_MODEL_CACHE`` to reuse a model directory; it defaults to DocLens's
``../doclens/data/models`` when that exists.
"""

import hashlib
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(__file__).resolve().parent / ".cache"

PREFIXES = {
    "nomic-ai/modernbert-embed-base": ("search_query: ", "search_document: "),
    "nomic-ai/nomic-embed-text-v1.5": ("search_query: ", "search_document: "),
}
DEFAULT_MODEL = "nomic-ai/modernbert-embed-base"
# Pinned commits, so results are reproducible and the DocLens cache (which has no
# "main" ref) can be used offline.
REVISIONS = {
    "nomic-ai/modernbert-embed-base": "d556a88e332558790b210f7bdbe87da2fa94a8d8",
}


def _model_cache() -> str | None:
    if env := os.environ.get("RAG_MODEL_CACHE"):
        return env
    doclens = REPO_ROOT.parent / "doclens" / "data" / "models"
    return str(doclens) if doclens.is_dir() else None


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 16):
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        self.model_name = model_name
        self.batch_size = batch_size
        self.query_prefix, self.doc_prefix = PREFIXES.get(model_name, ("", ""))
        self._model = SentenceTransformer(
            model_name,
            revision=REVISIONS.get(model_name),
            cache_folder=_model_cache(),
            device="cpu",
        )

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 64,
        )
        return vectors.astype(np.float32)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts])

    def embed_documents(self, texts: list[str], cache: bool = True) -> np.ndarray:
        """Embed chunk texts; results are cached by model + exact texts."""
        key = hashlib.sha256(
            "\x1f".join([self.model_name, self.doc_prefix, *texts]).encode("utf-8")
        ).hexdigest()[:16]
        path = CACHE_DIR / f"{self.model_name.replace('/', '__')}-{key}.npy"
        if cache and path.exists():
            return np.load(path)
        vectors = self._encode([self.doc_prefix + t for t in texts])
        if cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(path, vectors)
        return vectors
