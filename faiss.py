"""In-memory FAISS vector store backed by a local sentence-transformers model."""
from __future__ import annotations

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .ingest import Chunk, load_chunks


class VectorStore:
    """Holds the embedding model, a FAISS index, and the parallel chunk list."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []

    @property
    def model(self) -> SentenceTransformer:
        # Lazy-load so importing the module is cheap.
        if self._model is None:
            print(f"[vectorstore] loading embedding model {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,  # cosine similarity via inner product
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype="float32")

    def build(self, chunks: list[Chunk] | None = None) -> int:
        """Build the in-memory index from the given (or freshly loaded) chunks."""
        self.chunks = chunks if chunks is not None else load_chunks()
        if not self.chunks:
            self.index = None
            return 0
        embeddings = self._embed([c.text for c in self.chunks])
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self.index = index
        print(f"[vectorstore] index built with {index.ntotal} vectors")
        return index.ntotal

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        if self.index is None or not self.chunks:
            return []
        k = max(1, min(k, len(self.chunks)))
        q = self._embed([query])
        scores, ids = self.index.search(q, k)
        results: list[tuple[Chunk, float]] = []
        for idx, score in zip(ids[0], scores[0]):
            if idx < 0:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index is not None else 0