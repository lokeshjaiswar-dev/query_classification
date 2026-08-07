# ============================================
# FILE: vector_store.py
# PURPOSE: In-memory FAISS vector store with employee metadata
# ============================================

from __future__ import annotations

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ingest import Chunk, load_chunks


class VectorStore:
    """Holds the embedding model, a FAISS index, and the parallel chunk list."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._model is None:
            print(f"[vectorstore] loading embedding model {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Convert texts to embeddings."""
        vecs = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,  # cosine similarity via inner product
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype="float32")

    def build(self, chunks: list[Chunk] | None = None) -> int:
        """Build the in-memory index from the given chunks."""
        self.chunks = chunks if chunks is not None else load_chunks()
        
        if not self.chunks:
            self.index = None
            return 0
        
        print(f"[vectorstore] 🔄 Building embeddings for {len(self.chunks)} chunks...")
        embeddings = self._embed([c.text for c in self.chunks])
        
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self.index = index
        
        print(f"[vectorstore] ✅ Index built with {index.ntotal} vectors")
        return index.ntotal

    def search(self, query: str, k: int = 5, employee_id: str | None = None) -> list[tuple[Chunk, float]]:
        """
        Search for the top k most relevant chunks.
        Optionally filter by employee_id.
        """
        if self.index is None or not self.chunks:
            return []
        
        # If employee_id is specified, filter chunks first
        if employee_id:
            filtered_chunks = [c for c in self.chunks if c.employee_id == employee_id]
            if not filtered_chunks:
                print(f"[vectorstore] ⚠️ No chunks found for employee: {employee_id}")
                return []
            
            # Build temporary index for filtered chunks
            embeddings = self._embed([c.text for c in filtered_chunks])
            temp_index = faiss.IndexFlatIP(embeddings.shape[1])
            temp_index.add(embeddings)
            
            k = max(1, min(k, len(filtered_chunks)))
            q = self._embed([query])
            scores, ids = temp_index.search(q, k)
            
            results: list[tuple[Chunk, float]] = []
            for idx, score in zip(ids[0], scores[0]):
                if idx < 0:
                    continue
                results.append((filtered_chunks[idx], float(score)))
            
            return results
        
        # Search all chunks
        k = max(1, min(k, len(self.chunks)))
        q = self._embed([query])
        scores, ids = self.index.search(q, k)
        
        results: list[tuple[Chunk, float]] = []
        for idx, score in zip(ids[0], scores[0]):
            if idx < 0:
                continue
            results.append((self.chunks[idx], float(score)))
        
        return results

    def search_by_employee(self, employee_id: str, k: int = 10) -> list[tuple[Chunk, float]]:
        """Get chunks for a specific employee (ordered by relevance if query provided)."""
        chunks = [c for c in self.chunks if c.employee_id == employee_id]
        return [(c, 1.0) for c in chunks[:k]]

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index is not None else 0
    
    def get_employees(self) -> list[tuple[str, str]]:
        """Get list of all employees with their names."""
        seen = set()
        employees = []
        for chunk in self.chunks:
            if chunk.employee_id and chunk.employee_id not in seen:
                seen.add(chunk.employee_id)
                employees.append((chunk.employee_id, chunk.employee_name))
        return sorted(employees)