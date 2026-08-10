# ============================================
# FILE: vector_store.py
# PURPOSE: Hybrid Search + Cross-Encoder Reranking + Temporal Sorting
# ============================================

from __future__ import annotations

import re
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from ingest import Chunk, load_chunks


class VectorStore:
    """
    Combines FAISS (dense semantic vectors), BM25 (sparse exact keyword matching),
    and a Cross-Encoder Reranker with temporal logic for precise retrieval.
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model

        self._embed_model: SentenceTransformer | None = None
        self._reranker: CrossEncoder | None = None

        self.index: faiss.Index | None = None
        self.vectors: np.ndarray | None = None
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None

    @property
    def embed_model(self) -> SentenceTransformer:
        """Lazy-load sentence transformer embedding model."""
        if self._embed_model is None:
            print(f"[vectorstore] 🔄 Loading embedding model: {self.embedding_model_name}")
            self._embed_model = SentenceTransformer(self.embedding_model_name)
        return self._embed_model

    @property
    def reranker(self) -> CrossEncoder:
        """Lazy-load Cross-Encoder reranker model."""
        if self._reranker is None:
            print(f"[vectorstore] 🔄 Loading Cross-Encoder reranker: {self.reranker_model_name}")
            self._reranker = CrossEncoder(self.reranker_model_name)
        return self._reranker

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Convert texts into normalized vector embeddings."""
        vecs = self.embed_model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype="float32")

    def _tokenize(self, text: str) -> list[str]:
        """Convert text into lowercase word tokens for BM25 keyword matching."""
        return re.findall(r"\w+", text.lower())

    def build(self, chunks: list[Chunk] | None = None) -> int:
        """Build both FAISS vector index and BM25 keyword index."""
        self.chunks = chunks if chunks is not None else load_chunks()

        if not self.chunks:
            self.index = None
            self.bm25 = None
            return 0

        print(f"[vectorstore] 🔄 Building FAISS embeddings & BM25 index for {len(self.chunks)} chunks...")

        # 1. Build Dense FAISS Index
        self.vectors = self._embed([c.text for c in self.chunks])
        index = faiss.IndexFlatIP(self.vectors.shape[1])
        index.add(self.vectors)
        self.index = index

        # 2. Build Sparse BM25 Keyword Index
        tokenized_corpus = [self._tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"[vectorstore] ✅ Hybrid index built successfully with {index.ntotal} chunks")
        return index.ntotal

    def _hybrid_candidate_retrieval(
        self,
        query: str,
        fetch_k: int = 20,
        employee_id: str | None = None
    ) -> list[Chunk]:
        """Stage 1: Retrieve top candidate chunks via BM25 + FAISS Reciprocal Rank Fusion (RRF)."""
        candidate_indices = list(range(len(self.chunks)))
        if employee_id:
            candidate_indices = [
                i for i, c in enumerate(self.chunks)
                if c.employee_id.lower() == employee_id.lower()
            ]
            if not candidate_indices:
                return []

        # A. FAISS Dense Vector Ranks
        q_vec = self._embed([query])
        if employee_id:
            sub_vectors = self.vectors[candidate_indices]
            sub_index = faiss.IndexFlatIP(sub_vectors.shape[1])
            sub_index.add(sub_vectors)
            top_vector_k = min(len(candidate_indices), max(fetch_k, 20))
            _, sub_ids = sub_index.search(q_vec, top_vector_k)
            vector_ranks = {candidate_indices[sub_id]: rank + 1 for rank, sub_id in enumerate(sub_ids[0]) if sub_id >= 0}
        else:
            top_vector_k = min(len(self.chunks), max(fetch_k, 20))
            _, vector_ids = self.index.search(q_vec, top_vector_k)
            vector_ranks = {idx: rank + 1 for rank, idx in enumerate(vector_ids[0]) if idx >= 0}

        # B. BM25 Sparse Keyword Ranks
        query_tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)

        if employee_id:
            candidate_scores = [(idx, bm25_scores[idx]) for idx in candidate_indices]
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            top_bm25_items = candidate_scores[:max(fetch_k, 20)]
        else:
            top_bm25_indices = np.argsort(bm25_scores)[::-1][:max(fetch_k, 20)]
            top_bm25_items = [(idx, bm25_scores[idx]) for idx in top_bm25_indices]

        bm25_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(top_bm25_items)}

        # C. Reciprocal Rank Fusion (RRF)
        all_candidate_ids = set(vector_ranks.keys()).union(set(bm25_ranks.keys()))
        rrf_scores = {}

        for idx in all_candidate_ids:
            v_rank = vector_ranks.get(idx, 1000)
            b_rank = bm25_ranks.get(idx, 1000)
            rrf_scores[idx] = (1.0 / (60.0 + v_rank)) + (1.0 / (60.0 + b_rank))

        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:fetch_k]
        return [self.chunks[idx] for idx, _ in sorted_indices]

    def search(
        self,
        query: str,
        k: int = 5,
        employee_id: str | None = None,
        fetch_k: int = 20
    ) -> list[tuple[Chunk, float]]:
        """
        Stage 1: Pull top `fetch_k` candidate chunks via Hybrid Search.
        Stage 2: Cross-Encoder dynamically reranks candidates based on query relevance.
        Stage 3: Applies temporal ordering if chronological terms ('first', 'latest') are detected.
        """
        if self.index is None or not self.chunks or self.bm25 is None:
            return []

        # 1. Fetch broad candidate set (e.g., top 20)
        candidate_chunks = self._hybrid_candidate_retrieval(query, fetch_k=fetch_k, employee_id=employee_id)
        if not candidate_chunks:
            return []

        # 2. Dynamic Reranking via Cross-Encoder
        pairs = [[query, chunk.text] for chunk in candidate_chunks]
        scores = self.reranker.predict(pairs)
        scored_chunks = list(zip(candidate_chunks, scores))

        # 3. Temporal / Year Sorting for Chronological Queries
        query_lower = query.lower()
        is_first_query = any(w in query_lower for w in ["first", "initial", "earliest", "start", "starting"])
        is_latest_query = any(w in query_lower for w in ["latest", "recent", "last", "current"])

        if is_first_query or is_latest_query:
            def extract_year(chunk: Chunk) -> int:
                # Extract 4-digit year from filename or chunk text (e.g., increment_letter_2020.pdf -> 2020)
                match = re.search(r"\b(20\d{2})\b", chunk.filename + " " + chunk.text)
                return int(match.group(1)) if match else 9999

            # Keep only relevant chunks filtered by reranker (score threshold)
            relevant_chunks = [item for item in scored_chunks if item[1] > 0.0]
            if relevant_chunks:
                # Sort relevant chunks by year: Ascending for "first", Descending for "latest"
                relevant_chunks.sort(key=lambda x: extract_year(x[0]), reverse=is_latest_query)
                remaining_chunks = [item for item in scored_chunks if item[1] <= 0.0]
                scored_chunks = relevant_chunks + remaining_chunks
            else:
                scored_chunks.sort(key=lambda x: x[1], reverse=True)
        else:
            # Standard sort by reranker relevance score
            scored_chunks.sort(key=lambda x: x[1], reverse=True)

        return [(chunk, float(score)) for chunk, score in scored_chunks[:k]]

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