"""Dense retriever using SentenceTransformer + FAISS (cosine similarity)."""

import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from .base import BaseRetriever
from config import DENSE_EMBEDDING_MODEL, EXISTING_FAISS_INDEX, EXISTING_FAISS_METADATA


class FaissRetriever(BaseRetriever):
    def __init__(self, embedding_model_name: str = DENSE_EMBEDDING_MODEL,
                 load_existing_index: bool = False):
        self.model_name = embedding_model_name
        self.model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.metadata = []
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.load_existing = load_existing_index

    def build_index(self, documents: List[str], metadata: List[Dict[str, Any]] = None) -> None:
        """If load_existing is True, load from disk instead of building."""
        if self.load_existing:
            self._load_existing()
            return

        print(f"[{self.name}] Building FAISS index with {len(documents)} documents...")
        self.metadata = metadata if metadata is not None else [{"doc_id": i, "text": doc} for i, doc in enumerate(documents)]

        embeddings = self.model.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32
        ).astype(np.float32)

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        print(f"[{self.name}] Index built, {self.index.ntotal} vectors.")

    def _load_existing(self):
        """Load pre‑built FAISS index and metadata from disk."""
        print(f"[{self.name}] Loading existing FAISS index from {EXISTING_FAISS_INDEX}")
        self.index = faiss.read_index(EXISTING_FAISS_INDEX)
        print(f"[{self.name}] Loading metadata from {EXISTING_FAISS_METADATA}")
        with open(EXISTING_FAISS_METADATA, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        print(f"[{self.name}] Loaded {self.index.ntotal} vectors and {len(self.metadata)} metadata entries.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None:
            return []
        q_emb = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            entry = self.metadata[idx].copy()
            entry["score"] = float(score)
            entry["rank"] = len(results) + 1
            results.append(entry)
        return results

    @property
    def name(self) -> str:
        return f"FAISS_{self.model_name.replace('/', '_')}"