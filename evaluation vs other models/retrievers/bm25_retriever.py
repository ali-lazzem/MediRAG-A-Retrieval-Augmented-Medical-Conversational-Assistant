"""BM25 sparse retriever using rank_bm25."""

import re
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
from .base import BaseRetriever


class BM25Retriever(BaseRetriever):
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or self._default_tokenizer
        self.bm25 = None
        self.metadata = []

    @staticmethod
    def _default_tokenizer(text: str) -> List[str]:
        """Simple tokenizer: lowercase, split on non-alphanumeric."""
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens

    def build_index(self, documents: List[str], metadata: List[Dict[str, Any]] = None) -> None:
        print(f"[{self.name}] Building BM25 index with {len(documents)} documents...")
        self.metadata = metadata if metadata is not None else [{"doc_id": i, "text": doc} for i, doc in enumerate(documents)]
        tokenized_docs = [self.tokenizer(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        print(f"[{self.name}] BM25 index ready.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.bm25 is None:
            return []
        tokenized_query = self.tokenizer(query)
        scores = self.bm25.get_scores(tokenized_query)
        # Get top_k indices by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            entry = self.metadata[idx].copy()
            entry["score"] = float(scores[idx])
            entry["rank"] = len(results) + 1
            results.append(entry)
        return results

    @property
    def name(self) -> str:
        return "BM25"