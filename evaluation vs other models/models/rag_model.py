"""Generic RAG model: retrieves context and prompts LLM with it."""

import requests
from typing import List, Dict, Any
from retrievers.base import BaseRetriever
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TEMPERATURE


class RAGModel:
    def __init__(self, retriever: BaseRetriever):
        self.retriever = retriever
        self.model_name = OLLAMA_MODEL
        self.base_url = OLLAMA_BASE_URL.rstrip('/')
        self.temperature = OLLAMA_TEMPERATURE

    def generate(self, question: str, top_k: int = 5) -> str:
        """Retrieve relevant chunks and generate answer."""
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        if not retrieved:
            # Fallback: pure LLM if no context
            prompt = f"Answer the following medical question as best you can (no context available).\n\nQuestion: {question}\n\nAnswer:"
        else:
            context_parts = []
            for i, chunk in enumerate(retrieved, 1):
                text = chunk.get("text", chunk.get("content", ""))
                if not text:
                    continue
                context_parts.append(f"[Chunk {i}]\n{text}")
            context_str = "\n\n".join(context_parts)
            prompt = (
                f"You are a medical assistant. Use ONLY the following context to answer the question. "
                f"If the context does not contain enough information, say so.\n\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {question}\n\n"
                f"Answer:"
            )

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": self.temperature}
        }
        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"].strip()
        except Exception as e:
            print(f"RAGModel error: {e}")
            return "Error: Could not generate answer."

    @property
    def name(self) -> str:
        return f"RAG_{self.retriever.name}"