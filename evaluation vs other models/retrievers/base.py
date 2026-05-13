"""Abstract base class for retrievers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseRetriever(ABC):
    @abstractmethod
    def build_index(self, documents: List[str], metadata: List[Dict[str, Any]] = None) -> None:
        """Build search index from a list of documents."""
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return top_k retrieved chunks with scores and metadata."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass