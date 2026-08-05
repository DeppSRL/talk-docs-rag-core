"""
Abstract interface for keyword-based search implementations.

This module provides the abstract base class for keyword index implementations
that will be used in hybrid search architecture.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class KeywordSearchResult:
    """Result from keyword-based search."""

    content: str
    score: float  # BM25 or similar relevance score
    source: str
    chunk_id: str
    metadata: dict[str, Any]


class KeywordIndex(ABC):
    """Abstract interface for keyword-based search implementations."""

    @abstractmethod
    async def index_document(self, content: str, chunk_id: str, source: str, metadata: dict[str, Any]) -> None:
        """Add a document chunk to the keyword index."""
        pass

    @abstractmethod
    async def search(
        self, query: str, top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[KeywordSearchResult]:
        """Execute keyword-based search."""
        pass

    @abstractmethod
    async def delete_document(self, chunk_id: str) -> None:
        """Remove document from keyword index."""
        pass

    @abstractmethod
    async def update_document(self, chunk_id: str, content: str, metadata: dict[str, Any]) -> None:
        """Update existing document in keyword index."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the keyword index (create schema, etc.)."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up and close the keyword index."""
        pass

    @abstractmethod
    async def clear_index(self) -> None:
        """Delete all documents from the index (used by --recreate)."""
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the keyword index."""
        pass
