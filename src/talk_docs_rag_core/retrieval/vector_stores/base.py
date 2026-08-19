from abc import ABC, abstractmethod
from typing import Any

from ..models.document import Document, RetrievalResult, SearchResult


class VectorStore(ABC):
    """Abstract base class for vector store implementations."""

    @abstractmethod
    async def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the vector store."""
        pass

    @abstractmethod
    async def update_document(self, doc_id: str, document: Document) -> bool:
        """Update a document in the vector store."""
        pass

    @abstractmethod
    async def delete_documents(self, doc_ids: list[str]) -> bool:
        """Delete documents from the vector store."""
        pass

    @abstractmethod
    async def search(
        self, query: str, filters: dict[str, Any], top_k: int, min_relevance_score: float = 0.0
    ) -> list[SearchResult]:
        """Search for similar documents."""
        pass

    @abstractmethod
    async def get_document_metadata(self, doc_id: str) -> dict[str, Any] | None:
        """Get metadata for a specific document."""
        pass

    @abstractmethod
    def get_all_document_ids(self) -> list[str]:
        """Get all unique document IDs in the vector store."""
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the vector store."""
        pass


class RetrievalEngine(ABC):
    """Abstract base class for retrieval engines."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    @abstractmethod
    async def retrieve_context(self, query: str, filters: dict[str, Any], max_chunks: int, **kwargs) -> RetrievalResult:
        """Retrieve context for a query."""
        pass


class BasicRetrievalEngine(RetrievalEngine):
    """Basic retrieval engine using simple vector similarity search."""

    async def retrieve_context(
        self, query: str, filters: dict[str, Any], max_chunks: int, min_relevance_score: float = 0.0, **kwargs
    ) -> RetrievalResult:
        """Retrieve context using basic vector similarity search."""
        search_results = await self.vector_store.search(query, filters, max_chunks, min_relevance_score)

        filtered_results = search_results

        return RetrievalResult(
            query=query, context_chunks=filtered_results, total_chunks=len(filtered_results), filters_applied=filters
        )
