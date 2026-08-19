from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Document metadata structure."""

    source_id: str
    name: str
    path: str
    size: int
    mime_type: str
    author: str
    created_date: datetime
    modified_date: datetime
    folder_id: str | None = None
    folder_path: str | None = None
    hash: str | None = None
    permissions: dict[str, Any] | None = None
    custom_properties: dict[str, Any] | None = None


class DocumentChunk(BaseModel):
    """Document chunk structure."""

    chunk_id: str
    content: str
    embedding: list[float] | None = None
    start_index: int
    end_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """Document structure."""

    source_id: str
    content: str
    metadata: DocumentMetadata
    chunks: list[DocumentChunk] = Field(default_factory=list)


class SearchResult(BaseModel):
    """Search result structure."""

    chunk_id: str
    content: str
    score: float
    document_metadata: DocumentMetadata
    chunk_metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Retrieval result structure."""

    query: str
    context_chunks: list[SearchResult]
    total_chunks: int
    filters_applied: dict[str, Any] = Field(default_factory=dict)
