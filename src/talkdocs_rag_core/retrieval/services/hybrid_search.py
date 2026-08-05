"""
Hybrid search service that combines vector and keyword-based search.

This service implements a hybrid search approach that merges results from
both vector similarity search and keyword-based search using configurable
ranking strategies.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from ..models.document import SearchResult
from ..vector_stores.base import VectorStore
from ..vector_stores.keyword_index import KeywordIndex, KeywordSearchResult


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search behavior."""

    vector_weight: float = 0.7  # Weight for vector search results (0.0 to 1.0)
    keyword_weight: float = 0.3  # Weight for keyword search results (0.0 to 1.0)
    min_vector_score: float = 0.0  # Minimum vector similarity score
    min_keyword_score: float = 0.0  # Minimum keyword relevance score
    normalization_method: str = "min_max"  # "min_max" or "z_score"
    merge_strategy: str = "rrf"  # "rrf" (Reciprocal Rank Fusion) or "weighted_sum"
    rrf_constant: int = 60  # Constant for RRF calculation


@dataclass
class HybridSearchResult:
    """Result from hybrid search combining vector and keyword results."""

    content: str
    score: float  # Combined hybrid score
    vector_score: float | None  # Original vector similarity score
    keyword_score: float | None  # Original keyword relevance score
    source: str
    chunk_id: str
    metadata: dict[str, Any]
    rank: int  # Final rank in hybrid results


class HybridSearchService:
    """Service for performing hybrid search combining vector and keyword search."""

    def __init__(
        self, vector_store: VectorStore, keyword_index: KeywordIndex, config: HybridSearchConfig | None = None
    ):
        """Initialize hybrid search service."""
        self.vector_store = vector_store
        self.keyword_index = keyword_index
        self.config = config or HybridSearchConfig()

        # Validate weights sum to 1.0
        total_weight = self.config.vector_weight + self.config.keyword_weight
        if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
            raise ValueError(f"Vector and keyword weights must sum to 1.0, got {total_weight}")

        # Validate weight ranges
        if not (0.0 <= self.config.vector_weight <= 1.0):
            raise ValueError(f"Vector weight must be between 0.0 and 1.0, got {self.config.vector_weight}")
        if not (0.0 <= self.config.keyword_weight <= 1.0):
            raise ValueError(f"Keyword weight must be between 0.0 and 1.0, got {self.config.keyword_weight}")

    async def search(
        self,
        query: str,
        top_k: int = 10,
        vector_top_k: int | None = None,
        keyword_top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[HybridSearchResult]:
        """Perform hybrid search combining vector and keyword results."""
        # Set default fetch sizes to get more results for better merging
        vector_top_k = vector_top_k or min(top_k * 2, 50)
        keyword_top_k = keyword_top_k or min(top_k * 2, 50)

        # Use empty dict for filters if None provided (VectorStore requires dict)
        search_filters = filters or {}

        # Perform both searches concurrently
        vector_task = asyncio.create_task(self.vector_store.search(query, search_filters, vector_top_k))
        keyword_task = asyncio.create_task(self.keyword_index.search(query, top_k=keyword_top_k, filters=filters))

        vector_results, keyword_results = await asyncio.gather(vector_task, keyword_task)

        # Filter results by minimum scores
        vector_results = [r for r in vector_results if r.score >= self.config.min_vector_score]
        keyword_results = [r for r in keyword_results if r.score >= self.config.min_keyword_score]

        # Merge and rank results
        if self.config.merge_strategy == "rrf":
            hybrid_results = self._merge_with_rrf(vector_results, keyword_results)
        else:  # weighted_sum
            hybrid_results = self._merge_with_weighted_sum(vector_results, keyword_results)

        # Return top_k results
        return hybrid_results[:top_k]

    def _merge_with_rrf(
        self, vector_results: list[SearchResult], keyword_results: list[KeywordSearchResult]
    ) -> list[HybridSearchResult]:
        """Merge results using Reciprocal Rank Fusion (RRF)."""
        # Create lookup maps for efficient merging
        chunk_to_vector = {r.chunk_id: (i + 1, r) for i, r in enumerate(vector_results)}
        chunk_to_keyword = {r.chunk_id: (i + 1, r) for i, r in enumerate(keyword_results)}

        # Get all unique chunk IDs
        all_chunk_ids = set(chunk_to_vector.keys()) | set(chunk_to_keyword.keys())

        hybrid_results = []

        for chunk_id in all_chunk_ids:
            rrf_score = 0.0
            vector_score = None
            keyword_score = None
            content = ""
            source = ""
            metadata = {}

            # Add RRF contribution from vector results
            if chunk_id in chunk_to_vector:
                rank, vector_result = chunk_to_vector[chunk_id]
                rrf_score += self.config.vector_weight / (self.config.rrf_constant + rank)
                vector_score = vector_result.score
                content = vector_result.content
                source = vector_result.document_metadata.name
                metadata = vector_result.chunk_metadata

            # Add RRF contribution from keyword results
            if chunk_id in chunk_to_keyword:
                rank, keyword_result = chunk_to_keyword[chunk_id]
                rrf_score += self.config.keyword_weight / (self.config.rrf_constant + rank)
                keyword_score = keyword_result.score
                if not content:  # Use keyword result if no vector result
                    content = keyword_result.content
                    source = keyword_result.source
                    metadata = keyword_result.metadata

            hybrid_results.append(
                HybridSearchResult(
                    content=content,
                    score=rrf_score,
                    vector_score=vector_score,
                    keyword_score=keyword_score,
                    source=source,
                    chunk_id=chunk_id,
                    metadata=metadata,
                    rank=0,  # Will be set after sorting
                )
            )

        # Sort by RRF score (descending)
        hybrid_results.sort(key=lambda x: x.score, reverse=True)

        # Set ranks
        for i, hybrid_result in enumerate(hybrid_results):
            hybrid_result.rank = i + 1

        return hybrid_results

    def _merge_with_weighted_sum(
        self, vector_results: list[SearchResult], keyword_results: list[KeywordSearchResult]
    ) -> list[HybridSearchResult]:
        """Merge results using weighted sum of normalized scores."""
        # Normalize scores
        normalized_vector = self._normalize_scores([r.score for r in vector_results])
        normalized_keyword = self._normalize_scores([r.score for r in keyword_results])

        # Create lookup maps with normalized scores
        chunk_to_vector = {
            r.chunk_id: (norm_score, r) for r, norm_score in zip(vector_results, normalized_vector, strict=False)
        }
        chunk_to_keyword = {
            r.chunk_id: (norm_score, r) for r, norm_score in zip(keyword_results, normalized_keyword, strict=False)
        }

        # Get all unique chunk IDs
        all_chunk_ids = set(chunk_to_vector.keys()) | set(chunk_to_keyword.keys())

        hybrid_results = []

        for chunk_id in all_chunk_ids:
            combined_score = 0.0
            vector_score = None
            keyword_score = None
            content = ""
            source = ""
            metadata = {}

            # Add weighted vector score
            if chunk_id in chunk_to_vector:
                norm_score, vector_result = chunk_to_vector[chunk_id]
                combined_score += self.config.vector_weight * norm_score
                vector_score = vector_result.score
                content = vector_result.content
                source = vector_result.document_metadata.name
                metadata = vector_result.chunk_metadata

            # Add weighted keyword score
            if chunk_id in chunk_to_keyword:
                norm_score, keyword_result = chunk_to_keyword[chunk_id]
                combined_score += self.config.keyword_weight * norm_score
                keyword_score = keyword_result.score
                if not content:  # Use keyword result if no vector result
                    content = keyword_result.content
                    source = keyword_result.source
                    metadata = keyword_result.metadata

            hybrid_results.append(
                HybridSearchResult(
                    content=content,
                    score=combined_score,
                    vector_score=vector_score,
                    keyword_score=keyword_score,
                    source=source,
                    chunk_id=chunk_id,
                    metadata=metadata,
                    rank=0,  # Will be set after sorting
                )
            )

        # Sort by combined score (descending)
        hybrid_results.sort(key=lambda x: x.score, reverse=True)

        # Set ranks
        for i, hybrid_result in enumerate(hybrid_results):
            hybrid_result.rank = i + 1

        return hybrid_results

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        """Normalize scores using the configured method."""
        if not scores:
            return []

        if self.config.normalization_method == "min_max":
            min_score = min(scores)
            max_score = max(scores)
            if max_score == min_score:
                return [1.0] * len(scores)  # All scores are equal
            return [(score - min_score) / (max_score - min_score) for score in scores]

        elif self.config.normalization_method == "z_score":
            mean_score = sum(scores) / len(scores)
            variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
            std_dev = variance**0.5
            if std_dev == 0:
                return [0.0] * len(scores)  # All scores are equal
            return [(score - mean_score) / std_dev for score in scores]

        else:
            raise ValueError(f"Unknown normalization method: {self.config.normalization_method}")
