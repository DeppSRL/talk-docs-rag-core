"""
Whoosh-based implementation of the KeywordIndex interface.

Uses Whoosh's Italian LanguageAnalyzer (Snowball stemmer + stopwords + elision)
for proper Italian text analysis.
"""

import re
import shutil
from pathlib import Path
from typing import Any

try:
    from whoosh import fields, index
    from whoosh.analysis import LanguageAnalyzer
    from whoosh.qparser import OrGroup, QueryParser

    WHOOSH_AVAILABLE = True
except ImportError:
    WHOOSH_AVAILABLE = False

from .keyword_index import KeywordIndex, KeywordSearchResult


class WhooshKeywordIndex(KeywordIndex):
    """Whoosh-based implementation of KeywordIndex using Italian text analysis."""

    def __init__(self, index_dir: str = "data/whoosh_index"):
        if not WHOOSH_AVAILABLE:
            raise ImportError("Whoosh is not installed. Install it with: uv add whoosh")

        self.index_dir = Path(index_dir)
        self._index = None
        self._query_parser = None

    async def initialize(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)

        analyzer = LanguageAnalyzer("it")
        schema = fields.Schema(
            chunk_id=fields.ID(stored=True, unique=True),
            content=fields.TEXT(analyzer=analyzer, stored=True),
            source=fields.ID(stored=True),
            metadata=fields.STORED,
        )

        if not index.exists_in(str(self.index_dir)):
            self._index = index.create_in(str(self.index_dir), schema)
        else:
            self._index = index.open_dir(str(self.index_dir))

        assert self._index is not None
        self._query_parser = QueryParser("content", self._index.schema, group=OrGroup)

    async def index_document(self, content: str, chunk_id: str, source: str, metadata: dict[str, Any]) -> None:
        if not self._index:
            raise RuntimeError("Index not initialized. Call initialize() first.")

        writer = self._index.writer()
        try:
            writer.update_document(chunk_id=chunk_id, content=content, source=source, metadata=metadata)
            writer.commit()
        except Exception:
            writer.cancel()
            raise

    # DIVERGENZA dal commit pinnato (vedi NOTICE): aggiunta per l'ingest batch del banco.
    async def index_documents(self, items: list[dict[str, Any]]) -> None:
        """Indicizza N chunk con UN writer e UN commit.

        ``index_document`` committa per documento: un segmento per commit, merge sempre
        più costosi. Misurato su 13.670 chunk: 33,6 → 24,2 doc/s in calo, 10-20 minuti.
        Qui il commit è unico. Ogni item: ``chunk_id``, ``content``, ``source``, ``metadata``.
        """
        if not self._index:
            raise RuntimeError("Index not initialized. Call initialize() first.")
        if not items:
            return

        writer = self._index.writer(limitmb=256)
        try:
            for it in items:
                writer.update_document(
                    chunk_id=it["chunk_id"],
                    content=it["content"],
                    source=it["source"],
                    metadata=it["metadata"],
                )
            writer.commit()
        except Exception:
            writer.cancel()
            raise

    async def search(
        self, query: str, top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[KeywordSearchResult]:
        if not self._index or not self._query_parser:
            raise RuntimeError("Index not initialized. Call initialize() first.")

        if not query or not query.strip():
            return []

        # Strip characters that are special to Whoosh query parser
        clean_query = re.sub(r"[^\w\s]", " ", query, flags=re.UNICODE)
        try:
            parsed_query = self._query_parser.parse(clean_query)
        except Exception:
            return []

        with self._index.searcher() as searcher:
            hits = searcher.search(parsed_query, limit=top_k)
            max_score = hits[0].score if hits else 1.0

            return [
                KeywordSearchResult(
                    content=hit["content"],
                    score=hit.score / max_score if max_score > 0 else 0.0,
                    source=hit["source"],
                    chunk_id=hit["chunk_id"],
                    metadata=hit["metadata"] or {},
                )
                for hit in hits
            ]

    async def delete_document(self, chunk_id: str) -> None:
        if not self._index:
            raise RuntimeError("Index not initialized. Call initialize() first.")

        writer = self._index.writer()
        try:
            writer.delete_by_term("chunk_id", chunk_id)
            writer.commit()
        except Exception:
            writer.cancel()
            raise

    async def update_document(self, chunk_id: str, content: str, metadata: dict[str, Any]) -> None:
        source = metadata.get("source", "")
        await self.index_document(content, chunk_id, source, metadata)

    async def close(self) -> None:
        if self._index:
            self._index.close()
            self._index = None
            self._query_parser = None

    async def clear_index(self) -> None:
        if self._index:
            await self.close()
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
        await self.initialize()

    def get_stats(self) -> dict[str, Any]:
        if not self._index:
            return {"total_documents": 0}

        with self._index.searcher() as searcher:
            count = searcher.doc_count()

        return {
            "total_documents": count,
            "index_dir": str(self.index_dir),
            "index_size_bytes": sum(f.stat().st_size for f in self.index_dir.iterdir() if f.is_file()),
        }
