"""C4b — cache semantica su collection Chroma dedicata.

Meccanica:
- ogni risposta *servita e non rifiutata* viene memorizzata con l'embedding della query,
  la risposta, le **citazioni originali** (chunk_id), i claims e il ``corpus_version``;
- al lookup, l'embedding della query nuova è confrontato (coseno) con le voci dello
  **stesso ``corpus_version``**: sopra ``cache_sim_threshold`` (prudente) è un HIT e si
  restituisce la risposta cachata con le sue citazioni originali;
- invalidazione: il filtro ``where={"corpus_version": ...}`` esclude di fatto le voci di
  versioni precedenti del corpus (cambiano hash → non vengono più pescate).

Collection creata con ``hnsw:space=cosine`` → ``similarità = 1 - distanza``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import chromadb

from config import RagConfig
from vendor.talkdocs.services.embeddings import EmbeddingService


@dataclass
class CacheHit:
    answer_text: str
    cited_chunk_ids: list[str]
    claims: list[dict]
    similarity: float
    matched_query: str


class SemanticCache:
    def __init__(self, cfg: RagConfig, embedding_service: EmbeddingService, chroma_client: chromadb.ClientAPI):
        self.cfg = cfg
        self.embedding_service = embedding_service
        self.collection = chroma_client.get_or_create_collection(
            cfg.chroma_collection_cache, metadata={"hnsw:space": "cosine"}
        )

    @staticmethod
    def _entry_id(corpus_version: str, query: str) -> str:
        return hashlib.sha256(f"{corpus_version}\n{query}".encode()).hexdigest()[:24]

    async def lookup(self, query: str, corpus_version: str) -> CacheHit | None:
        if self.collection.count() == 0:
            return None
        emb = await self.embedding_service.get_single_embedding(query)
        res = self.collection.query(
            query_embeddings=[emb],
            n_results=1,
            where={"corpus_version": corpus_version},
        )
        ids = res.get("ids") or [[]]
        if not ids[0]:
            return None

        distance = res["distances"][0][0]
        similarity = 1.0 - distance
        if similarity < self.cfg.cache_sim_threshold:
            return None

        meta = res["metadatas"][0][0]
        return CacheHit(
            answer_text=meta.get("answer_text", ""),
            cited_chunk_ids=json.loads(meta.get("cited_chunk_ids", "[]")),
            claims=json.loads(meta.get("claims", "[]")),
            similarity=similarity,
            matched_query=meta.get("query", ""),
        )

    async def store(
        self,
        query: str,
        corpus_version: str,
        answer_text: str,
        cited_chunk_ids: list[str],
        claims: list[dict],
    ) -> None:
        emb = await self.embedding_service.get_single_embedding(query)
        self.collection.upsert(
            ids=[self._entry_id(corpus_version, query)],
            embeddings=[emb],
            documents=[query],
            metadatas=[
                {
                    "query": query,
                    "corpus_version": corpus_version,
                    "answer_text": answer_text,
                    "cited_chunk_ids": json.dumps(cited_chunk_ids, ensure_ascii=False),
                    "claims": json.dumps(claims, ensure_ascii=False),
                }
            ],
        )

    def clear(self) -> None:
        existing = self.collection.get()
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])
