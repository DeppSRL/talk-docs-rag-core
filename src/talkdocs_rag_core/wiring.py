"""C2 wiring — costruzione dei componenti dal ``RagConfig`` iniettato.

Un solo punto dove il core vendored viene istanziato con la config della run. Niente
singleton: ogni factory riceve ``cfg`` esplicito, così l'eval può spazzare parametri.
"""

from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI

from talkdocs_rag_core.config import RagConfig
from talkdocs_rag_core.rag.guard import TermStats
from talkdocs_rag_core.retrieval.services.embeddings import EmbeddingService
from talkdocs_rag_core.retrieval.services.hybrid_search import HybridSearchConfig, HybridSearchService
from talkdocs_rag_core.retrieval.services.mistral_client import create_mistral_client
from talkdocs_rag_core.retrieval.vector_stores.chroma import ChromaVectorStore
from talkdocs_rag_core.retrieval.vector_stores.whoosh_index import WhooshKeywordIndex


def build_client(cfg: RagConfig) -> OpenAI:
    client = create_mistral_client(
        cfg.mistral_api_key,
        cfg.mistral_base_url,
        timeout_s=cfg.http_timeout_s,
        connect_timeout_s=cfg.http_connect_timeout_s,
        max_retries=cfg.http_max_retries,
    )
    if client is None:
        raise RuntimeError("MISTRAL_API_KEY assente: impossibile creare il client Mistral.")
    return client


def build_embedding_service(cfg: RagConfig, client: OpenAI | None = None) -> EmbeddingService:
    client = client or build_client(cfg)
    return EmbeddingService(client=client, model=cfg.mistral_embed_model, dimension=cfg.embed_dim)


def build_chroma_client(cfg: RagConfig) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=cfg.chroma_persist_dir, settings=ChromaSettings(anonymized_telemetry=False)
    )


def build_retrieval_store(
    cfg: RagConfig,
    embedding_service: EmbeddingService,
    chroma_client: chromadb.ClientAPI | None = None,
) -> ChromaVectorStore:
    chroma_client = chroma_client or build_chroma_client(cfg)
    return ChromaVectorStore(
        collection_name=cfg.chroma_collection_retrieval,
        embedding_service=embedding_service,
        client=chroma_client,
    )


def build_cache_store(
    cfg: RagConfig,
    embedding_service: EmbeddingService,
    chroma_client: chromadb.ClientAPI | None = None,
) -> ChromaVectorStore:
    chroma_client = chroma_client or build_chroma_client(cfg)
    return ChromaVectorStore(
        collection_name=cfg.chroma_collection_cache,
        embedding_service=embedding_service,
        client=chroma_client,
    )


def build_term_stats(cfg: RagConfig, chroma_client: chromadb.ClientAPI | None = None) -> TermStats | None:
    """Statistiche IDF per il guardiano di astensione.

    Si legge il file persistito dall'ingest. Se manca (corpus indicizzato prima del
    guardiano) si ricalcola dai chunk e si salva: una passata su tutto il corpus è
    accettabile una volta, non a ogni ``ask``.
    """
    if cfg.abstention_idf_threshold <= 0:
        return None  # guardiano spento: non pagare il costo
    stats = TermStats.load(cfg.term_df_path)
    if stats is not None:
        return stats
    chroma_client = chroma_client or build_chroma_client(cfg)
    col = chroma_client.get_or_create_collection(cfg.chroma_collection_retrieval)
    docs = col.get(include=["documents"])["documents"] or []
    if not docs:
        return None
    stats = TermStats.from_documents([d for d in docs if d])
    stats.save(cfg.term_df_path)
    return stats


async def build_whoosh(cfg: RagConfig) -> WhooshKeywordIndex:
    idx = WhooshKeywordIndex(index_dir=cfg.whoosh_index_dir)
    await idx.initialize()
    return idx


def build_hybrid(cfg: RagConfig, retrieval_store: ChromaVectorStore, whoosh: WhooshKeywordIndex) -> HybridSearchService:
    hcfg = HybridSearchConfig(
        vector_weight=cfg.hybrid_vector_weight,
        keyword_weight=cfg.hybrid_keyword_weight,
        rrf_constant=cfg.rrf_constant,
        merge_strategy="rrf",
    )
    return HybridSearchService(vector_store=retrieval_store, keyword_index=whoosh, config=hcfg)
