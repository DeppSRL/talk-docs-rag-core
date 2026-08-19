"""Il contratto di `talk-docs-rag-core`: nucleo importabile senza strato applicativo.

Nato come smoke test del vendoring (exit criterion M2), dal ritaglio del 13 agosto 2026 è
diventato **la promessa che il pacchetto fa ai suoi consumatori** — e vale più di prima,
perché ora c'è un consumatore esterno (talk-docs) che quella promessa la usa per decidere di
dipendere da qui.

Le due proprietà difese:

- **importare il core non trascina FastAPI** né uno strato applicativo. Se un giorno qualcuno
  aggiunge una comodità che importa il web framework, talk-docs — che ha già il *suo* — si
  ritrova due mondi nello stesso processo, e il conflitto si scopre in produzione;
- **niente variabili d'ambiente lette all'import.** La config è iniettata: è ciò che rende
  possibile un provider e una soglia **per indice** invece che per processo, che è il
  requisito da cui dipende la demo multi-indice.
"""

import sys

import pytest


def test_import_non_trascina_fastapi():
    """Importare il nucleo non deve importare FastAPI né lo strato app di talk-docs."""
    # import mirati
    from talk_docs_rag_core.retrieval.services.embeddings import EmbeddingService  # noqa: F401
    from talk_docs_rag_core.retrieval.services.hybrid_search import HybridSearchService  # noqa: F401
    from talk_docs_rag_core.retrieval.services.llm_generation import LLMGenerationService  # noqa: F401
    from talk_docs_rag_core.retrieval.services.mistral_client import create_mistral_client  # noqa: F401
    from talk_docs_rag_core.retrieval.vector_stores.chroma import ChromaVectorStore  # noqa: F401

    assert "fastapi" not in sys.modules, "il vendoring non deve trascinare FastAPI"
    # Il namespace applicativo 'app' di talk-docs non deve essere presente.
    assert "app" not in sys.modules or getattr(sys.modules.get("app"), "__file__", "").find("talk-docs") == -1


def test_import_non_legge_env(monkeypatch):
    """Nessun modulo vendored deve leggere env all'import (config iniettata)."""
    # Rimuovo ogni chiave: se un modulo leggesse settings all'import, fallirebbe qui.
    for k in ("MISTRAL_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)

    import importlib

    for mod in (
        "talk_docs_rag_core.retrieval.services.embeddings",
        "talk_docs_rag_core.retrieval.services.mistral_client",
        "talk_docs_rag_core.retrieval.services.llm_generation",
        "talk_docs_rag_core.retrieval.vector_stores.chroma",
        "talk_docs_rag_core.retrieval.services.hybrid_search",
    ):
        importlib.reload(importlib.import_module(mod))  # non deve sollevare


def test_client_none_senza_chiave():
    """Senza API key la factory ritorna None (nessuna eccezione all'import)."""
    from talk_docs_rag_core.retrieval.services.mistral_client import create_mistral_client

    assert create_mistral_client("") is None


def test_embedding_service_richiede_client():
    """EmbeddingService pretende un client iniettato (config non importata)."""
    from talk_docs_rag_core.retrieval.services.embeddings import EmbeddingService

    with pytest.raises(ValueError):
        EmbeddingService(client=None, model="mistral-embed-2312", dimension=1024)


def test_hybrid_search_config_free():
    """HybridSearchService si costruisce con una config esplicita, senza settings."""
    from talk_docs_rag_core.retrieval.services.hybrid_search import HybridSearchConfig, HybridSearchService

    cfg = HybridSearchConfig(vector_weight=0.7, keyword_weight=0.3, rrf_constant=60)
    svc = HybridSearchService(vector_store=object(), keyword_index=object(), config=cfg)
    assert abs(svc.config.vector_weight + svc.config.keyword_weight - 1.0) < 1e-9


def test_models_importabili():
    from talk_docs_rag_core.retrieval.models.document import Document, DocumentChunk, SearchResult  # noqa: F401
