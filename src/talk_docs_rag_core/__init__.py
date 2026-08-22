"""`talk-docs-rag-core` — il nucleo RAG grounded, senza strato applicativo.

Contiene ciò che rende una risposta *difendibile*, non ciò che la serve: retrieval ibrido,
rifiuto deterministico su soglia, astensione per segnale IDF, verifica delle citazioni a
valle, guardia verbatim, tupla di audit rigiocabile, e l'harness che misura tutto questo.
FastAPI, auth, multi-index e UI **non** stanno qui: sono del consumatore.

## Il patto della configurazione

`RagConfig` è una dataclass **frozen iniettata nei costruttori**, non un singleton letto
all'import. Non è una preferenza di stile: è ciò che rende chunk size, pesi dell'hybrid,
RRF-k, modello, soglie e temperatura **parametri che l'eval spazza per run** — e, per un
consumatore multi-indice come talk-docs, ciò che rende possibile un provider e una soglia
**per indice** invece che per processo.

## Il patto delle soglie — da leggere prima di riusare un numero

`support_threshold` è una **similarità coseno su un modello di embedding specifico**. Il
valore tarato sul corpus delle delibere CIPE/CIPESS (0,79 su `mistral-embed-2312`, 1024
dimensioni) **non significa nulla** su un altro corpus o con un altro modello di embedding.

Per questo il pacchetto spedisce la **procedura di taratura**
(`talk_docs_rag_core.eval.tara_soglia`) e non un default sensato: un secondo indice che
eredita la soglia del primo eredita un numero trovato altrove, e la sua disciplina del
rifiuto diventa una coincidenza.
"""

from __future__ import annotations

from talk_docs_rag_core.config import RagConfig
from talk_docs_rag_core.ingest.pipeline import IngestReport, StatoIngest, run_ingest
from talk_docs_rag_core.pipeline import RagPipeline, build_pipeline, load_corpus_version
from talk_docs_rag_core.rag.generation import RagResult

__all__ = [
    # configurazione — il punto d'ingresso di ogni consumatore
    "RagConfig",
    # interrogazione
    "RagPipeline",
    "build_pipeline",
    "RagResult",
    # indicizzazione
    "run_ingest",
    "IngestReport",
    # ciò che si sa di una run anche quando fallisce: lo store è stato toccato?
    "StatoIngest",
    # provenienza del corpus: l'hash con cui una misura si dichiara riproducibile
    "load_corpus_version",
]

__version__ = "0.1.0"
