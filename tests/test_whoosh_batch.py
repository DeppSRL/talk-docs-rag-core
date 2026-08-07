"""Indicizzazione keyword in batch (arm keyword dell'RRF).

Il vendored ``index_document`` apre un writer e **committa per ogni chunk**: forma
corretta per l'upload occasionale di un documento in talk-docs (app-shaped), rovinosa
per l'ingest batch del banco. Misurato sulle 511 delibere: 33,6 → 24,2 doc/s *in calo*
(ogni commit aggiunge un segmento, i merge si fanno via via più costosi), cioè 10-20
minuti per 13.670 chunk — da ripagare a ogni run, perché `poc-runner.yml` ricostruisce
l'indice su runner effimero prima di ogni retrieve/ask/eval.

``index_documents`` scrive tutto con **un solo writer e un solo commit**.
"""

import asyncio

import pytest

from vendor.talkdocs.vector_stores.whoosh_index import WhooshKeywordIndex


def _chunks(n: int, prefisso: str = "delibere/2024/E240001.txt"):
    return [
        {
            "chunk_id": f"{prefisso}::{i}",
            "content": f"Delibera CIPESS assegnazione di risorse numero {i} al Fondo sviluppo e coesione.",
            "source": prefisso,
            "metadata": {"doc_id": prefisso, "chunk_index": i},
        }
        for i in range(n)
    ]


@pytest.fixture
def indice(tmp_path):
    ix = WhooshKeywordIndex(index_dir=str(tmp_path / "whoosh"))
    asyncio.run(ix.initialize())
    yield ix
    asyncio.run(ix.close())


def test_batch_indicizza_tutti_i_chunk(indice):
    asyncio.run(indice.index_documents(_chunks(50)))
    assert indice.get_stats()["total_documents"] == 50


def test_batch_rende_i_chunk_cercabili(indice):
    asyncio.run(indice.index_documents(_chunks(20)))
    res = asyncio.run(indice.search("Fondo sviluppo coesione", top_k=5))
    assert len(res) == 5
    assert all(r.chunk_id.startswith("delibere/2024/E240001.txt::") for r in res)


def test_batch_preserva_metadati_e_source(indice):
    asyncio.run(indice.index_documents(_chunks(5)))
    res = asyncio.run(indice.search("assegnazione risorse", top_k=1))
    assert res[0].source == "delibere/2024/E240001.txt"
    assert res[0].metadata["doc_id"] == "delibere/2024/E240001.txt"


def test_batch_vuoto_non_rompe(indice):
    asyncio.run(indice.index_documents([]))
    assert indice.get_stats()["total_documents"] == 0


def test_batch_equivalente_al_ciclo_singolo(tmp_path):
    """Stesso stato finale del vecchio ciclo per-documento: è un'ottimizzazione, non un cambio di semantica."""
    items = _chunks(30)

    uno = WhooshKeywordIndex(index_dir=str(tmp_path / "uno"))
    asyncio.run(uno.initialize())
    for it in items:
        asyncio.run(uno.index_document(it["content"], it["chunk_id"], it["source"], it["metadata"]))
    singolo = asyncio.run(uno.search("Fondo sviluppo coesione", top_k=10))
    n_singolo = uno.get_stats()["total_documents"]
    asyncio.run(uno.close())

    batch_ix = WhooshKeywordIndex(index_dir=str(tmp_path / "batch"))
    asyncio.run(batch_ix.initialize())
    asyncio.run(batch_ix.index_documents(items))
    batch = asyncio.run(batch_ix.search("Fondo sviluppo coesione", top_k=10))
    n_batch = batch_ix.get_stats()["total_documents"]
    asyncio.run(batch_ix.close())

    assert n_batch == n_singolo == 30
    assert {r.chunk_id for r in batch} == {r.chunk_id for r in singolo}


def test_batch_richiede_initialize(tmp_path):
    ix = WhooshKeywordIndex(index_dir=str(tmp_path / "non-inizializzato"))
    with pytest.raises(RuntimeError):
        asyncio.run(ix.index_documents(_chunks(1)))
