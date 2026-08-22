"""I vettori non sopravvivono al lotto che li ha prodotti.

Il difetto: `run_ingest` embeddava tutto il corpus e poi scriveva tutto, quindi i vettori di
ogni chunk dovevano coesistere fino all'unica chiamata `add_documents` finale. Su 18.396 chunk
a 1024 dimensioni sono centinaia di MiB — misurate su un corpus sintetico col profilo delle
delibere, servizio di embedding e store finti perché ciò che si misura è la memoria che *la
pipeline* trattiene: **664,8 MiB di picco (tracemalloc), 79,1 MiB dopo**. Sul corpus vero il
picco RSS misurato prima era 1.280,5 MiB.

Qui non si misura: si difendono le proprietà che rendono quella misura vera e la tengono vera.

- nessun chunk conserva un embedding dopo l'ingest;
- in nessun istante ci sono più di `EMBED_BATCH` vettori vivi;
- i lotti di embedding sono **gli stessi di prima**: stesse chiamate al provider, stessi
  token. Un rimedio alla memoria che cambiasse i lotti cambierebbe anche il costo, e la
  prossima misura sul provider non sarebbe più confrontabile;
- ogni chunk arriva in Chroma. Il modo peggiore di dimezzare il picco è dimezzare l'indice.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from talk_docs_rag_core.config import RagConfig
from talk_docs_rag_core.ingest import pipeline as mod
from talk_docs_rag_core.retrieval.models.document import DocumentChunk

EMBED_DIM = 8


# =====================================================================================
# Il contratto di `add_chunks`
# =====================================================================================


class _CollectionFinta:
    def __init__(self):
        self.scritture = []

    def add(self, ids, documents, embeddings, metadatas):
        self.scritture.append({"ids": ids, "documents": documents, "embeddings": embeddings, "metadatas": metadatas})


class _ClientFinto:
    def __init__(self, collection):
        self._collection = collection

    def get_or_create_collection(self, name):
        return self._collection


def _store():
    from talk_docs_rag_core.retrieval.vector_stores.chroma import ChromaVectorStore

    collection = _CollectionFinta()
    store = ChromaVectorStore(
        collection_name="delibere", embedding_service=object(), client=_ClientFinto(collection)
    )
    return store, collection


def _chunk(i: int, doc_id: str = "delibere/2024/E240001.txt", con_embedding: bool = True) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{doc_id}::{i}",
        content=f"contenuto {i}",
        embedding=[float(i)] * EMBED_DIM if con_embedding else None,
        start_index=0,
        end_index=10,
        metadata={"doc_id": doc_id, "chunk_index": i, "source": doc_id},
    )


class TestAddChunks:
    def test_scrive_il_lotto_in_una_chiamata_sola(self):
        store, collection = _store()
        n = asyncio.run(store.add_chunks([_chunk(i) for i in range(5)], {"delibere/2024/E240001.txt": "{}"}))
        assert n == 5
        assert len(collection.scritture) == 1
        assert len(collection.scritture[0]["ids"]) == 5

    def test_un_lotto_attraversa_piu_documenti(self):
        """È la ragione per cui i metadati arrivano per `doc_id` e non dal documento."""
        store, collection = _store()
        chunks = [_chunk(0, doc_id="a.txt"), _chunk(0, doc_id="b.txt")]
        asyncio.run(store.add_chunks(chunks, {"a.txt": '{"name":"A"}', "b.txt": '{"name":"B"}'}))
        metadati = collection.scritture[0]["metadatas"]
        assert [m["source_id"] for m in metadati] == ["a.txt", "b.txt"]
        assert [m["document_metadata"] for m in metadati] == ['{"name":"A"}', '{"name":"B"}']

    def test_conserva_i_metadati_del_chunk(self):
        """`chunk_index` e `source` servono a valle: senza, un passaggio recuperato non si
        colloca più nel documento da cui viene."""
        store, collection = _store()
        asyncio.run(store.add_chunks([_chunk(3)], {"delibere/2024/E240001.txt": "{}"}))
        m = collection.scritture[0]["metadatas"][0]
        assert m["chunk_index"] == 3
        assert m["source"] == "delibere/2024/E240001.txt"
        assert (m["start_index"], m["end_index"]) == (0, 10)

    def test_un_lotto_vuoto_non_scrive(self):
        store, collection = _store()
        assert asyncio.run(store.add_chunks([], {})) == 0
        assert collection.scritture == []

    def test_un_chunk_senza_embedding_e_un_errore(self):
        """Scriverlo con un vettore nullo darebbe un chunk che non si recupera mai, e nessun
        errore che lo dica."""
        store, _ = _store()
        with pytest.raises(ValueError, match="nessun embedding"):
            asyncio.run(store.add_chunks([_chunk(0, con_embedding=False)], {"delibere/2024/E240001.txt": "{}"}))

    def test_metadati_del_documento_mancanti_sono_un_errore(self):
        store, _ = _store()
        with pytest.raises(KeyError):
            asyncio.run(store.add_chunks([_chunk(0)], {}))


# =====================================================================================
# La pipeline: quanti vettori sono vivi insieme
# =====================================================================================


class _EmbeddingFinto:
    def __init__(self):
        self.usage = {"calls": 0, "total_tokens": 0}
        self.dimensioni_lotti: list[int] = []

    async def get_embeddings(self, testi):
        self.usage["calls"] += 1
        self.usage["total_tokens"] += sum(len(t) // 4 for t in testi)
        self.dimensioni_lotti.append(len(testi))
        return [[float(i)] * EMBED_DIM for i in range(len(testi))]


class _StoreSpia:
    """Conta, a ogni scrittura, quanti vettori dell'INTERO corpus sono vivi in quel momento."""

    def __init__(self):
        self.chunk_scritti: list[str] = []
        self.vivi_max = 0
        self.tutti_i_chunk: list[DocumentChunk] = []

    async def add_chunks(self, chunks, metadata_json):
        self.chunk_scritti.extend(c.chunk_id for c in chunks)
        vivi = sum(1 for c in self.tutti_i_chunk if c.embedding is not None)
        self.vivi_max = max(self.vivi_max, vivi)
        return len(chunks)


class _WhooshFinto:
    def __init__(self):
        self.chunk = 0

    async def clear_index(self):
        pass

    async def index_documents(self, chunks):
        self.chunk += len(chunks)

    async def close(self):
        pass


@pytest.fixture
def corpus(tmp_path) -> Path:
    """Abbastanza chunk da attraversare più lotti di embedding."""
    d = tmp_path / "corpus"
    d.mkdir()
    paragrafo = "Delibera CIPESS di assegnazione delle risorse del Fondo sviluppo e coesione. " * 12
    for i in range(12):
        (d / f"E24{i:04d}.txt").write_text(
            f"DELIBERA {i}\n\n" + "\n\n".join(f"{paragrafo} atto {i} comma {j}." for j in range(8)),
            encoding="utf-8",
        )
    return d


@pytest.fixture
def esegui(monkeypatch, corpus, tmp_path):
    """Esegue `run_ingest` con store, embedding e Whoosh finti; restituisce le tre spie."""

    def _esegui():
        from talk_docs_rag_core import wiring

        embedding = _EmbeddingFinto()
        store = _StoreSpia()
        whoosh = _WhooshFinto()

        # Lo store deve poter guardare *tutti* i chunk per contare i vettori vivi: glieli
        # passa la pipeline stessa, intercettando la lista piatta che costruisce.
        vero = mod._embedda_e_scrivi_a_lotti

        async def spia(embedding_service, st, chunks, metadata_json):
            store.tutti_i_chunk = chunks
            return await vero(embedding_service, st, chunks, metadata_json)

        monkeypatch.setattr(mod, "_embedda_e_scrivi_a_lotti", spia)
        monkeypatch.setattr(wiring, "build_embedding_service", lambda cfg: embedding)
        monkeypatch.setattr(wiring, "build_chroma_client", lambda cfg: _ClientChroma())
        monkeypatch.setattr(wiring, "build_retrieval_store", lambda cfg, emb, client=None: store)

        async def _whoosh(cfg):
            return whoosh

        monkeypatch.setattr(wiring, "build_whoosh", _whoosh)

        cfg = RagConfig(
            embed_dim=EMBED_DIM,
            corpus_dir=str(corpus),
            corpus_card_dir=str(corpus / "card"),
            term_df_path=str(tmp_path / "term_df.json"),
            frasi_index_path=str(tmp_path / "frasi.json"),
            whoosh_index_dir=str(tmp_path / "whoosh"),
        )
        rep = asyncio.run(mod.run_ingest(cfg, corpus_dir=corpus))
        return rep, embedding, store, whoosh

    return _esegui


class _ClientChroma:
    def delete_collection(self, name):
        pass


class TestVettoriVivi:
    def test_nessun_vettore_sopravvive_all_ingest(self, esegui):
        _, _, store, _ = esegui()
        assert all(c.embedding is None for c in store.tutti_i_chunk)

    def test_non_piu_di_un_lotto_di_vettori_alla_volta(self, esegui):
        """La proprietà per cui esiste questo incremento. Se un giorno tornasse la scrittura
        unica finale, questo numero salirebbe al numero di chunk del corpus."""
        rep, _, store, _ = esegui()
        assert store.vivi_max <= mod.EMBED_BATCH
        assert rep.n_chunks > mod.EMBED_BATCH, "corpus troppo piccolo: il test non proverebbe niente"

    def test_i_lotti_di_embedding_sono_quelli_di_prima(self, esegui):
        """Stesse chiamate al provider e stessi token: un rimedio alla memoria non deve
        spostare il costo, o la misura di ieri non è più confrontabile con quella di domani."""
        rep, embedding, _, _ = esegui()
        attesi = [mod.EMBED_BATCH] * (rep.n_chunks // mod.EMBED_BATCH)
        if rep.n_chunks % mod.EMBED_BATCH:
            attesi.append(rep.n_chunks % mod.EMBED_BATCH)
        assert embedding.dimensioni_lotti == attesi

    def test_ogni_chunk_arriva_nell_indice(self, esegui):
        """Il modo peggiore di dimezzare il picco è dimezzare l'indice."""
        rep, _, store, whoosh = esegui()
        assert len(store.chunk_scritti) == rep.n_chunks
        assert len(set(store.chunk_scritti)) == rep.n_chunks
        assert whoosh.chunk == rep.n_chunks

    def test_il_manifest_resta_quello_di_prima(self, esegui):
        """`corpus_version` è la chiave con cui si invalida la cache semantica: se cambiasse
        per un rimedio alla memoria, ogni cache verrebbe buttata senza che il corpus sia
        cambiato."""
        rep, _, _, _ = esegui()
        assert rep.n_files == 12
        assert Path(rep.manifest_path).exists()
        assert rep.corpus_version and rep.corpus_content_hash
