"""La risposta meta: **generata** sulla scheda, con i numeri calcolati come passaggio.

Il punto difendibile è la separazione delle fonti: il contesto viene dalla scheda, i
numeri vengono da una query sullo store — e la query sta nell'esito, come citazione. Il
modello scrive la prosa a partire da entrambi, e non vede altro: se una cifra non sta nel
passaggio del perimetro, la guardia verbatim la segna non verificata come su ogni altra
risposta.

Senza generatore si degrada alla concatenazione delle sezioni: prolissa ma vera.
"""

from config import RagConfig
from rag import router
from rag.corpus_card import CorpusCard
from rag.generation import RagResult
from structured.service import serve_meta
from structured.store import StructuredStore


def _doc(path, title):
    return {"path": path, "title": title, "content_hash": "x", "n_chunks": 1}


MANIFEST = {
    "files": [
        _doc("delibere/2019/E190001.txt", "Delibera CIPE n. 1/2019"),
        _doc("delibere/2021/E210075.txt", "Delibera CIPESS n. 75/2021"),
        _doc("delibere/2024/E240047.txt", "Delibera CIPESS n. 47/2024"),
        # Un file senza codice delibera non entra nelle statistiche.
        _doc("campione/manuale.txt", "Documento campione"),
    ]
}

CARD = CorpusCard(sections=(("00-contesto", "Le delibere CIPE/CIPESS sono atti pubblici."),))


def _rotta():
    return router.Route(router.META, signals={"forma_meta": True})


class _GeneratoreFinto:
    """Registra i passaggi ricevuti e restituisce una risposta di senso compiuto."""

    def __init__(self, esito=None):
        self.passaggi = None
        self.cache_key = None
        self.esito = esito

    def genera_da_passaggi(self, query, passages, cache_key, **kw):
        self.passaggi = passages
        self.cache_key = cache_key
        if isinstance(self.esito, Exception):
            raise self.esito
        return RagResult(
            query=query,
            answer_text="Il corpus raccoglie delibere CIPE/CIPESS [1]; ne sono indicizzate 3 [2].",
            refused=False,
            refusal_reason=None,
            support_score=1.0,
            cited_passages=[1, 2],
            cited_chunk_ids=[p.chunk_id for p in passages[:2]],
            invalid_citations=[],
            claims=[],
            passages=passages,
            usage={"total_tokens": 42},
            raw_output="{}",
            model="finto",
            params={},
        )


def test_risposta_generata_sui_passaggi_della_scheda():
    gen = _GeneratoreFinto()
    res = serve_meta(
        RagConfig(), StructuredStore.from_manifest(MANIFEST), CARD, "di cosa parla?", _rotta(), generator=gen
    )
    assert res.route == "meta" and res.refused is False
    # La risposta è prosa, non la scheda riversata: la sezione è la *fonte*, non l'output.
    assert res.answer_text.startswith("Il corpus raccoglie")
    assert "Le delibere CIPE/CIPESS sono atti pubblici" not in res.answer_text

    # Il modello ha visto la scheda **e** il blocco calcolato, in quest'ordine.
    chunk_ids = [p.chunk_id for p in gen.passaggi]
    assert chunk_ids == ["scheda::00-contesto", "scheda::perimetro"]
    assert [p.n for p in gen.passaggi] == [1, 2]
    assert "atti pubblici" in gen.passaggi[0].content
    assert "3 delibere interrogabili" in gen.passaggi[1].content
    # Prefisso di cache proprio del ramo: le meta-domande non pescano la cache del puntuale.
    assert gen.cache_key.startswith("meta:")

    # La citazione dei numeri resta la query, rigiocabile dall'audit.
    assert res.structured is not None
    assert res.structured.intent == "corpus_stats"
    assert res.structured.computed_value == 3
    assert "GROUP BY comitato" in res.structured.sql
    assert res.params["route"] == "meta"
    assert res.router_signals == {"forma_meta": True}


def test_generazione_fallita_degrada_alla_scheda_invece_di_fallire():
    gen = _GeneratoreFinto(esito=RuntimeError("provider giù"))
    res = serve_meta(
        RagConfig(), StructuredStore.from_manifest(MANIFEST), CARD, "di cosa parla?", _rotta(), generator=gen
    )
    assert res.refused is False
    assert "atti pubblici" in res.answer_text          # la scheda
    assert "3 delibere" in res.answer_text             # il totale calcolato
    assert res.structured is not None and res.structured.computed_value == 3
    assert res.usage == {}


def test_senza_generatore_resta_la_concatenazione():
    res = serve_meta(RagConfig(), StructuredStore.from_manifest(MANIFEST), CARD, "di cosa parla?", _rotta())
    assert "atti pubblici" in res.answer_text
    assert "3 delibere" in res.answer_text
    assert "CIPE: 1 delibere, anni 2019–2019" in res.answer_text or "CIPE: 1" in res.answer_text
    assert res.usage == {}


def test_senza_scheda_restano_i_numeri_e_la_mancanza_e_dichiarata():
    res = serve_meta(RagConfig(), StructuredStore.from_manifest(MANIFEST), None, "di cosa parla?", _rotta())
    assert res.structured is not None and res.structured.computed_value == 3
    # C'è comunque un passaggio (il perimetro): la risposta si può generare.
    assert "3 delibere" in res.answer_text


def test_senza_store_ne_scheda_la_mancanza_e_dichiarata():
    res = serve_meta(RagConfig(), None, None, "di cosa parla?", _rotta())
    assert "non è stata compilata una scheda" in res.answer_text
    assert res.structured is None


def test_senza_store_resta_la_scheda_senza_numeri_inventati():
    res = serve_meta(RagConfig(), None, CARD, "di cosa parla?", _rotta())
    assert "atti pubblici" in res.answer_text
    assert res.structured is None
