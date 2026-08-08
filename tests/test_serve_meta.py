"""La risposta meta: scheda scritta a mano + perimetro calcolato, mai il modello.

Il punto difendibile è la separazione delle fonti: il contesto viene dalla scheda, i
numeri vengono da una query sullo store — e la query sta nell'esito, come citazione.
"""

from config import RagConfig
from rag import router
from rag.corpus_card import CorpusCard
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


def test_scheda_e_statistiche_insieme():
    res = serve_meta(RagConfig(), StructuredStore.from_manifest(MANIFEST), CARD, "di cosa parla?", _rotta())
    assert res.route == "meta" and res.refused is False
    assert "atti pubblici" in res.answer_text                      # la scheda
    assert "3 delibere" in res.answer_text                          # il totale calcolato
    assert "CIPE: 1 delibere, anni 2019–2019" in res.answer_text or "CIPE: 1" in res.answer_text
    # La citazione è la query: sta nell'esito, rigiocabile dall'audit.
    assert res.structured is not None
    assert res.structured.intent == "corpus_stats"
    assert res.structured.computed_value == 3
    assert "GROUP BY comitato" in res.structured.sql
    # Zero chiamate: usage vuoto per costruzione, come tutto il file.
    assert res.usage == {}


def test_senza_scheda_restano_i_numeri_e_la_mancanza_e_dichiarata():
    res = serve_meta(RagConfig(), StructuredStore.from_manifest(MANIFEST), None, "di cosa parla?", _rotta())
    assert "non è stata compilata una scheda" in res.answer_text
    assert res.structured is not None and res.structured.computed_value == 3


def test_senza_store_resta_la_scheda_senza_numeri_inventati():
    res = serve_meta(RagConfig(), None, CARD, "di cosa parla?", _rotta())
    assert "atti pubblici" in res.answer_text
    assert res.structured is None
