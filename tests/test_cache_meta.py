"""Cache persistente delle risposte meta: congelare la prosa, non i fatti.

Il patto che questi test difendono è uno solo, e ha due facce: la risposta deve restare
**identica** finché nulla di rilevante cambia (altrimenti la cache non serve a niente, ed
esiste per far sopravvivere i giudizi umani), e deve **cadere subito** quando cambia il
corpus, la scheda o il modello (altrimenti si serve la descrizione di un archivio che non
c'è più — il guasto peggiore proprio nel componente nato per dichiarare i limiti).
"""

from talk_docs_rag_core.cache.meta import CacheMeta, VoceMeta, impronta_scheda
from talk_docs_rag_core.config import RagConfig
from talk_docs_rag_core.rag import router
from talk_docs_rag_core.rag.corpus_card import CorpusCard
from talk_docs_rag_core.structured.service import serve_meta
from talk_docs_rag_core.structured.store import StructuredStore

CARD = CorpusCard(sections=(("00-contesto", "Le delibere CIPE/CIPESS sono atti pubblici."),))
ALTRA_CARD = CorpusCard(sections=(("00-contesto", "Le delibere CIPE/CIPESS sono atti amministrativi."),))
def _doc(path, title):
    return {"path": path, "title": title, "content_hash": "x", "n_chunks": 1}


MANIFEST = {
    "files": [
        _doc("delibere/2019/E190001.txt", "Delibera CIPE n. 1/2019"),
        _doc("delibere/2021/E210075.txt", "Delibera CIPESS n. 75/2021"),
    ]
}


def _cache(tmp_path, card=CARD, corpus="v1", model="m1"):
    return CacheMeta(tmp_path / "meta.json", corpus_version=corpus, card_hash=impronta_scheda(card), model=model)


def _voce(testo="Il corpus raccoglie delibere [1]."):
    return VoceMeta(
        answer_text=testo, claims=[{"statement": "x", "passages": [1], "verbatim": "atti pubblici"}],
        cited_passages=[1], cited_chunk_ids=["scheda::00-contesto"], raw_output="{}",
        query="di cosa parla?", model="m1",
    )


def test_la_stessa_domanda_ritrova_la_risposta(tmp_path):
    c = _cache(tmp_path)
    c.scrivi("Di cosa parla questo corpus?", _voce())
    # Spazi e maiuscole non fanno una domanda diversa; una riformulazione sì.
    assert c.leggi("  di cosa PARLA questo corpus?  ").answer_text.startswith("Il corpus")
    assert c.leggi("Che periodo copre l'archivio?") is None


def test_la_cache_sopravvive_alla_run(tmp_path):
    """È il punto: fra due run la voce deve essere ancora lì, o i giudizi umani si perdono."""
    _cache(tmp_path).scrivi("Di cosa parla?", _voce())
    assert _cache(tmp_path).leggi("Di cosa parla?") is not None


def test_cambiare_la_scheda_invalida_la_risposta(tmp_path):
    """`corpus_version` NON copre la scheda: vive dentro `corpus/` ma è esclusa
    dall'ingest. Senza la sua impronta nella chiave, correggere la scheda — che è il lavoro
    di setup di un corpus — lascerebbe in circolo risposte che descrivono quella di ieri."""
    _cache(tmp_path).scrivi("Di cosa parla?", _voce())
    assert _cache(tmp_path, card=ALTRA_CARD).leggi("Di cosa parla?") is None


def test_cambiare_corpus_o_modello_invalida_la_risposta(tmp_path):
    _cache(tmp_path).scrivi("Di cosa parla?", _voce())
    assert _cache(tmp_path, corpus="v2").leggi("Di cosa parla?") is None
    assert _cache(tmp_path, model="m2").leggi("Di cosa parla?") is None


def test_le_voci_obsolete_si_ripuliscono(tmp_path):
    """Non serve alla correttezza — la chiave le rende irraggiungibili — ma questo file si
    legge a occhio: è la prosa che si serve agli utenti."""
    _cache(tmp_path).scrivi("Di cosa parla?", _voce())
    nuova = _cache(tmp_path, corpus="v2")
    assert nuova.pulisci_obsolete() == 1
    assert _cache(tmp_path).leggi("Di cosa parla?") is None  # tolta davvero dal file


def test_un_file_corrotto_non_impedisce_di_rispondere(tmp_path):
    (tmp_path / "meta.json").write_text("{non è json", encoding="utf-8")
    c = _cache(tmp_path)
    assert c.leggi("Di cosa parla?") is None
    c.scrivi("Di cosa parla?", _voce())  # e si riparte scrivendo
    assert _cache(tmp_path).leggi("Di cosa parla?") is not None


class _GeneratoreContato:
    def __init__(self):
        self.chiamate = 0

    def genera_da_passaggi(self, query, passages, cache_key, **kw):
        from talk_docs_rag_core.rag.generation import RagResult

        self.chiamate += 1
        return RagResult(
            query=query, answer_text=f"risposta numero {self.chiamate} [1]", refused=False,
            refusal_reason=None, support_score=1.0, cited_passages=[1],
            cited_chunk_ids=[passages[0].chunk_id], invalid_citations=[],
            claims=[{"statement": "x", "passages": [1], "verbatim": "atti pubblici"}],
            passages=passages, usage={"total_tokens": 10}, raw_output="{}", model="finto", params={},
        )


def test_alla_seconda_run_la_risposta_e_la_stessa_e_non_costa(tmp_path):
    """Il difetto che chiude: 5 risposte meta su 6 cambiavano fra due run consecutive, e
    ogni cambio costava una rilettura del giudizio umano."""
    cfg = RagConfig()
    store = StructuredStore.from_manifest(MANIFEST)
    gen = _GeneratoreContato()
    cache = _cache(tmp_path)
    rotta = router.Route(router.META, signals={"forma_meta": True})

    prima = serve_meta(cfg, store, CARD, "di cosa parla?", rotta, generator=gen, cache=cache)
    dopo = serve_meta(cfg, store, CARD, "di cosa parla?", rotta, generator=gen, cache=_cache(tmp_path))

    assert gen.chiamate == 1, "la seconda run ha richiamato il modello"
    assert dopo.answer_text == prima.answer_text
    assert dopo.from_cache and dopo.cache_kind == "meta"
    assert dopo.usage == {}  # nessuna chiamata: nessun token da contare


def test_dalla_cache_i_passaggi_sono_quelli_di_adesso(tmp_path):
    """I passaggi non si congelano con la risposta: si ricostruiscono. Sono la cosa che si
    mostra a chi giudica, e devono essere quelli veri di questa run — la guardia verbatim,
    che è deterministica e gratuita, si ricalcola su di essi."""
    cfg = RagConfig()
    store = StructuredStore.from_manifest(MANIFEST)
    gen = _GeneratoreContato()
    rotta = router.Route(router.META)
    serve_meta(cfg, store, CARD, "di cosa parla?", rotta, generator=gen, cache=_cache(tmp_path))
    res = serve_meta(cfg, store, CARD, "di cosa parla?", rotta, generator=gen, cache=_cache(tmp_path))

    assert [p.chunk_id for p in res.passages] == ["scheda::00-contesto", "scheda::perimetro"]
    assert res.structured is not None and res.structured.computed_value == 2
    assert res.verbatim is not None  # ricalcolata, non ripresa dal file


def test_senza_cache_il_comportamento_e_quello_di_prima(tmp_path):
    cfg = RagConfig()
    gen = _GeneratoreContato()
    rotta = router.Route(router.META)
    a = serve_meta(cfg, StructuredStore.from_manifest(MANIFEST), CARD, "q?", rotta, generator=gen)
    b = serve_meta(cfg, StructuredStore.from_manifest(MANIFEST), CARD, "q?", rotta, generator=gen)
    assert gen.chiamate == 2 and a.answer_text != b.answer_text
