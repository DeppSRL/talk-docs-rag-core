"""Ordine della cascata: il router viene PRIMA della cache semantica.

Il rischio da coprire è specifico e grave: «quante delibere nel 2024» e «quante delibere nel
2023» stanno sopra la soglia di similarità della cache (0,92). Se il ramo aggregativo la
consultasse, servirebbe il conteggio dell'anno sbagliato — con un numero esatto, e quindi
credibilissimo.
"""

import asyncio
import logging

from talk_docs_rag_core.config import RagConfig
from talk_docs_rag_core.pipeline import RagPipeline
from talk_docs_rag_core.structured.store import StructuredStore

MANIFEST = {
    "files": [
        {"path": "delibere/2024/E240001.txt", "title": "Delibera CIPESS n. 1/2024", "content_hash": "a", "n_chunks": 1},
    ]
}


class CacheSpia:
    """Registra se qualcuno la interroga. Non deve succedere sul ramo aggregativo."""

    def __init__(self):
        self.lookups = 0
        self.stores = 0

    async def lookup(self, query, corpus_version):
        self.lookups += 1
        return None

    async def store(self, **kwargs):
        self.stores += 1


class HybridSpia:
    def __init__(self):
        self.searches = 0

    async def search(self, query, top_k):
        self.searches += 1
        return []


class GeneratorFinto:
    def __init__(self):
        self.chiamate = 0

    def generate(self, query, results, cache_key):
        from talk_docs_rag_core.rag.generation import RagResult

        self.chiamate += 1
        return RagResult(
            query=query,
            answer_text="risposta puntuale",
            refused=False,
            refusal_reason=None,
            support_score=0.9,
            cited_passages=[],
            cited_chunk_ids=[],
            invalid_citations=[],
            claims=[],
            passages=[],
            usage={"prompt_tokens": 10},
            raw_output="{}",
            model="test",
            params={},
        )


def _pipeline(cfg=None):
    cache, hybrid, gen = CacheSpia(), HybridSpia(), GeneratorFinto()
    p = RagPipeline(
        cfg=cfg or RagConfig(),
        hybrid=hybrid,
        generator=gen,
        semantic_cache=cache,
        corpus_version="v-test",
        store=StructuredStore.from_manifest(MANIFEST),
    )
    return p, cache, hybrid, gen


def test_il_ramo_aggregativo_non_tocca_la_cache_ne_il_retrieval():
    p, cache, hybrid, gen = _pipeline()
    res = asyncio.run(p.ask("Quante delibere ha adottato il CIPESS nel 2024?", use_cache=True))
    assert res.route == "structured"
    assert res.structured.computed_value == 1
    assert cache.lookups == 0 and cache.stores == 0
    assert hybrid.searches == 0 and gen.chiamate == 0


def test_il_rifiuto_dichiarato_non_chiama_nulla():
    p, cache, hybrid, gen = _pipeline()
    res = asyncio.run(p.ask("Quanto è stato speso per le ferrovie?", use_cache=True))
    assert res.route == "uncovered" and res.refused is True
    assert cache.lookups == 0 and hybrid.searches == 0 and gen.chiamate == 0


def test_la_domanda_puntuale_segue_la_strada_di_sempre():
    p, cache, hybrid, gen = _pipeline()
    res = asyncio.run(p.ask("Che cosa prevede la delibera 75/2021?", use_cache=True))
    assert res.route == "pointwise"
    assert cache.lookups == 1 and hybrid.searches == 1 and gen.chiamate == 1
    assert res.router_signals["delibera_specifica"] is True


def test_router_spento_riporta_la_pipeline_a_prima():
    p, cache, hybrid, gen = _pipeline(RagConfig(router_enabled=False))
    res = asyncio.run(p.ask("Quante delibere ha adottato il CIPESS nel 2024?", use_cache=True))
    assert res.route == "pointwise"
    assert hybrid.searches == 1
    # Router spento: nessun segnale. È l'invariante che rende distinguibili in audit «spento»
    # e «acceso e non ha rilevato nulla» — `classify()` popola sempre tutte e cinque le chiavi.
    assert res.router_signals == {}


def test_il_ramo_meta_non_tocca_la_cache_ne_il_retrieval():
    """La risposta meta è deterministica e dipende dal corpus_version come quella
    strutturata: cacharla o farla passare dal retrieval sarebbe lo stesso guasto."""
    from talk_docs_rag_core.rag.corpus_card import CorpusCard

    p, cache, hybrid, gen = _pipeline()
    p.card = CorpusCard(sections=(("00-contesto", "Delibere CIPE/CIPESS, atti pubblici."),))
    res = asyncio.run(p.ask("Di cosa parla questo corpus?", use_cache=True))
    assert res.route == "meta" and res.refused is False
    assert "atti pubblici" in res.answer_text
    assert "1 delibere" in res.answer_text or "1 delibera" in res.answer_text
    assert cache.lookups == 0 and cache.stores == 0
    assert hybrid.searches == 0 and gen.chiamate == 0


def test_meta_senza_scheda_ne_store_degrada_a_puntuale(caplog):
    p, cache, hybrid, gen = _pipeline()
    p.card = None
    p.store = None
    with caplog.at_level(logging.WARNING, logger="talk_docs_rag_core.pipeline"):
        res = asyncio.run(p.ask("Di cosa parla questo corpus?", use_cache=True))
    assert res.route == "pointwise"
    assert hybrid.searches == 1 and gen.chiamate == 1
    assert any("ramo meta disattivato" in r.getMessage() for r in caplog.records)


class AgenticFinto:
    """Propone sempre la stessa route: basta a verificare il wiring della cascata."""

    def __init__(self, route, intent=None, params=None):
        self._route, self._intent, self._params = route, intent, params or {}
        self.consultato = 0

    def classify(self, query, lessicale):
        from talk_docs_rag_core.rag import router as r

        self.consultato += 1
        return r.Route(self._route, intent=self._intent, params=self._params,
                       signals=lessicale.signals, source="llm",
                       llm={"proposta": {"route": self._route}, "usage": {"prompt_tokens": 7}, "error": None})


def test_il_router_agentico_recupera_la_colloquiale():
    """`ag-05`: il lessicale non la riconosce, la proposta LLM validata la calcola."""
    p, cache, hybrid, gen = _pipeline()
    p.agentic = AgenticFinto("structured", intent="count_delibere", params={"anno": 2024, "comitato": "CIPESS"})
    res = asyncio.run(p.ask("Nel 2024 il Comitato quante ne ha approvate?", use_cache=True))
    assert res.route == "structured" and res.structured.computed_value == 1
    assert res.router_source == "llm" and res.router_llm["usage"] == {"prompt_tokens": 7}
    assert gen.chiamate == 0


def test_la_delibera_specifica_non_consulta_il_router_agentico():
    """Guardia dura: la regola deterministica vince, e la chiamata non si paga."""
    p, cache, hybrid, gen = _pipeline()
    p.agentic = AgenticFinto("uncovered")
    res = asyncio.run(p.ask("Che cosa prevede la delibera 75/2021?", use_cache=True))
    assert res.route == "pointwise" and p.agentic.consultato == 0
    assert res.router_source == "lexical" and res.router_llm is None


def test_router_agentico_acceso_per_default():
    """Acceso dal 2026-08-08, dopo la validazione su domande mai usate per correggere
    (routing 13/16 contro 10/16). Il test fissa la decisione: chi lo spegne lo fa
    esplicitamente, non per una svista in `RagConfig`."""
    assert RagConfig().router_llm_enabled is True


def test_senza_store_il_router_non_instrada(caplog):
    """Nessun manifest = nessuna tabella: si torna alla pipeline di sempre invece di esplodere."""
    p, cache, hybrid, gen = _pipeline()
    p.store = None
    with caplog.at_level(logging.WARNING, logger="talk_docs_rag_core.pipeline"):
        res = asyncio.run(p.ask("Quante delibere ha adottato il CIPESS nel 2024?", use_cache=True))
    assert res.route == "pointwise"
    # Il gemello puntuale verifica che il ramo sia stato *eseguito*: qui vale lo stesso, o
    # «pointwise» direbbe solo che nessuno ha instradato, non che qualcuno ha risposto.
    assert hybrid.searches == 1 and gen.chiamate == 1
    # Il fall-through lascia una traccia per-richiesta: la spec vuole che si conti, non che
    # si deduca rifacendo a mano la cascata sui `router_signals`.
    tracce = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("ramo aggregativo disattivato" in m and "CIPESS nel 2024" in m for m in tracce)
