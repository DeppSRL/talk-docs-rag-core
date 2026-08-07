"""Ordine della cascata: il router viene PRIMA della cache semantica.

Il rischio da coprire è specifico e grave: «quante delibere nel 2024» e «quante delibere nel
2023» stanno sopra la soglia di similarità della cache (0,92). Se il ramo aggregativo la
consultasse, servirebbe il conteggio dell'anno sbagliato — con un numero esatto, e quindi
credibilissimo.
"""

import asyncio
import logging

from app.pipeline import RagPipeline
from config import RagConfig
from structured.store import StructuredStore

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
        from rag.generation import RagResult

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


def test_senza_store_il_router_non_instrada(caplog):
    """Nessun manifest = nessuna tabella: si torna alla pipeline di sempre invece di esplodere."""
    p, cache, hybrid, gen = _pipeline()
    p.store = None
    with caplog.at_level(logging.WARNING, logger="app.pipeline"):
        res = asyncio.run(p.ask("Quante delibere ha adottato il CIPESS nel 2024?", use_cache=True))
    assert res.route == "pointwise"
    # Il gemello puntuale verifica che il ramo sia stato *eseguito*: qui vale lo stesso, o
    # «pointwise» direbbe solo che nessuno ha instradato, non che qualcuno ha risposto.
    assert hybrid.searches == 1 and gen.chiamate == 1
    # Il fall-through lascia una traccia per-richiesta: la spec vuole che si conti, non che
    # si deduca rifacendo a mano la cascata sui `router_signals`.
    tracce = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("ramo aggregativo disattivato" in m and "CIPESS nel 2024" in m for m in tracce)
