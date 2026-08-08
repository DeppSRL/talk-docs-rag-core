"""C2 wiring dell'ask end-to-end: retrieval → (cache semantica) → generazione grounded
→ audit. Toggle cache on/off (deliverable A/B). Il caching provider è sempre attivo lato
richiesta (``prompt_cache_key``); la cache semantica è governata da ``use_cache``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from cache.semantic import SemanticCache
from config import REPO_ROOT, RagConfig
from rag import router
from rag.agentic_router import AgenticRouter
from rag.corpus_card import CorpusCard
from rag.generation import MistralGenerator, RagResult
from structured.service import serve_meta, serve_structured, serve_uncovered
from structured.store import StructuredStore

logger = logging.getLogger(__name__)


def _finish(res: RagResult, t0: float, w0: float) -> RagResult:
    """Chiude i due orologi. Sono quattro i punti di uscita di `ask`: dimenticarne uno
    corromperebbe la latenza di una run in silenzio, ed è per non accorgersene che il
    doppio orologio esiste."""
    res.latency_s = time.perf_counter() - t0
    res.latency_wall_s = time.time() - w0
    return res


def load_corpus_version(corpus_dir: Path | None = None) -> str:
    corpus_dir = corpus_dir or (REPO_ROOT / "corpus")
    manifest = corpus_dir / "manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text())["corpus_version"]
    return "no-manifest"


@dataclass
class RagPipeline:
    """Contiene i componenti costruiti una volta e serve le domande."""

    cfg: RagConfig
    hybrid: object
    generator: MistralGenerator
    semantic_cache: SemanticCache
    corpus_version: str
    # Tabella dei metadati per il ramo aggregativo. `None` = nessun manifest: il router
    # non instrada e la pipeline si comporta come prima.
    store: StructuredStore | None = None
    # Scheda del corpus (incremento 1b): contesto semantico scritto a mano, sorgente
    # della risposta meta e input del router agentico. `None` = nessuna scheda.
    card: CorpusCard | None = None
    # Router agentico. `None` = spento (`router_llm_enabled = False`, il default): la
    # cascata resta quella lessicale dell'incremento 1, byte-identica.
    agentic: AgenticRouter | None = None

    @staticmethod
    def _tag(res: RagResult, rotta: router.Route) -> RagResult:
        """Porta sorgente e traccia del routing su OGNI uscita. La traccia LLM esiste
        anche quando il fallback l'ha annullata (o quando la route servita è un ramo
        senza modello): perderla su un'uscita renderebbe il disaccordo lessicale/LLM
        non misurabile proprio sui casi interessanti."""
        res.router_source = rotta.source
        res.router_llm = rotta.llm
        return res

    async def ask(self, query: str, use_cache: bool, provider_cache_key: str | None = None) -> RagResult:
        """Serve una domanda.

        ``use_cache`` governa la cache semantica (C4b). ``provider_cache_key`` governa il
        prompt caching provider (C4a): se ``None`` usa il ``corpus_version`` (chiave stabile
        → cache provider attiva); l'eval passa una chiave unica per-richiesta nel ramo OFF
        così da sopprimere anche il provider cache e ottenere un baseline pulito.
        """
        # Due orologi: ``perf_counter`` è monotonico e non avanza mentre la macchina è in
        # suspend — è la latenza vera del servizio. ``time.time()`` è wall-clock e include
        # il sonno. Registrarli entrambi rende una run contaminata *visibile* nell'audit
        # invece di produrre una media silenziosamente sbagliata.
        t0 = time.perf_counter()
        w0 = time.time()

        # --- Router (incremento 1) ---
        # PRIMA della cache semantica, non dopo: la cache scatta a similarità ≥ 0,92 e
        # «quante delibere nel 2024» contro «…nel 2023» ci sta sopra. Cachare questo ramo
        # significherebbe servire il conteggio dell'anno sbagliato — esatto e credibile.
        # Invariante: `signals` vuoto in audit = router spento — `classify()` popola sempre tutte le chiavi.
        rotta = router.Route(router.POINTWISE)
        if self.cfg.router_enabled:
            rotta = router.classify(query)
            # Router agentico (incremento 1b), a valle del lessicale. La regola «delibera
            # specifica → puntuale» è una guardia dura decisa QUI, prima del modello: è
            # deterministica, gratuita, e il classificatore non deve poterla scavalcare.
            if self.agentic is not None and not rotta.signals.get("delibera_specifica"):
                rotta = self.agentic.classify(query, rotta)
            if rotta.route == router.UNCOVERED:
                return _finish(self._tag(serve_uncovered(self.cfg, query, rotta), rotta), t0, w0)
            if rotta.route == router.META:
                if self.card is not None or self.store is not None:
                    return _finish(self._tag(serve_meta(self.cfg, self.store, self.card, query, rotta), rotta), t0, w0)
                # Come per il ramo aggregativo senza manifest: si degrada contando, non in silenzio.
                logger.warning(
                    "ramo meta disattivato (nessuna scheda né manifest): meta-domanda "
                    "servita dal ramo puntuale — query: %s",
                    query[:120],
                )
            if rotta.route == router.STRUCTURED:
                if self.store is not None:
                    return _finish(self._tag(serve_structured(self.cfg, self.store, query, rotta), rotta), t0, w0)
                # Il fall-through va contato, non dedotto a posteriori dai `router_signals`:
                # la domanda è aggregativa e finisce nel ramo puntuale solo perché manca la
                # tabella. Una riga per richiesta, non una per sessione.
                logger.warning(
                    "ramo aggregativo disattivato (manifest assente): domanda aggregativa "
                    "servita dal ramo puntuale — query: %s",
                    query[:120],
                )

        # --- Cache semantica (C4b) ---
        if use_cache:
            hit = await self.semantic_cache.lookup(query, self.corpus_version)
            if hit is not None:
                res = RagResult(
                    query=query,
                    answer_text=hit.answer_text,
                    refused=False,
                    refusal_reason=None,
                    support_score=float("nan"),
                    cited_passages=[],
                    cited_chunk_ids=hit.cited_chunk_ids,
                    invalid_citations=[],
                    claims=hit.claims,
                    passages=[],
                    usage={},
                    raw_output="",
                    model=self.cfg.mistral_model,
                    params=self.generator._params(),
                    router_signals=rotta.signals,
                    from_cache=True,
                    cache_kind="semantic",
                    extra={"cache_similarity": hit.similarity, "matched_query": hit.matched_query},
                )
                return _finish(self._tag(res, rotta), t0, w0)

        # --- Retrieval + generazione ---
        results = await self.hybrid.search(query, top_k=self.cfg.rag_top_k)
        cache_key = provider_cache_key if provider_cache_key is not None else self.corpus_version
        res = self.generator.generate(query, results, cache_key=cache_key)
        res.router_signals = rotta.signals
        self._tag(res, rotta)
        # Gli orologi si fermano **prima** dello store in cache: scriverci dentro costa un
        # embedding e una write su Chroma, e finirebbe nella latenza servita — falsando
        # proprio il confronto A/B cache on/off che è il deliverable.
        _finish(res, t0, w0)

        # --- Store in cache semantica (solo risposte servite e non rifiutate) ---
        if use_cache and not res.refused:
            await self.semantic_cache.store(
                query=query,
                corpus_version=self.corpus_version,
                answer_text=res.answer_text,
                cited_chunk_ids=res.cited_chunk_ids,
                claims=res.claims,
            )
        return res


async def build_pipeline(cfg: RagConfig) -> RagPipeline:
    from app.wiring import (
        build_chroma_client,
        build_client,
        build_embedding_service,
        build_hybrid,
        build_retrieval_store,
        build_term_stats,
        build_whoosh,
    )

    client = build_client(cfg)
    embedding_service = build_embedding_service(cfg, client)
    chroma_client = build_chroma_client(cfg)
    retrieval_store = build_retrieval_store(cfg, embedding_service, chroma_client)
    whoosh = await build_whoosh(cfg)
    hybrid = build_hybrid(cfg, retrieval_store, whoosh)
    generator = MistralGenerator(cfg, client, term_stats=build_term_stats(cfg, chroma_client))
    semantic_cache = SemanticCache(cfg, embedding_service, chroma_client)

    store = StructuredStore.from_path(REPO_ROOT / "corpus" / "manifest.json")
    if store is None:
        logger.warning("manifest.json assente: ramo aggregativo disattivato per questa sessione")

    card_dir = Path(cfg.corpus_card_dir)
    if not card_dir.is_absolute():
        card_dir = REPO_ROOT / card_dir
    card = CorpusCard.load(card_dir)
    if card is None:
        logger.warning("scheda del corpus assente (%s): risposta meta ridotta alle sole statistiche", card_dir)

    corpus_version = load_corpus_version()
    agentic = None
    if cfg.router_llm_enabled:
        agentic = AgenticRouter(cfg, client, card, corpus_version=corpus_version)

    return RagPipeline(
        cfg=cfg,
        hybrid=hybrid,
        generator=generator,
        semantic_cache=semantic_cache,
        corpus_version=corpus_version,
        store=store,
        card=card,
        agentic=agentic,
    )
