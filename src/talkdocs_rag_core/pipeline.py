"""C2 wiring dell'ask end-to-end: retrieval → (cache semantica) → generazione grounded
→ audit. Toggle cache on/off (deliverable A/B). Il caching provider è sempre attivo lato
richiesta (``prompt_cache_key``); la cache semantica è governata da ``use_cache``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from cache.semantic import SemanticCache
from config import REPO_ROOT, RagConfig
from rag.generation import MistralGenerator, RagResult


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
                    from_cache=True,
                    cache_kind="semantic",
                    latency_s=time.perf_counter() - t0,
                    latency_wall_s=time.time() - w0,
                    extra={"cache_similarity": hit.similarity, "matched_query": hit.matched_query},
                )
                return res

        # --- Retrieval + generazione ---
        results = await self.hybrid.search(query, top_k=self.cfg.rag_top_k)
        cache_key = provider_cache_key if provider_cache_key is not None else self.corpus_version
        res = self.generator.generate(query, results, cache_key=cache_key)
        res.latency_s = time.perf_counter() - t0
        res.latency_wall_s = time.time() - w0

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

    return RagPipeline(
        cfg=cfg,
        hybrid=hybrid,
        generator=generator,
        semantic_cache=semantic_cache,
        corpus_version=load_corpus_version(),
    )
