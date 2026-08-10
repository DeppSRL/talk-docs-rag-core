"""C5 — audit logging: una tupla rigiocabile per ogni risposta.

Contenuto (spec §5): query, chunk_id[]+hash[], versione modello, parametri, output
grezzo, usage, verdetto. In più: corpus_version, info cache, latenza. Scrittura JSONL in
``logs/<run_id>.jsonl`` (append), così una run genera tuple rigiocabili e difendibili.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from rag.generation import RagResult


@dataclass
class AuditRecord:
    timestamp_utc: str
    run_id: str
    query: str
    # La risposta servita all'utente. Non è ricavabile da `raw_output`: sugli hit di cache
    # semantica `raw_output` è vuoto (nessuna chiamata al modello) e il testo esiste solo qui.
    # Senza questo campo la tupla non permette di *difendere* una risposta: non la contiene.
    answer_text: str
    corpus_version: str
    model: str
    params: dict
    # grounding
    support_score: float
    refused: bool
    refusal_reason: str | None
    cited_chunk_ids: list[str]
    chunk_hashes: list[str]
    # TUTTI i chunk finiti nel contesto, in ordine di rank — non solo quelli citati.
    # Senza questo non si può distinguere «il passaggio giusto c'era e il modello ha
    # sbagliato» da «il passaggio giusto non era stato recuperato», che è la distinzione da
    # cui dipende dove intervenire (reranker vs modello vs verifica). Misurato su ic-07-bis:
    # il chunk con la cifra corretta non era nei primi 20, e il support_score era 0,878 —
    # la pipeline era al massimo della fiducia con le prove sbagliate in mano.
    retrieved_chunk_ids: list[str]
    # Testo dei passaggi che NON stanno nell'indice, per `chunk_id`. Sono le sezioni della
    # scheda del corpus e il blocco delle statistiche calcolate del ramo meta: esistono in
    # memoria al momento della risposta e da nessuna altra parte.
    #
    # Senza questo campo la tupla registra citazioni `scheda::…` che nessuno può più
    # risolvere — chi giudica vede un rimando vuoto e la risposta non è difendibile. È lo
    # stesso motivo per cui `answer_text` è nella tupla invece di essere dedotta da
    # `raw_output`: l'audit deve **contenere** ciò che serve a rigiocare, non rimandarci.
    passages_inline: dict[str, dict]
    invalid_citations: list[int]
    claims: list[dict]
    # caching
    # Terzo esito (C3b) + il segnale che l'ha deciso, registrato SEMPRE (anche quando non
    # scatta) così la soglia si può ritarare su run passate senza rigiocarle.
    uncertain: bool
    missing_terms: list[str]
    abstention_signal: float
    # Risposta tagliata dal tetto max_output_tokens: va segnata, non presentata come completa.
    truncated: bool
    finish_reason: str | None
    from_cache: bool
    cache_kind: str | None
    cache_enabled: bool
    # provider usage
    usage: dict
    # output grezzo (per rigiocare/difendere)
    raw_output: str
    latency_s: float | None = None  # monotonico: latenza vera del servizio
    latency_wall_s: float | None = None  # wall-clock: include eventuale suspend della macchina
    extra: dict = field(default_factory=dict)
    # --- Router e guardia verbatim (incremento 1) ---
    # Registrati SEMPRE, anche quando non fanno scattare nulla: le soglie si ritarano su
    # run passate senza rigiocarle, com'è già per `abstention_signal`.
    route: str = "pointwise"
    router_signals: dict = field(default_factory=dict)
    # Router agentico (incremento 1b): chi ha deciso la route e la traccia della
    # chiamata di classificazione (proposta grezza, usage, errore, latenza). La traccia
    # c'è anche quando il fallback l'ha annullata: una proposta scartata è un dato.
    router_source: str = "lexical"
    router_llm: dict | None = None
    # Sul ramo aggregativo la tupla deve permettere di RIESEGUIRE la query: sql + params +
    # corpus_version. È la citazione, non un dettaglio diagnostico.
    structured: dict | None = None
    verbatim: dict | None = None
    # Le formule ricorrenti dichiarate nella risposta, con i conteggi. La nota è calcolata:
    # la sua difendibilità sta qui, come per `structured` sul ramo aggregativo — sql e
    # parametri là, frase e numero di documenti qui.
    provenienza: dict | None = None
    uncertain_reason: str | None = None


def _chunk_hashes(result: RagResult) -> list[str]:
    """Hash-contenuto dei chunk citati, dai passaggi in contesto.

    Sugli HIT di cache semantica i passaggi non sono ricaricati (``passages`` vuoto):
    in quel caso la lista resta vuota — le citazioni autoritative sono i
    ``cited_chunk_ids``, e la tupla di generazione originale porta già gli hash reali.
    """
    import hashlib

    by_id = {p.chunk_id: p for p in result.passages}
    hashes = []
    for cid in result.cited_chunk_ids:
        p = by_id.get(cid)
        if p is None:
            continue
        hashes.append(hashlib.sha256(p.content.encode("utf-8")).hexdigest())
    return hashes


class AuditWriter:
    def __init__(self, run_id: str, audit_dir: str):
        self.run_id = run_id
        self.dir = Path(audit_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{run_id}.jsonl"

    def record(self, result: RagResult, corpus_version: str, cache_enabled: bool) -> AuditRecord:
        rec = AuditRecord(
            timestamp_utc=datetime.now(tz=UTC).isoformat(),
            run_id=self.run_id,
            query=result.query,
            answer_text=result.answer_text,
            corpus_version=corpus_version,
            model=result.model,
            params=result.params,
            support_score=result.support_score,
            refused=result.refused,
            refusal_reason=result.refusal_reason,
            cited_chunk_ids=result.cited_chunk_ids,
            chunk_hashes=_chunk_hashes(result),
            retrieved_chunk_ids=[p.chunk_id for p in result.passages],
            passages_inline={
                p.chunk_id: {"source": p.source, "text": p.content}
                for p in result.passages
                if not getattr(p, "in_index", True)
            },
            invalid_citations=result.invalid_citations,
            claims=result.claims,
            uncertain=result.uncertain,
            missing_terms=result.missing_terms,
            abstention_signal=result.abstention_signal,
            truncated=result.truncated,
            finish_reason=result.finish_reason,
            from_cache=result.from_cache,
            cache_kind=result.cache_kind,
            cache_enabled=cache_enabled,
            usage=result.usage,
            raw_output=result.raw_output,
            latency_s=result.latency_s,
            latency_wall_s=result.latency_wall_s,
            route=result.route,
            router_signals=result.router_signals,
            router_source=result.router_source,
            router_llm=result.router_llm,
            structured=asdict(result.structured) if result.structured else None,
            verbatim=asdict(result.verbatim) if result.verbatim else None,
            provenienza=result.provenienza,
            uncertain_reason=result.uncertain_reason,
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
        return rec
