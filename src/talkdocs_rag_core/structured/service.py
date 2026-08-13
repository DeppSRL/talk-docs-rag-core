"""Dal router al ``RagResult``: il ramo aggregativo, il rifiuto dichiarato, la meta-domanda.

Sui rami **calcolati** (`serve_structured`, `serve_uncovered`) non c'è nessuna chiamata al
modello e ``usage`` resta vuoto per costruzione: se comparissero token, il confronto A/B
del caching starebbe misurando una chiamata che non esiste.

`serve_meta` è l'eccezione dichiarata: il modello scrive la prosa, ma su passaggi che sono
la scheda del corpus e un blocco di statistiche calcolate — nessun numero nasce lì. Il suo
`usage` è reale e va contato, perché la chiamata c'è.
"""

from __future__ import annotations

import logging

from talkdocs_rag_core.config import RagConfig
from talkdocs_rag_core.rag import router
from talkdocs_rag_core.rag.corpus_card import CorpusCard
from talkdocs_rag_core.rag.generation import Passage, RagResult
from talkdocs_rag_core.rag.outcomes import StructuredOutcome
from talkdocs_rag_core.structured import intents
from talkdocs_rag_core.structured.answer import componi
from talkdocs_rag_core.structured.store import StructuredStore

logger = logging.getLogger(__name__)

# Statistiche del perimetro indicizzato per la risposta meta. La scheda del corpus non
# porta numeri per regola (divergerebbero dal corpus alla prima run di ingest): li porta
# questa query, che in audit è la citazione — come per ogni risposta calcolata.
_SQL_STATS = (
    "SELECT comitato, COUNT(*) AS n, MIN(anno) AS anno_min, MAX(anno) AS anno_max "
    "FROM documenti WHERE is_delibera GROUP BY comitato ORDER BY comitato"
)
_INTENT_CORPUS_STATS = "corpus_stats"

_MOTIVO_UNCOVERED = "aggregazione fuori copertura: nessuna tabella degli importi"

# Il perimetro dichiarato è lo stesso della risposta calcolata: **corpus indicizzato**, non
# «archivio». Non è una variante stilistica — «archivio» prometterebbe una copertura (l'intero
# archivio CIPE/CIPESS) che il corpus non ha, e la prometterebbe proprio nella frase che serve
# a dichiarare i limiti del sistema.
_TESTO_UNCOVERED = (
    "Questa è una domanda di aggregazione — un totale, un conteggio o un ammontare — e non "
    "posso calcolarla sul corpus indicizzato: dispongo dei metadati delle delibere (anno, "
    "numero, comitato), non di una tabella degli importi o delle grandezze citate nei testi. "
    "Preferisco dirlo invece di stimare un numero a partire da qualche passaggio. "
    "Posso invece contare o elencare le delibere per anno e per comitato."
)

# Il multi-anno non è l'assenza degli importi: i metadati bastano, è il semantic layer a essere
# chiuso. Dire all'utente «non ho la tabella degli importi» sarebbe falso, e offrirgli «posso
# contare le delibere per anno» sarebbe offrirgli ciò che ha appena chiesto. Le alternative qui
# sono le sole forme che il filtro anno sa esprimere. Sono dette come **forme**: gli anni citati
# nella domanda non arrivano fin qui (`rotta.params` è vuoto su questo ramo), e un esempio con
# anni concreti — «dal 2019 al 2021» a chi ha chiesto del 2015 e del 2023 — non spiega la sintassi,
# denuncia il template.
_TESTO_ANNI_MULTIPLI = (
    "La domanda cita più anni che non formano un solo intervallo, e in questa forma non posso "
    "contare le delibere: sul corpus indicizzato il filtro per anno esprime un anno singolo "
    "oppure un intervallo continuo, non un insieme di anni sciolti. Applicare solo una parte del "
    "filtro produrrebbe un numero calcolato — e proprio per questo credibile — in risposta a una "
    "domanda diversa da quella posta: preferisco dirlo. Posso rispondere su un anno per volta, "
    "sull'intervallo continuo che li comprende, indicandone il primo e l'ultimo anno nella forma "
    "«dal … al …», oppure sulla distribuzione anno per anno."
)

# Motivo → prosa. Una mappa e non una catena di `if`: il motivo è già il discriminante scelto dal
# router, e la prosa deve seguirlo per costruzione. `.get(motivo, _TESTO_UNCOVERED)` in lettura —
# un motivo nuovo, non ancora mappato, degrada al testo di default invece di sollevare `KeyError`.
# Che il default non abbia mai a servire lo garantisce `router.MOTIVI` con il test che ne verifica
# la copertura: il buco si rompe in fase di sviluppo, non in produzione.
_TESTI_RIFIUTO = {
    _MOTIVO_UNCOVERED: _TESTO_UNCOVERED,
    router.MOTIVO_ANNI_MULTIPLI: _TESTO_ANNI_MULTIPLI,
}


def _base(cfg: RagConfig, query: str, rotta: router.Route, route_servita: str) -> dict:
    return {
        "query": query,
        "support_score": 0.0,
        "cited_passages": [],
        "cited_chunk_ids": [],
        "invalid_citations": [],
        "claims": [],
        "passages": [],
        "usage": {},
        "raw_output": "",
        "model": cfg.mistral_model,
        # Le due rotte divergono nel degrado (`serve_structured` → `serve_uncovered`): registrarne
        # una sola vorrebbe dire perdere l'informazione «il router aveva proposto altro». Un
        # degrado è un dato, ma solo se si vede: `route` è ciò che ha risposto, `route_proposta`
        # ciò che il router aveva classificato.
        "params": {"model": cfg.mistral_model, "route": route_servita, "route_proposta": rotta.route},
        "router_signals": rotta.signals,
    }


def serve_uncovered(cfg: RagConfig, query: str, rotta: router.Route) -> RagResult:
    """Rifiuto **dichiarato**: dice perché non sa, non solo che non sa."""
    # Il rifiuto deve dire *quale* motivo. Il router ne porta uno proprio quando il caso non è
    # l'assenza degli importi (p.es. più anni citati senza intervallo): cablare qui «nessuna
    # tabella degli importi» su quel caso sarebbe falso. `None` = il default.
    motivo = rotta.reason or _MOTIVO_UNCOVERED
    if motivo not in _TESTI_RIFIUTO:
        # Qui si logga e non si solleva, all'opposto di `structured.answer` su un intento non
        # mappato: là il guasto di programmazione produrrebbe **prosa sbagliata su un calcolo
        # eseguito**, qui manca solo il testo di un rifiuto che nel merito resta corretto. Far
        # esplodere una risposta in produzione per un testo mancante sarebbe peggio del generico.
        # Il generico su un motivo diverso resta però una divergenza audit/prosa: che si veda.
        logger.warning("Motivo di rifiuto senza prosa dedicata (%s) — fallback al testo di default", motivo)
    route_servita = router.UNCOVERED
    return RagResult(
        # …e la prosa deve dire lo stesso motivo dell'audit. Un `refusal_reason` giusto sotto un
        # testo cablato sull'altro caso è lo stesso guasto che `reason` esiste per evitare,
        # spostato di un livello: l'utente legge la prosa, non il campo.
        answer_text=_TESTI_RIFIUTO.get(motivo, _TESTO_UNCOVERED),
        refused=True,
        refusal_reason=motivo,
        route=route_servita,
        **_base(cfg, query, rotta, route_servita),
    )


def _passaggi_meta(
    store: StructuredStore | None, card: CorpusCard | None
) -> tuple[list[Passage], StructuredOutcome | None]:
    """I «passaggi» del ramo meta: una sezione della scheda per passaggio, più il blocco
    delle statistiche calcolate. Stessa forma dei passaggi del retrieval, così la
    generazione, la verifica delle citazioni e la guardia verbatim funzionano identiche.

    Le statistiche entrano come **passaggio a sé**: se il modello scrive una cifra, quella
    cifra deve stare qui o nella scheda, o la guardia verbatim la segna non verificata.
    Il numero resta calcolato — il modello lo può copiare, non inventare.
    """
    passaggi: list[Passage] = []
    if card is not None:
        for n, (nome, testo) in enumerate(card.sections, start=1):
            passaggi.append(
                Passage(
                    n=n,
                    chunk_id=f"scheda::{nome}",
                    source=f"scheda del corpus — {nome}",
                    content=testo,
                    in_index=False,
                )
            )

    esito = None
    if store is not None:
        rows = store.query(_SQL_STATS, [])
        totale = sum(int(r["n"]) for r in rows)
        righe = [f"- {r['comitato']}: {r['n']} delibere, anni {r['anno_min']}–{r['anno_max']}" for r in rows]
        testo = (
            f"Perimetro indicizzato: {totale} delibere interrogabili, così distribuite:\n"
            + "\n".join(righe)
            + "\nQuesti numeri sono calcolati sui metadati del corpus indicizzato; "
            "l'archivio storico completo è più ampio."
        )
        passaggi.append(
            Passage(
                n=len(passaggi) + 1,
                chunk_id="scheda::perimetro",
                source="perimetro calcolato",
                content=testo,
                in_index=False,
            )
        )
        esito = StructuredOutcome(
            intent=_INTENT_CORPUS_STATS,
            sql=_SQL_STATS,
            params=[],
            rows=rows,
            n_rows=len(rows),
            computed_value=totale,
            completeness={},
            cited_doc_ids=[],
        )
    return passaggi, esito


def _da_cache(cfg, query, rotta, passaggi, esito, voce, base) -> RagResult:
    """Risposta meta servita dalla cache persistente.

    I passaggi sono **ricostruiti adesso** (scheda + statistiche appena calcolate), non
    ripresi dalla voce: se il corpus fosse cambiato la chiave non combacerebbe, ma i
    passaggi restano la cosa che si mostra a chi giudica, e devono essere quelli veri di
    questa run — non una copia congelata che nessuno ha più confrontato con l'indice.

    La guardia verbatim si **ricalcola** sugli stessi claim: è deterministica e non costa
    nulla, e un esito ripreso dal file sarebbe un numero di cui nessuno ha più verificato
    il presupposto.
    """
    from talkdocs_rag_core.rag.verbatim import verifica

    esito_verbatim = None
    if cfg.verbatim_enabled and voce.claims:
        esito_verbatim = verifica(voce.claims, passaggi, cfg.verbatim_min_chars)
    return RagResult(
        answer_text=voce.answer_text,
        refused=False,
        refusal_reason=None,
        route=router.META,
        structured=esito,
        verbatim=esito_verbatim,
        from_cache=True,
        # Non è la cache semantica e non va confusa con quella nel report: è persistente,
        # attiva in entrambi i bracci dell'A/B, e vale solo per il ramo meta.
        cache_kind="meta",
        **{
            **base,
            "claims": voce.claims,
            "cited_passages": voce.cited_passages,
            "cited_chunk_ids": voce.cited_chunk_ids,
            "passages": passaggi,
            "raw_output": voce.raw_output,
        },
    )


def serve_meta(
    cfg: RagConfig,
    store: StructuredStore | None,
    card: CorpusCard | None,
    query: str,
    rotta: router.Route,
    generator=None,
    cache=None,
) -> RagResult:
    """Meta-domanda sulla collezione: **risposta generata**, fondata sulla scheda.

    La prima versione concatenava tutte le sezioni della scheda. Era un difetto, rilevato
    giudicando: a «che periodo copre l'archivio?» rispondeva anche come sono fatte le
    premesse e cosa il sistema sa calcolare. Chi chiede vuole una risposta di senso
    compiuto; la sezione della scheda è la **fonte**, e come tale si cita — non si serve
    al posto della risposta.

    È l'unica eccezione alla regola «su questo ramo il modello non entra», e la ragione
    è che qui non c'è nessun numero da proteggere: le cifre stanno nel passaggio del
    perimetro, calcolate, e la guardia verbatim verifica che quelle scritte esistano. Il
    modello scrive la prosa, non i fatti.

    Senza generatore (o se la chiamata fallisce) si degrada alla concatenazione di prima:
    una risposta prolissa è meglio di nessuna risposta.

    ``cache`` (opzionale) congela la prosa: a parità di corpus, scheda e modello la
    risposta è la stessa a ogni run. Vedi ``cache/meta.py`` per il perché — in breve:
    questo ramo non dipende dal retrieval, quindi la variazione fra due run è rumore del
    decoder, e pagarla significa rileggere sei giudizi umani per sempre.
    """
    passaggi, esito = _passaggi_meta(store, card)
    route_servita = router.META
    base = _base(cfg, query, rotta, route_servita)

    if cache is not None and passaggi:
        voce = cache.leggi(query)
        if voce is not None:
            return _da_cache(cfg, query, rotta, passaggi, esito, voce, base)

    if generator is not None and passaggi:
        try:
            res = generator.genera_da_passaggi(query, passaggi, cache_key=f"meta:{cfg.mistral_model}")
            res.route = route_servita
            res.structured = esito
            res.params = base["params"]
            res.router_signals = rotta.signals
            if cache is not None:
                from talkdocs_rag_core.cache.meta import VoceMeta

                cache.scrivi(
                    query,
                    VoceMeta(
                        answer_text=res.answer_text,
                        claims=res.claims,
                        cited_passages=res.cited_passages,
                        cited_chunk_ids=res.cited_chunk_ids,
                        raw_output=res.raw_output,
                        query=query,
                        model=cfg.mistral_model,
                    ),
                )
            return res
        except Exception as exc:  # una meta-domanda non deve fallire per un errore di rete
            logger.warning("Generazione meta fallita (%s): degrado alla scheda integrale", exc.__class__.__name__)

    testo = "\n\n".join(p.content for p in passaggi) or (
        "Per questo corpus non è stata compilata una scheda descrittiva e non risulta un "
        "corpus indicizzato: non posso descriverlo."
    )
    return RagResult(
        answer_text=testo,
        refused=False,
        refusal_reason=None,
        route=route_servita,
        structured=esito,
        **base,
    )


def serve_structured(cfg: RagConfig, store: StructuredStore, query: str, rotta: router.Route) -> RagResult:
    costruita = intents.build(rotta.intent or "", rotta.params)
    if costruita is None:
        # Parametri non legabili o intento non coperto: meglio un rifiuto che una query su
        # un filtro indovinato.
        return serve_uncovered(cfg, query, rotta)

    sql, sql_params = costruita
    rows = store.query(sql, sql_params)
    testo, esito = componi(
        intent=rotta.intent,
        params=rotta.params,
        rows=rows,
        sql=sql,
        sql_params=sql_params,
        max_rows=cfg.structured_max_rows,
    )
    route_servita = router.STRUCTURED
    return RagResult(
        answer_text=testo,
        refused=False,
        refusal_reason=None,
        route=route_servita,
        structured=esito,
        **_base(cfg, query, rotta, route_servita),
    )
