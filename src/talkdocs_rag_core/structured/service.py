"""Dal router al ``RagResult``: il ramo aggregativo e il rifiuto dichiarato.

Nessuna chiamata al modello in tutto il file. ``usage`` resta vuoto per costruzione: se
comparissero token, il confronto A/B del caching starebbe misurando una chiamata che non
esiste.
"""

from __future__ import annotations

import logging

from config import RagConfig
from rag import router
from rag.generation import RagResult
from structured import intents
from structured.answer import componi
from structured.store import StructuredStore

logger = logging.getLogger(__name__)

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
