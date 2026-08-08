"""Router agentico: un classificatore LLM a valle del router lessicale.

**Il modello propone, la pipeline valida.** La proposta è vincolata al semantic layer
chiuso: route nell'insieme ``router.ROUTES``, intento in ``intents.TUTTI``, parametri
sanificati e legabili (``intents.build``). Il modello non scrive mai SQL, numeri o testi
di rifiuto — sceglie fra query a template, che è il task su cui un 14B regge
(istruttoria §2.2). Qualunque violazione, JSON malformato o errore di rete fa ricadere
sulla classificazione lessicale: **il routing non può mai far fallire una risposta.**

Perché esiste, coi numeri (run ``eval-20260807T181416Z``): richiamo lessicale 71,4%
(le riformulazioni colloquiali ``ag-05``/``ag-06`` non matchano), ``fp-04`` conta
delibere a chi chiede atti processuali (la parola nuda ``atti`` matcha, e distinguere
è conoscenza di dominio, non una regex), ``ic-06`` perde una risposta che c'era. La
conoscenza che manca al punto di instradamento è *che cosa sia questo corpus*: la
scheda (``rag/corpus_card.py``) la porta, questo modulo la usa.

**Nasce spento** (``router_llm_enabled = False``): introduce varianza nel punto che
decide quale pipeline risponde — la ragione per cui l'incremento 1 lo escludeva resta
vera — quindi prima una run dedicata che lo misuri contro il lessicale, poi la
decisione. La proposta finisce in audit anche quando il fallback la annulla.

Guardia dura non scavalcabile: la regola «delibera specifica → puntuale» è decisa in
pipeline *prima* di consultare questo modulo.
"""

from __future__ import annotations

import json
import logging
import time

from openai import OpenAI

from config import RagConfig
from rag import router
from rag.corpus_card import CorpusCard
from rag.generation import usage_dict
from structured import intents

logger = logging.getLogger(__name__)

# Il primo anno dell'archivio storico (1967) come limite basso di plausibilità di un
# filtro anno; il limite alto è largo apposta — la validazione difende il contratto
# della query, non indovina il corpus.
_ANNO_MIN, _ANNO_MAX = 1900, 2100

# Le istruzioni dicono REGOLE, non esempi presi dall'eval set. Mettere in prompt le
# domande su cui il router viene misurato (`ic-07`, `ag-01`, `oc-01`…) le farebbe passare
# per costruzione e renderebbe la misura successiva priva di valore: si insegnerebbe al
# test. Le poche formulazioni illustrative qui sotto sono inventate e non compaiono in
# `eval/eval_set.jsonl`.
_ISTRUZIONI = (
    "Sei il router di un sistema di domande e risposte su un corpus documentale. Il tuo "
    "UNICO compito è classificare la domanda dell'utente in una route. Non rispondi mai "
    "alla domanda, non giudichi se la risposta esista: decidi solo CHI deve cercarla.\n\n"
    "La discriminante non è l'argomento della domanda, è DOVE STA LA RISPOSTA.\n\n"
    "Route disponibili:\n"
    '- "pointwise": la risposta è scritta nel testo di uno o pochi documenti. È la route '
    "PREDEFINITA: nel dubbio si sceglie questa, perché a valle il sistema ha già le sue "
    "guardie e rifiuta da sé quando i documenti non bastano.\n"
    "  Vi rientrano anche: le domande su un importo, una data, una percentuale o una "
    "quantità RIPORTATI in un atto (un valore scritto è un fatto da leggere, non un "
    "totale da calcolare — «quanto», «a quanto ammonta», «quante risorse» NON bastano a "
    "spostare la domanda altrove); e le domande su un argomento estraneo al corpus, che "
    "il sistema rifiuterà semplicemente perché non trova documenti pertinenti.\n"
    '- "structured": conteggio, elenco o distribuzione di DELIBERE del Comitato, '
    "calcolabile sui metadati (anno, numero, comitato). Intenti disponibili:\n"
    '  - "count_delibere" (parametri opzionali: "anno" oppure "anno_da"+"anno_a", "comitato")\n'
    '  - "list_delibere" (stessi parametri)\n'
    '  - "count_by_year" (parametro opzionale: "comitato")\n'
    '  "comitato" vale solo "CIPE" o "CIPESS". Un anno singolo e un intervallo sono '
    "alternativi. Senza alcun filtro il conteggio vale sull'intero corpus ed è "
    "ugualmente calcolabile. Un insieme di anni sciolti NON è esprimibile: in quel caso "
    'la route giusta è "uncovered".\n'
    '- "uncovered": SOLO le domande che per rispondere richiederebbero di attraversare '
    "molti documenti e sommare grandezze prese dai loro testi — spese complessive di un "
    "settore in un periodo, quantità fisiche aggregate — oppure di contare oggetti che "
    "NON sono delibere del Comitato. Non è la route del «non lo so»: è la route di un "
    "calcolo che manca. Se la domanda non chiede un aggregato, non è questa.\n"
    '- "meta": la domanda riguarda la collezione in quanto tale — che cos\'è, di che cosa '
    "si occupa, che periodo copre, come sono fatti i documenti, cosa si può chiedere. "
    "Chiedere QUANTE delibere ci sono non è una meta-domanda: è un conteggio "
    '("structured").\n\n'
    "Rispondi SOLO con un oggetto JSON: "
    '{"route": "...", "intent": null | "...", "params": {}}. '
    '"intent" e "params" si valorizzano solo per "structured". Non inventare intenti, '
    "route o parametri fuori da quelli elencati."
)


def _sanifica_params(grezzi: dict) -> dict | None:
    """Parametri della proposta ridotti al contratto di ``intents._where``.

    ``None`` = proposta non conforme (si ricade sul lessicale). Le chiavi ignote si
    scartano senza invalidare: un campo in più è rumore del modello, un anno impossibile
    o un comitato inventato sono una proposta da non eseguire. ``anno`` insieme
    all'intervallo, o un intervallo monco, violano il contratto («mutuamente esclusivi
    per contratto del chiamante») che qui — dove il chiamante è un modello — va
    verificato invece che assunto.
    """
    if not isinstance(grezzi, dict):
        return None
    puliti: dict = {}
    for chiave in ("anno", "anno_da", "anno_a"):
        val = grezzi.get(chiave)
        if val is None:
            continue
        try:
            anno = int(val)
        except (TypeError, ValueError):
            return None
        if not (_ANNO_MIN <= anno <= _ANNO_MAX):
            return None
        puliti[chiave] = anno
    if "anno" in puliti and ("anno_da" in puliti or "anno_a" in puliti):
        return None
    if ("anno_da" in puliti) != ("anno_a" in puliti):
        return None
    if "anno_da" in puliti and puliti["anno_da"] > puliti["anno_a"]:
        return None
    comitato = grezzi.get("comitato")
    if comitato is not None:
        if not isinstance(comitato, str) or comitato.strip().upper() not in ("CIPE", "CIPESS"):
            return None
        puliti["comitato"] = comitato.strip().upper()
    return puliti


class AgenticRouter:
    """Costruito una volta in ``build_pipeline`` (prompt stabile → provider cache);
    consultato per domanda quando ``router_llm_enabled`` e la guardia dura non ha già
    deciso."""

    def __init__(self, cfg: RagConfig, client: OpenAI, card: CorpusCard | None, corpus_version: str = ""):
        self.cfg = cfg
        # Budget di retry PROPRIO, più alto di quello della generazione. Misurato sulla run
        # `eval-20260808T122852Z`: 4 chiamate su 55 morte in `RateLimitError` e ricadute sul
        # lessicale — fra queste tre delle sette aggregative, cioè il richiamo del router
        # misurava la rete invece del classificatore. Il ramo agentico raddoppia le richieste
        # al provider e l'eval le fa in sequenza stretta: è il punto della pipeline che tocca
        # per primo il rate limit, e l'unico dove un fallimento si traveste da decisione.
        self.client = client.with_options(max_retries=cfg.router_llm_max_retries)
        self.cache_key = f"router:{corpus_version}"
        scheda = card.text if card is not None else "(nessuna scheda compilata per questo corpus)"
        # Prefisso byte-identico a ogni richiesta, scheda inclusa: stesso principio
        # cache-friendly del SYSTEM_PREFIX di generazione, con prompt_cache_key proprio.
        self.system_prompt = f"{_ISTRUZIONI}\n\nSCHEDA DEL CORPUS:\n\n{scheda}"

    def _chiama(self, query: str):
        common = dict(
            model=self.cfg.mistral_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Domanda: {query}"},
            ],
            temperature=self.cfg.llm_temperature,
            max_tokens=self.cfg.router_llm_max_tokens,
            response_format={"type": "json_object"},
        )
        if self.cfg.llm_seed is not None:
            common["random_seed"] = self.cfg.llm_seed
        return self.client.chat.completions.create(extra_body={"prompt_cache_key": self.cache_key}, **common)

    def classify(self, query: str, lessicale: router.Route) -> router.Route:
        """La route servita: la proposta LLM se supera la validazione, il lessicale
        altrimenti. In entrambi i casi la traccia della chiamata viaggia in ``Route.llm``
        e da lì nell'audit — una proposta annullata è un dato, non un evento perso."""
        t0 = time.perf_counter()
        traccia: dict = {"proposta": None, "raw": "", "usage": {}, "error": None}

        def _fallback(errore: str) -> router.Route:
            traccia["error"] = errore
            traccia["latency_s"] = round(time.perf_counter() - t0, 3)
            logger.warning("Router agentico: fallback sul lessicale (%s) — query: %.80s", errore, query)
            lessicale.llm = traccia
            return lessicale

        try:
            response = self._chiama(query)
        except Exception as exc:  # il routing non può far fallire la risposta
            return _fallback(f"chiamata fallita: {exc.__class__.__name__}")

        traccia["usage"] = usage_dict(response)
        raw = response.choices[0].message.content or ""
        traccia["raw"] = raw
        try:
            proposta = json.loads(raw)
        except json.JSONDecodeError:
            return _fallback("JSON non parsabile")
        if not isinstance(proposta, dict):
            return _fallback("proposta non è un oggetto")
        traccia["proposta"] = proposta

        route = proposta.get("route")
        if route not in router.ROUTES:
            return _fallback(f"route fuori insieme: {route!r}")

        intent = None
        params: dict = {}
        if route == router.STRUCTURED:
            intent = proposta.get("intent")
            if intent not in intents.TUTTI:
                return _fallback(f"intento fuori insieme: {intent!r}")
            sanificati = _sanifica_params(proposta.get("params") or {})
            if sanificati is None:
                return _fallback("parametri non conformi")
            if intents.build(intent, sanificati) is None:
                return _fallback("parametri non legabili")
            params = sanificati

        traccia["latency_s"] = round(time.perf_counter() - t0, 3)
        # I segnali lessicali restano nel risultato: il disaccordo lessicale/LLM si conta
        # in audit confrontandoli con la route servita, senza rigiocare la run.
        return router.Route(
            route=route,
            intent=intent,
            params=params,
            signals=lessicale.signals,
            source="llm",
            llm=traccia,
        )
