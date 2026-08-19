"""Router deterministico: quale via risponde a questa domanda.

**Nessun LLM.** Il banco gira a ``temperature=0`` e un classificatore introdurrebbe
varianza proprio nel punto che decide *quale* pipeline risponde, sporcando il confronto
A/B del caching che è il deliverable centrale del PoC. Rifiuto e astensione sono già rami
di codice deterministici: questo è il terzo.

**Limite noto, non risolto (spec §7).** Il difetto di un router lessicale non è il rifiuto,
è il silenzio: un'aggregativa formulata fuori dai pattern non finisce in ``UNCOVERED``,
cade in ``POINTWISE`` — cioè nel RAG di oggi, che su «quante delibere nel 2024» risponde
«almeno due» invece di 93, fluentemente e con una citazione ben formata. Per questo i
segnali grezzi finiscono nel risultato e da lì nell'audit: un fall-through si conta.

L'errore opposto è altrettanto reale: «**quanto** vale il fondo previsto dalla delibera 47»
è puntuale e rispondibile. La regola che lo disinnesca è la prima della cascata — il
riferimento a una delibera **specifica** vince su qualunque forma aggregativa.

**Perché più anni citati sono un rifiuto e non un conteggio.** Gli intenti sono chiusi: sanno
filtrare un anno o un intervallo, non un insieme di anni sciolti. Su «quante delibere del CIPE
nel 2019 e nel 2021» il filtro anno non è esprimibile, e lasciarlo cadere in silenzio non
produce né un rifiuto né un'allucinazione — produce il conteggio **globale** del comitato: un
numero *calcolato*, e quindi credibile, alla domanda sbagliata. È la classe di guasto peggiore
per questo banco, perché non lascia traccia di sé nella risposta. Vale qui il principio già
scritto in ``structured.intents.build``: meglio dire «non so contare questo» che eseguire una
query su un filtro indovinato. Un intervallo riconosciuto non basta a chiudere il caso: se la
domanda cita anche un anno **oltre** i suoi estremi («dal 2019 al 2021 e nel 2024») il range da
solo produrrebbe un conteggio *monco*, indistinguibile da uno giusto — stessa classe di guasto.

**Compromesso dichiarato su ``atti``.** In ``_OGGETTO`` la parola nuda ``atti`` copre
formulazioni legittime («quanti atti del 2024») ma cattura anche domande fuori dominio
(«quanti atti processuali»), che ricevono un conteggio di delibere: il rischio è basso su
questo corpus ed è **misurato dall'eval** (item ``fp-04``), non assunto.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from talk_docs_rag_core.structured import intents

STRUCTURED = "structured"
UNCOVERED = "uncovered"
POINTWISE = "pointwise"
META = "meta"

# Tutte le route esprimibili: è l'insieme chiuso contro cui il router agentico valida la
# proposta del modello (rag/agentic_router.py).
ROUTES = (STRUCTURED, UNCOVERED, POINTWISE, META)

# Riferimento a un atto specifico: «delibera 47», «delibera n. 75/2021». Vince su tutto.
_DELIBERA_SPECIFICA = re.compile(r"deliber\w*\s+(?:n\.?\s*)?\d+", re.IGNORECASE)
# Forma di CONTEGGIO: numerabile. «quante/quanti», non «quanto» (che è di massa).
_FORMA_CONTEGGIO = re.compile(r"\b(quant[ei]|numero\s+di|elenc\w+|quante\s+volte|lista\s+d)\b", re.IGNORECASE)
# Forma di MASSA: chiede un totale o un ammontare. Non abbiamo importi: sempre scoperta.
# Pattern spezzato solo per il limite di riga: le due raw string si concatenano a compile-time.
_FORMA_MASSA = re.compile(
    r"\b(in\s+totale|complessiv\w+|ammontare\s+totale|somma\s+totale"
    r"|totale\s+d\w+|quanto\s+(?:è\s+stato\s+)?spes\w+)\b",
    re.IGNORECASE,
)
# Oggetto coperto: gli atti del Comitato — l'unica cosa che il manifest sa contare.
_OGGETTO = re.compile(r"\b(deliber\w+|atti|provvediment\w+)\b", re.IGNORECASE)
# «quali sono» conta come richiesta di elenco SOLO insieme all'oggetto coperto: da solo è
# troppo debole e rifiuterebbe domande puntuali legittime («quali sono gli obiettivi del…»).
_ELENCO = re.compile(r"\b(elenc\w+|quali\s+sono|lista\s+d)\b", re.IGNORECASE)
_PER_ANNO = re.compile(r"\b(per\s+anno|anno\s+per\s+anno|distribu\w+)\b", re.IGNORECASE)
_ANNO = re.compile(r"\b(19[6-9]\d|20[0-9]\d)\b")
_RANGE = re.compile(r"\b(?:dal|da)\s+(\d{4})\s+(?:al|a)\s+(\d{4})\b", re.IGNORECASE)
_CIPESS = re.compile(r"\bcipess\b", re.IGNORECASE)
_CIPE = re.compile(r"\bcipe\b", re.IGNORECASE)
# Meta-domanda: chiede della COLLEZIONE, non di un contenuto. Due segnali entrambi necessari
# — la forma interrogativa meta e il riferimento esplicito alla collezione — perché la
# risposta (la scheda del corpus) è giusta solo se la domanda è davvero sull'archivio:
# «di cosa parlano le delibere del 2020?» è una domanda tematica (famiglia A, scoperta),
# non una meta-domanda, e deve restare POINTWISE.
_FORMA_META = re.compile(
    r"\b(di\s+cosa\s+(?:parla|tratta)|che\s+cos'?[èe]|cosa\s+(?:contiene|c'?[èe])"
    r"|(?:che|quali|quant[ei])\s+(?:tipo\s+di\s+)?document\w+|che\s+periodo|quale\s+periodo"
    r"|cosa\s+(?:posso|si\s+può)\s+chieder\w*|cosa\s+puoi\s+(?:fare|dirmi)"
    r"|come\s+sono\s+fatt\w+)\b",
    re.IGNORECASE,
)
_OGGETTO_CORPUS = re.compile(
    r"\b(corpus|archivio|raccolta|banca\s+dati|dataset|quest[oi]\s+document\w+)\b",
    re.IGNORECASE,
)

# Motivo del rifiuto per il caso multi-anno. Il rifiuto deve dire *quale* motivo: il testo di
# default parla dell'assenza della tabella degli importi, che qui sarebbe falso. Copre entrambe
# le varianti — anni sciolti, e un anno oltre gli estremi di un intervallo riconosciuto.
MOTIVO_ANNI_MULTIPLI = (
    "aggregazione fuori copertura: gli anni citati non stanno in un solo anno né in un solo intervallo"
)

# Tutti i motivi che questo modulo può emettere. Ogni voce **deve** avere la sua prosa in
# ``structured.service._TESTI_RIFIUTO``: là la mappa si legge con un default, quindi un motivo
# nuovo non mappato non esploderebbe — mostrerebbe all'utente il testo sbagliato («non ho la
# tabella degli importi») sotto un ``refusal_reason`` giusto. Un test verifica la copertura, così
# il buco si rompe in fase di sviluppo invece di mentire in produzione. Aggiungendo un motivo,
# aggiungilo qui.
MOTIVI = (MOTIVO_ANNI_MULTIPLI,)


@dataclass
class Route:
    route: str
    intent: str | None = None
    params: dict = field(default_factory=dict)
    signals: dict = field(default_factory=dict)
    # Motivo del rifiuto quando ``route == UNCOVERED``; ``None`` = il motivo di default
    # (nessuna tabella degli importi). Gli altri rami non lo valorizzano.
    reason: str | None = None
    # Chi ha deciso la route servita: "lexical" (questo modulo) o "llm" (il router
    # agentico, quando abilitato e quando la sua proposta supera la validazione).
    source: str = "lexical"
    # Traccia della classificazione LLM: proposta grezza, usage, eventuale errore.
    # Registrata anche quando il fallback la annulla — come `abstention_signal`, i
    # segnali si registrano sempre, le decisioni si ritarano su run passate.
    llm: dict | None = None


def _anni(query: str) -> tuple[list[int], tuple[int, int] | None]:
    """Anni distinti citati e l'eventuale intervallo «dal … al …».

    Sorgente unica per ``_filtri`` e ``classify``: «quanti anni» e «c'è un intervallo» sono
    la stessa lettura della domanda, non due conteggi scritti due volte che possono divergere.
    """
    m = _RANGE.search(query)
    intervallo = (int(m.group(1)), int(m.group(2))) if m else None
    return sorted({int(a) for a in _ANNO.findall(query)}), intervallo


def _filtri(query: str) -> dict:
    params: dict = {}
    anni, intervallo = _anni(query)
    if intervallo is not None:
        params["anno_da"], params["anno_a"] = intervallo
    elif len(anni) == 1:
        params["anno"] = anni[0]
    if _CIPESS.search(query):
        params["comitato"] = "CIPESS"
    elif _CIPE.search(query):
        params["comitato"] = "CIPE"
    return params


def _intento(query: str) -> str:
    if _PER_ANNO.search(query):
        return intents.COUNT_BY_YEAR
    if _ELENCO.search(query):
        return intents.LIST_DELIBERE
    return intents.COUNT_DELIBERE


def classify(query: str) -> Route:
    """Cascata deterministica. L'ordine delle regole *è* il progetto."""
    conteggio = bool(_FORMA_CONTEGGIO.search(query))
    massa = bool(_FORMA_MASSA.search(query))
    oggetto = bool(_OGGETTO.search(query))
    specifica = bool(_DELIBERA_SPECIFICA.search(query))
    anni, intervallo = _anni(query)
    # Un intervallo non spegne il segnale: se resta citato un anno che non è uno dei due
    # estremi, quell'anno non entra in nessun filtro esprimibile e il conteggio uscirebbe
    # monco — cioè calcolato, credibile e sbagliato, come i due anni sciolti.
    anni_multipli = bool(set(anni) - set(intervallo)) if intervallo is not None else len(anni) > 1
    forma_meta = bool(_FORMA_META.search(query))
    oggetto_corpus = bool(_OGGETTO_CORPUS.search(query))
    signals = {
        "forma_conteggio": conteggio,
        "forma_massa": massa,
        "oggetto_coperto": oggetto,
        "delibera_specifica": specifica,
        "anni_multipli": anni_multipli,
        "forma_meta": forma_meta,
        "oggetto_corpus": oggetto_corpus,
    }

    # 1. Un atto nominato per numero è una domanda puntuale, qualunque forma abbia.
    if specifica:
        return Route(POINTWISE, signals=signals)
    # 1b. Meta-domanda sulla collezione: la risposta è la scheda del corpus più le
    #     statistiche calcolate, non un contenuto. Prima delle forme di conteggio: «quanti
    #     documenti contiene il corpus» è una domanda sulla collezione, e un fall-through
    #     su UNCOVERED risponderebbe «non ho la tabella degli importi» a chi ha chiesto
    #     che cos'è l'archivio.
    if forma_meta and oggetto_corpus:
        return Route(META, signals=signals)
    # 2. Conteggio o elenco sugli atti del Comitato: è ciò che il manifest sa fare — ma solo se
    #    il filtro anno è esprimibile. Più anni sciolti non lo sono: rifiuto, non conteggio
    #    globale col filtro amputato via.
    if oggetto and (conteggio or _ELENCO.search(query)):
        if anni_multipli:
            return Route(UNCOVERED, signals=signals, reason=MOTIVO_ANNI_MULTIPLI)
        return Route(STRUCTURED, intent=_intento(query), params=_filtri(query), signals=signals)
    # 3. Conteggio su un oggetto che non sappiamo contare (chilometri, dipendenti).
    if conteggio:
        return Route(UNCOVERED, signals=signals)
    # 4. Un totale o un ammontare: non abbiamo una tabella degli importi.
    if massa:
        return Route(UNCOVERED, signals=signals)
    return Route(POINTWISE, signals=signals)
