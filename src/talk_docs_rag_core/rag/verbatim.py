"""Guardia verbatim: verifica che le parole dichiarate esistano dove il modello dice.

È lo stesso spirito del guardiano IDF di ``rag.guard``, applicato all'**output** invece
che all'input. Nessun giudice, nessun modello nel ciclo di verifica: un ``in`` fra stringhe
normalizzate.

**Limite da tenere presente leggendo i numeri** (spec §6): non cattura il caso in cui le
parole *esistono* ma si riferiscono ad altro — la cifra giusta della spesa sbagliata. Quel
guasto ha causa nel retrieval e non ha rimedio deterministico a valle.

La normalizzazione di base è **quella di** ``rag.guard`` (chiamata, non ricopiata): due nozioni
diverse di «stessa stringa» nello stesso sistema sono un guasto che si manifesta mesi dopo.
Qui si aggiungono solo i passi che servono al confronto di span lunghi (sillabazione da PDF,
whitespace collassato).

**Ripulitura degli artefatti del modello, e perché solo sullo span** (misurato su
``logs/run-20260807T162233Z.jsonl``: 0/2 validi su una risposta corretta nel merito). Il
modello restituisce i suoi span racchiusi fra virgolette che ha aggiunto lui, o troncati con
un'ellissi: le parole *sono* nel passaggio, a fallire è il confronto. Quella punteggiatura si
toglie **allo span e mai al passaggio** — nelle delibere le caporali sono testo della delibera
(«Regolamento interno del Comitato…») e ripulirle falserebbe il riferimento. Togliere di meno
allo span di quanto si toglie al passaggio non fa danno: se lo span pulito è contenuto nel
passaggio grezzo, il passaggio le sue virgolette ce le ha ancora, intorno.

Terzo artefatto, misurato su ``logs/eval-20260807T164834Z.jsonl``: il modello incolla i
**marcatori di citazione** del contratto (``[2]``, ``[3][4][5]``) *dentro* il campo ``verbatim``,
che dovrebbe contenere solo le parole del passaggio — «…il 5° lotto costruttivo **[2]**». Anche
qui le parole ci sono: a fallire è la sintassi del contratto finita nel campo sbagliato. Si
tolgono **ovunque** compaiano, non solo agli estremi, perché nella prosa del modello capitano
anche in mezzo. Solo cifre: ``[...]`` **non** è un marcatore ma un'ellissi in quadre, cioè testo
saltato, e quello span deve restare non valido.

Prezzo dichiarato dell'asimmetria: se le quadre col numero sono testo *del passaggio*, lo span
che le trascrive fedelmente risulta non valido. Ripulire anche il passaggio renderebbe però
indistinguibile un marcatore inventato da uno trascritto — falso negativo della guardia, errore
peggiore. Nel corpus il caso è raro: 3 occorrenze in 1 delibera su 511.

Un'ellissi **in mezzo** allo span non si tocca: lì il modello ha saltato del testo, lo span non
è più letterale, e deve continuare a risultare non valido.
"""

from __future__ import annotations

import re

from . import guard
from .outcomes import VerbatimOutcome

ESITO_VALIDO = "valido"
ESITO_MISATTRIBUITO = "misattribuito"
ESITO_NON_TROVATO = "non_trovato"
ESITO_TROPPO_CORTO = "troppo_corto"

_WS = re.compile(r"\s+")
_FINE_FRASE = (".", "!", "?", ":", ";")

# Ellissi di troncamento del modello, nelle due forme, solo agli estremi.
_ELLISSI = re.compile(r"^\s*(?:\.{3,}|…)\s*|\s*(?:\.{3,}|…)\s*$")
# Marcatore di citazione del contratto, ovunque nello span. **Solo cifre**: `[...]` è
# un'ellissi in quadre — testo saltato dal modello — e non va tolta.
_MARCATORE = re.compile(r"\[\d+\]")
# Solo coppie che *racchiudono*: una virgoletta spaiata è testo, non contorno.
_COPPIE = (('"', '"'), ("«", "»"), ("“", "”"), ("‘", "’"), ("”", "”"), ("„", "“"))


def normalizza(testo: str) -> str:
    """Minuscole, apostrofi unificati, sillabazione da PDF riunita, whitespace collassato.

    Il collasso è essenziale: il testo estratto da PDF ha a capo arbitrari a metà frase, e
    senza di esso qualunque citazione letterale corretta risulterebbe assente.
    """
    t = guard.normalizza(testo)
    t = guard.SILLABAZIONE.sub("", t)
    return _WS.sub(" ", t).strip()


def ripulisci_span(span: str) -> str:
    """Toglie allo span gli artefatti di *formattazione* del modello.

    Virgolette di contorno (dritte, caporali, curve) ed ellissi di troncamento **agli estremi**,
    annidate in qualunque ordine — da cui il ciclo; marcatori di citazione ``[n]`` **ovunque**,
    perché nella prosa del modello capitano anche in mezzo, con gli spazi doppi che ne restano.
    Non si applica ai passaggi: vedi il docstring del modulo.
    """
    t = span.strip()
    while True:
        prima = t
        senza_marcatori = _MARCATORE.sub("", t)
        if senza_marcatori != t:
            t = _WS.sub(" ", senza_marcatori).strip()
        t = _ELLISSI.sub("", t).strip()
        for apre, chiude in _COPPIE:
            if len(t) >= 2 and t.startswith(apre) and t.endswith(chiude):
                t = t[1:-1].strip()
                break
        if t == prima:
            return t


def _a_confine_di_frase(span: str, contenuto: str) -> bool:
    """Diagnostica, non guardia: lo span comincia a inizio frase e finisce con punteggiatura."""
    i = contenuto.find(span)
    if i < 0:
        return False
    prima_ok = i == 0 or contenuto[i - 1] in (" ",) and (i < 2 or contenuto[i - 2] in _FINE_FRASE)
    dopo = contenuto[i + len(span) :].lstrip()
    dopo_ok = not dopo or span.endswith(_FINE_FRASE) or dopo[0] in _FINE_FRASE
    return bool(prima_ok and dopo_ok)


def verifica(claims: list[dict], passaggi: list, min_chars: int) -> VerbatimOutcome:
    """Esito per ogni claim + gli aggregati che finiscono in audit e nell'eval."""
    norm_per_n = {p.n: normalizza(p.content) for p in passaggi}
    per_claim: list[dict] = []
    n_valid = n_mis = n_non = n_corto = 0
    n_confine = 0

    for c in claims:
        # Lo span registrato è quello **confrontato**: chi rilegge l'audit deve poter rifare
        # l'`in` e ottenere lo stesso esito. La dichiarazione grezza del modello resta in `claims`.
        span = ripulisci_span(c.get("verbatim") or "")
        citati = [int(n) for n in c.get("passages", []) if int(n) in norm_per_n]
        voce = {
            "statement": c.get("statement", ""),
            "passages": citati,
            "verbatim": span,
            "matched_passage": None,
        }
        # Lo span vuoto non dipende dalla soglia: `"" in qualunque_testo` è sempre vero, e con
        # `min_chars=0` ogni claim senza verbatim risulterebbe valido — metrica ottimistica per
        # configurazione, non per merito.
        if not span or len(span) < min_chars:
            n_corto += 1
            voce["esito"] = ESITO_TROPPO_CORTO
            per_claim.append(voce)
            continue

        n_span = normalizza(span)
        trovato_in = next((n for n in citati if n_span in norm_per_n[n]), None)
        if trovato_in is not None:
            n_valid += 1
            voce["esito"] = ESITO_VALIDO
            voce["matched_passage"] = trovato_in
            if _a_confine_di_frase(n_span, norm_per_n[trovato_in]):
                n_confine += 1
        else:
            altrove = next((n for n, t in norm_per_n.items() if n_span in t), None)
            if altrove is not None:
                n_mis += 1
                voce["esito"] = ESITO_MISATTRIBUITO
                voce["matched_passage"] = altrove
            else:
                n_non += 1
                voce["esito"] = ESITO_NON_TROVATO
        per_claim.append(voce)

    n = len(claims)
    return VerbatimOutcome(
        n_claims=n,
        n_valid=n_valid,
        n_misattributed=n_mis,
        n_not_found=n_non,
        n_too_short=n_corto,
        # `None` quando non ci sono claim: il rapporto non è definito, e la guardia non
        # deve scattare su un valore inventato — stessa logica del guardiano IDF senza
        # `TermStats`.
        valid_ratio=(n_valid / n) if n else None,
        span_boundary_ratio=(n_confine / n_valid) if n_valid else None,
        per_claim=per_claim,
    )
