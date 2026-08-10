"""La nota di provenienza: dire che la fonte citata è una fra molte equivalenti.

Il problema, dalle note del giudizio umano: *«la fonte è arbitraria: la presidenza è nelle
premesse di centinaia di delibere quasi identiche»*, *«qualunque cinque il retrieval
peschi, la risposta esce»*. La risposta è giusta, la citazione formalmente valida, e
insieme il rimando non regge il controllo — perché indica **uno** dei 289 posti in cui la
stessa formula compare, senza dire che sono 289 né qual è la fonte vera.

Questo modulo non migliora il retrieval: **dichiara** ciò che il retrieval ha fatto. La
nota è calcolata come le risposte del ramo aggregativo — nessuna chiamata al modello, il
numero viene dall'indice e in audit c'è tutto per rifarlo.

## L'ancora è lo span, non il passaggio

Misurato sulla run `eval-20260809T143350Z`: ancorare al *passaggio citato* non funziona.
`ic-03` nomina undici norme nei suoi passaggi, e una nota che le elenca è peggio di nessuna
nota; `ic-07` — una risposta su un importo — ne nomina una presente in 207 documenti, dove
la nota sarebbe puro rumore perché il **fatto** non è boilerplate, lo è la premessa che gli
sta attorno.

La discriminante è *l'affermazione poggia su boilerplate?*, e l'ancora esatta esiste già:
lo **span verbatim** che la guardia C3c fa dichiarare al modello per ogni claim. La stessa
guardia bocciata come proxy della fedeltà — le infedeli avevano rapporto 1,00, quattro
fedeli 0,00 — è ciò che rende preciso questo meccanismo. Non era peso morto: era in cerca
del suo lavoro.

## Perché due chiavi

La **frase** è precisa ma fragile alla coda; la **norma** è robusta ma da sola non dice se
il fatto sia boilerplate. Si usano entrambe, con la regola che le tiene oneste: se lo span
porta una **cifra propria** — un importo, una percentuale — non è boilerplate, qualunque
cosa dica la premessa attorno. È lo stesso principio per cui `normalizza` maschera i
numeri: *un numero non è mai boilerplate*, e su un banco che esiste per non far passare
cifre inventate è l'invariante da non rompere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ingest.frasi import IndiceFrasi, dividi_in_frasi, normalizza, ricongiungi_sillabazione
from rag.norme import estrai_norme

# Una cifra che vale come «numero proprio» dello span: due o più cifre, per non contare
# come tale il numero di una legge già catturato dal riferimento normativo (che viene
# rimosso prima). Le date pure (1967) restano fuori dal conto perché fanno parte del
# richiamo normativo, non dell'affermazione.
_CIFRA = re.compile(r"\d[\d.,]*")
_RIFERIMENTO = re.compile(
    r"(?:legge|decreto[-\s]legge|decreto legislativo|decreto del Presidente[^,;]*)\s+"
    r"\d{1,2}\s+\w+\s+\d{4},?\s*n\.?\s*\d+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FonteRicorrente:
    """Una formula su cui poggia la risposta, con quanto è diffusa nel corpus."""

    testo: str  # la frase, come compare nel passaggio citato
    n_documenti: int  # documenti del corpus che la contengono
    norme: tuple[str, ...]  # le norme che nomina (la fonte vera, quando c'è)
    passaggi: tuple[int, ...]  # i passaggi citati in cui compare
    chiave: str  # "frase" | "norma" — come è stata contata


@dataclass(frozen=True)
class Provenienza:
    fonti: tuple[FonteRicorrente, ...]
    n_documenti_corpus: int
    soglia: int
    # Tutto ciò che serve a rifare il conto senza rigiocare la risposta.
    dettaglio: dict = field(default_factory=dict)

    @property
    def vuota(self) -> bool:
        return not self.fonti


def _numero_proprio(span: str) -> bool:
    """Lo span porta una cifra sua, cioè afferma un dato specifico?

    Si toglie prima il riferimento normativo per intero: «legge 27 febbraio 1967, n. 48» è
    fatta di cifre che appartengono al *nome della norma*, non all'affermazione. Ciò che
    resta, se è un numero, è il fatto — e un fatto numerico non è mai boilerplate.
    """
    resto = _RIFERIMENTO.sub(" ", span or "")
    return any(len(m.group(0).replace(".", "").replace(",", "")) >= 2 for m in _CIFRA.finditer(resto))


# Quanta parte dello span deve ritrovarsi nella frase perché sia *quella* frase.
# 0,6 tollera un refuso e una coda diversa, non due frasi che parlano della stessa cosa.
_SOVRAPPOSIZIONE_MIN = 0.6


def _frase_dello_span(passaggio: str, span: str) -> str | None:
    """La frase del passaggio su cui poggia lo span dichiarato dal modello.

    Il confronto **non può essere letterale**, e non è una concessione: lo span è
    *trascritto* dal modello, e la trascrizione sbaglia. Misurato su `ic-03`, l'item più
    emblematico del difetto — «Con quale legge è stato istituito il CIPE?»: lo span dice
    «Atribuzioni» dove il passaggio dice «Attribuzioni». Con il confronto per sottostringa
    la nota taceva proprio dove serviva. Lo stesso refuso è la ragione per cui quell'item
    aveva rapporto verbatim 0,00 (STATUS, taratura della guardia).

    Si sceglie quindi la frase con la maggiore sovrapposizione di parole, sopra soglia:
    tollera un refuso, la sillabazione e le virgolette, e continua a distinguere due frasi
    diverse dello stesso passaggio.
    """
    parole_span = set(normalizza(span).split())
    if len(parole_span) < 5:  # troppo corto per identificare una frase: meglio non indovinare
        return None
    migliore, punteggio = None, 0.0
    for frase in dividi_in_frasi(passaggio or ""):
        parole = set(normalizza(frase).split())
        if not parole:
            continue
        quota = len(parole_span & parole) / len(parole_span)
        if quota > punteggio:
            migliore, punteggio = frase, quota
    return migliore if punteggio >= _SOVRAPPOSIZIONE_MIN else None


def calcola(
    claims: list[dict],
    passaggi: dict[int, str],
    indice: IndiceFrasi,
    min_documenti: int,
) -> Provenienza:
    """Le fonti ricorrenti su cui poggiano i claim della risposta.

    ``passaggi`` mappa il numero mostrato al modello al testo del passaggio. ``claims`` è
    la lista già validata: ognuno porta ``verbatim`` e ``passages``.
    """
    if not indice.frasi and not indice.norme:
        return Provenienza(fonti=(), n_documenti_corpus=0, soglia=min_documenti)

    per_chiave: dict[str, dict] = {}
    for claim in claims or []:
        span = (claim.get("verbatim") or "").strip()
        if not span:
            continue
        span_ha_un_dato = _numero_proprio(span)
        for n in claim.get("passages") or []:
            frase = _frase_dello_span(passaggi.get(n, ""), span)
            if frase is None:
                continue
            ric = indice.cerca(frase)
            if ric is not None and ric.n_documenti >= min_documenti:
                v = per_chiave.setdefault(
                    ric.impronta,
                    {"testo": frase, "n": ric.n_documenti, "norme": ric.norme, "passaggi": set(), "chiave": "frase"},
                )
                v["passaggi"].add(n)
                continue
            # Ripiego sulla norma: la frase può non combaciare per via della coda, ma il
            # richiamo normativo che contiene è contato senza incertezza. Non si applica
            # quando lo span afferma un dato suo: lì la premessa è ricorrente, il fatto no.
            if span_ha_un_dato:
                continue
            for norma in estrai_norme(ricongiungi_sillabazione(frase)):
                quanti = indice.norma_diffusa(norma)
                if quanti < min_documenti:
                    continue
                v = per_chiave.setdefault(
                    f"norma::{norma}",
                    {"testo": frase, "n": quanti, "norme": (norma,), "passaggi": set(), "chiave": "norma"},
                )
                v["passaggi"].add(n)

    fonti = tuple(
        FonteRicorrente(
            testo=v["testo"],
            n_documenti=v["n"],
            norme=tuple(v["norme"]),
            passaggi=tuple(sorted(v["passaggi"])),
            chiave=v["chiave"],
        )
        for v in sorted(per_chiave.values(), key=lambda x: -x["n"])
    )
    return Provenienza(
        fonti=fonti,
        n_documenti_corpus=indice.n_documenti_corpus,
        soglia=min_documenti,
        dettaglio={"n_fonti": len(fonti), "chiavi": [f.chiave for f in fonti]},
    )


def _citazioni(passaggi: tuple[int, ...]) -> str:
    return "".join(f"[{n}]" for n in passaggi)


def rendi(prov: Provenienza) -> str:
    """La nota, in prosa. Vuota se non c'è niente da dichiarare.

    Tre cose e in quest'ordine, perché è l'ordine in cui servono a chi legge: **quanto è
    diffusa** la formula, **qual è la norma** che ne è la fonte, e che **il risultato non
    cambia**. L'ultima non è una rassicurazione di cortesia: senza, il numero «289» si legge
    come un'ammissione di debolezza invece che come una dichiarazione di copertura.
    """
    if prov.vuota:
        return ""
    prima = prov.fonti[0]
    quota = f"{prima.n_documenti} delle {prov.n_documenti_corpus} delibere indicizzate"
    if prima.norme:
        norme = " e ".join(f"**{n}**" for n in prima.norme[:2])
        apertura = f"L'affermazione poggia sul richiamo a {norme}, che ne è la fonte: ricorre in {quota}."
    else:
        apertura = f"L'affermazione poggia su una formula che ricorre in {quota}."

    cit = _citazioni(prima.passaggi)
    if len(prima.passaggi) == 1:
        rimando = f"Il passaggio citato in {cit} è uno di quelli"
    else:
        rimando = f"I passaggi citati in {cit} sono fra quelli"
    chiusura = (
        f"{rimando} — gli altri la riportano nei medesimi termini, e la risposta non cambia."
    )

    righe = ["**Provenienza.**", apertura, chiusura]
    # Le fonti successive si dichiarano in blocco: elencarle una per una raddoppierebbe la
    # nota per dire la stessa cosa. Il numero che conta l'ha già dato la prima.
    altre = prov.fonti[1:]
    if altre:
        n = max(f.n_documenti for f in altre)
        quante = "un'altra formula ricorrente" if len(altre) == 1 else f"altre {len(altre)} formule ricorrenti"
        righe.append(f"La risposta poggia anche su {quante} (fino a {n} delibere).")
    return " ".join(righe)


def applica(answer_text: str, prov: Provenienza) -> str:
    """Aggiunge la nota in coda alla risposta. Nota vuota ⇒ testo invariato."""
    nota = rendi(prov)
    return f"{answer_text}\n\n{nota}" if nota else answer_text
