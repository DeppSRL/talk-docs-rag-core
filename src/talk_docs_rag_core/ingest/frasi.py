"""Frasi ricorrenti: l'unità in cui vive un fatto che non appartiene a nessun documento.

Il difetto misurato (STATUS, 3b) sembrava ridondanza di *chunk* e non lo è: sui 13.670
chunk del corpus i duplicati esatti sono 123 e i quasi-duplicati 156 — **l'1,1%**. Fra i
595 chunk che contengono «legge 27 febbraio 1967, n. 48» la somiglianza mediana (Jaccard su
6-grammi) è **0,027**: non si somigliano affatto, condividono una frase dentro testi per il
resto diversi. Deduplicare i chunk avrebbe tolto l'1% del corpus e non avrebbe cambiato
niente.

L'unità giusta è la **frase**. Misurato: 648 frasi compaiono in 10 o più documenti, 74 in
50 o più, 22 in oltre 100. Sono i richiami normativi, le date istitutive e le definizioni
che ogni delibera ricopia nelle premesse. Quando la domanda verte su una di queste, il
retrieval non sceglie una fonte: ne **sorteggia** una fra centinaia equivalenti, e la
citazione che ne esce è corretta e arbitraria insieme.

Questo modulo costruisce, in ingest, l'indice di quelle frasi: quante volte ricorrono e in
quanti documenti. Serve a `rag.provenienza` per **dichiararlo** nella risposta, con numeri
calcolati e non stimati.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

# Abbreviazioni dopo le quali il punto NON chiude la frase. Senza, «legge 27 febbraio 1967,
# n. 48» si spezza in due e la frase ricorrente non si riconosce più: il primo tentativo di
# misura contava «Visto il decreto-legge 31 maggio 2010, n.» come frase a sé, 216 volte.
_ABBR = frozenset(
    "n art artt c cc lett cfr pag pagg es cap par comma tab fig prot reg ss nn "
    "lgs dlgs dl dpr dpcm dm on sig dott ing avv prof rep min".split()
)
# Candidato di fine frase: punteggiatura forte, spazio, e una maiuscola (o una virgoletta
# di apertura). Il filtro vero non sta qui ma in `dividi_in_frasi`, che guarda la parola
# *prima* del punto: il lookbehind a larghezza variabile non è esprimibile in `re`, e
# l'elenco delle abbreviazioni è lungo.
_CANDIDATO = re.compile(r"[.;:]\s+(?=[A-ZÀÈÉÌÒÙ«“\"])")
_PAROLA_PRIMA = re.compile(r"([\w.]+)[.;:]\s*$")

_LUNGHEZZA_MIN = 40  # sotto, è un frammento: «VISTO che» ricorre ovunque e non dice nulla
_LUNGHEZZA_MAX = 800


def _e_abbreviazione(testo: str, fine: int) -> bool:
    """La punteggiatura a `fine` chiude un'abbreviazione invece di una frase?"""
    m = _PAROLA_PRIMA.search(testo[:fine + 1])
    if m is None:
        return False
    parola = m.group(1).rstrip(".").replace(".", "").lower()
    # Solo l'elenco: proteggere anche le cifre («…, n. 141.») impedirebbe a una frase di
    # finire su un numero, cosa frequentissima in un corpus di richiami normativi — e
    # farebbe collassare in un'unica frase l'intera premessa di una delibera.
    return parola in _ABBR


def dividi_in_frasi(testo: str) -> list[str]:
    """Frasi di un testo, senza spezzare sulle abbreviazioni giuridiche.

    Il primo tentativo di misura tagliava su ogni punto e contava «Visto il decreto-legge
    31 maggio 2010, n.» come frase a sé — 216 volte. Non è un dettaglio di parsing: la
    frase ricorrente è **il richiamo alla norma**, e troncarla prima del numero toglie
    proprio la cosa che la identifica.
    """
    testo = testo or ""
    frasi, inizio = [], 0
    for m in _CANDIDATO.finditer(testo):
        if _e_abbreviazione(testo, m.start()):
            continue
        pezzo = " ".join(testo[inizio : m.start() + 1].split())
        if _LUNGHEZZA_MIN <= len(pezzo) <= _LUNGHEZZA_MAX:
            frasi.append(pezzo)
        inizio = m.end()
    coda = " ".join(testo[inizio:].split())
    if _LUNGHEZZA_MIN <= len(coda) <= _LUNGHEZZA_MAX:
        frasi.append(coda)
    return frasi


def ricongiungi_sillabazione(testo: str) -> str:
    """«Attri- buzioni» → «Attribuzioni».

    Il corpus conserva la sillabazione di fine riga della Gazzetta Ufficiale, e il punto in
    cui la parola si spezza dipende dall'impaginazione della singola delibera. È l'artefatto
    che rendeva invisibile il fenomeno: la frase sulla legge istitutiva del CIPE ricorre in
    289 documenti, ma come stringa è diversa in quasi tutti — «recan- te», «recante»,
    «recan-te». Senza questa riparazione l'indice contava 128 frasi ricorrenti e mancava
    proprio quelle su cui il giudizio umano aveva segnalato il problema.
    """
    return re.sub(r"(\w)-\s+(\w)", r"\1\2", testo or "")


def normalizza(frase: str) -> str:
    """Forma su cui si confrontano due frasi.

    I numeri diventano segnaposto: due delibere che richiamano la stessa legge la scrivono
    identica, ma la frase attorno può portare l'importo o la data della singola delibera.
    Mascherare le cifre riconosce la ricorrenza **senza** far coincidere due importi diversi
    — che è il rischio opposto, e sarebbe grave: un numero non è mai boilerplate.
    """
    f = re.sub(r"\d+", "#", ricongiungi_sillabazione(frase).lower())
    f = re.sub(r"[^\w#]+", " ", f)
    return " ".join(f.split())


def impronta(frase: str) -> str:
    return hashlib.sha256(normalizza(frase).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class FraseRicorrente:
    impronta: str
    testo: str  # una occorrenza, come esempio leggibile
    n_documenti: int
    n_occorrenze: int
    norme: tuple[str, ...]  # riferimenti normativi nominati dalla frase


@dataclass(frozen=True)
class IndiceFrasi:
    """Le sole frasi che ricorrono sopra soglia, per impronta. Il resto non serve.

    Accanto alle frasi c'è ``norme``: quante delibere nominano ciascuna norma in forma
    piena. Sono due chiavi diverse per lo stesso fenomeno, e servono entrambe perché
    falliscono in modi diversi. La frase è precisa ma **fragile alla coda**: «VISTA la legge
    23 agosto 1988, n. 400, recante …» finisce in modo diverso da una delibera all'altra e
    si spezza in due gruppi (111 e 74 documenti) invece di uno. La norma è **robusta**
    perché ignora ciò che le sta attorno — 289 documenti per la legge del 1967, contati
    senza incertezza — ma da sola non dice se il *fatto* affermato sia boilerplate.
    """

    frasi: dict[str, FraseRicorrente]
    norme: dict[str, int]
    n_documenti_corpus: int
    soglia: int

    def cerca(self, frase: str) -> FraseRicorrente | None:
        return self.frasi.get(impronta(frase))

    def in_passaggio(self, testo: str) -> list[FraseRicorrente]:
        """Le frasi ricorrenti contenute in un passaggio, dalla più diffusa alla meno."""
        trovate = {}
        for f in dividi_in_frasi(testo):
            r = self.cerca(f)
            if r is not None:
                trovate[r.impronta] = r
        return sorted(trovate.values(), key=lambda r: -r.n_documenti)

    def norma_diffusa(self, etichetta: str) -> int:
        """In quanti documenti compare la norma. 0 = sconosciuta o sotto soglia."""
        return self.norme.get(etichetta, 0)

    @classmethod
    def vuoto(cls) -> IndiceFrasi:
        return cls(frasi={}, norme={}, n_documenti_corpus=0, soglia=0)

    def salva(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "n_documenti_corpus": self.n_documenti_corpus,
                    "soglia": self.soglia,
                    "norme": dict(sorted(self.norme.items(), key=lambda kv: -kv[1])),
                    "frasi": [
                        {
                            "impronta": f.impronta,
                            "testo": f.testo,
                            "n_documenti": f.n_documenti,
                            "n_occorrenze": f.n_occorrenze,
                            "norme": list(f.norme),
                        }
                        for f in sorted(self.frasi.values(), key=lambda x: -x.n_documenti)
                    ],
                },
                ensure_ascii=False,
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        return p

    @classmethod
    def carica(cls, path: str | Path) -> IndiceFrasi:
        """Indice da file. **Assente ⇒ indice vuoto, non errore**: la provenienza è un di
        più dichiarativo, e un corpus indicizzato prima di questo incremento deve
        continuare a rispondere."""
        p = Path(path)
        if not p.exists():
            return cls.vuoto()
        d = json.loads(p.read_text(encoding="utf-8"))
        frasi = {
            f["impronta"]: FraseRicorrente(
                impronta=f["impronta"],
                testo=f["testo"],
                n_documenti=f["n_documenti"],
                n_occorrenze=f["n_occorrenze"],
                norme=tuple(f.get("norme") or ()),
            )
            for f in d.get("frasi", [])
        }
        return cls(
            frasi=frasi,
            norme=d.get("norme") or {},
            n_documenti_corpus=d.get("n_documenti_corpus", 0),
            soglia=d.get("soglia", 0),
        )


def costruisci_indice(documenti: list[tuple[str, str]], soglia: int) -> IndiceFrasi:
    """Indice delle frasi ricorrenti da ``[(doc_id, testo)]``.

    Il conteggio che conta è quello dei **documenti**, non delle occorrenze: una frase
    ripetuta tre volte nella stessa delibera non è boilerplate del corpus, è la struttura di
    quel documento. È la differenza fra «lo dicono in 289» e «lo dice una, tre volte».
    """
    from talk_docs_rag_core.rag.norme import estrai_norme

    documenti_per_frase: dict[str, set[str]] = {}
    documenti_per_norma: dict[str, set[str]] = {}
    occorrenze: dict[str, int] = {}
    esempio: dict[str, str] = {}
    for doc_id, testo in documenti:
        for norma in estrai_norme(ricongiungi_sillabazione(testo)):
            documenti_per_norma.setdefault(norma, set()).add(doc_id)
        for frase in dividi_in_frasi(testo):
            k = impronta(frase)
            documenti_per_frase.setdefault(k, set()).add(doc_id)
            occorrenze[k] = occorrenze.get(k, 0) + 1
            esempio.setdefault(k, frase)

    frasi = {
        k: FraseRicorrente(
            impronta=k,
            testo=esempio[k],
            n_documenti=len(docs),
            n_occorrenze=occorrenze[k],
            norme=tuple(estrai_norme(esempio[k])),
        )
        for k, docs in documenti_per_frase.items()
        if len(docs) >= soglia
    }
    norme = {n: len(docs) for n, docs in documenti_per_norma.items() if len(docs) >= soglia}
    return IndiceFrasi(
        frasi=frasi,
        norme=norme,
        n_documenti_corpus=len({d for d, _ in documenti}),
        soglia=soglia,
    )
