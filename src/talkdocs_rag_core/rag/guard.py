"""C3b — guardiano di astensione: il **terzo esito** della pipeline.

Problema misurato (run ``eval-20260805T083842Z``, item ``ic-07-bis``): la pipeline aveva due
soli esiti, rifiuto deterministico o risposta piena. Su «a quanto ammontano le somme
accantonate per le **quote premiali** 2020?» il ``support_score`` era **0,878 — sopra la
soglia 0,82** — mentre nessuno dei 5 passaggi conteneva «quote premiali»: il chunk giusto non
era nemmeno nei primi 20. Risultato: una cifra sbagliata, sicura e ben citata.

La causa è che ``support_score`` (miglior similarità densa) misura **vicinanza di argomento**,
non **presenza della risposta**: con il pavimento alto di ``mistral-embed`` (~0,77) e chunk
in-tema-ma-sbagliati a 0,88 non può distinguere i due casi. Serve un segnale ortogonale.

Segnale scelto: **IDF del termine mancante più raro**. Si estraggono i termini di contenuto
della domanda e si guarda quali non compaiono in *nessuno* dei passaggi recuperati; il
segnale è l'IDF (rarità nel corpus) del più raro fra i mancanti. Un termine comune assente
non dice nulla; un termine raro assente dice che la cosa specifica chiesta non c'è.

Perché l'IDF e non la copertura piatta: misurato, la copertura come frazione **non
discrimina** (le 2 infedeltà stavano a 0,86 e 1,00, sopra 9 risposte corrette) perché diluiva
«premiali» in 6 termini banali (``fondo``, ``nazionale``, ``2020``).

Discriminazione misurata sui 33 item, verità di riferimento dichiarata a mano
(«la risposta è nei passaggi recuperati?»), soglia 5,5:

    catturati 11/13 · falsi allarmi 0 · precisione 1,00 · richiamo 0,85

Cattura tutti i 7 out-of-corpus, i near-miss (``bd-02`` 1985, ``bd-06`` dipendenti ANAS,
``bd-04`` km realizzati) e — quello che conta — ``ic-07-bis``.

**Limite noto e strutturale:** non cattura le domande di **aggregazione** (``bd-03``, «quante
delibere nel 2024?», segnale 0,00). Lì tutti i termini *sono* nei passaggi: manca la vista
d'insieme, non un termine. Contare 93 delibere guardando 5 chunk su 13.670 è un disallineamento
di architettura, e va risolto con una query strutturata sui metadati, non con questo guardiano.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

# Termini funzionali e verbi/nomi di interrogazione: non portano contenuto discriminante.
# `qual` è in lista per misura: senza, produceva 3 falsi allarmi da solo.
STOPWORDS = frozenset(
    """
    che chi cosa come quale quali qual quanto quanta quanti quante quando dove perche perché
    con per del dello della delle dei degli dal dallo dalla dalle dai nel nello nella nelle nei
    negli sul sullo sulla sulle sui allo alla alle agli col coi
    una uno gli non sono essere stato stata stati state hanno aveva avere viene
    vengono deve devono sia suo sua loro questo questa questi queste tutto tutta tutti tutte
    ancora anche solo dopo prima ogni senza presso circa riguardo merito base sede caso casi
    modo parte seguito fine ammontano ammonta prevede prevedono differenza indicazione
    """.split()
)

_TOKEN = re.compile(r"[a-zà-ù0-9]+", re.IGNORECASE)
# Prefisso usato come radice: assorbe la morfologia italiana (premiali/premiale,
# ferroviaria/ferroviario) senza portarsi dietro uno stemmer.
_RADICE = 6
_MIN_LEN = 4
# Sillabazione da estrazione PDF: «intel- ligenza» va riunito prima di cercare i termini.
# Misurato: era la causa degli unici 2 falsi allarmi del guardiano.
_SILLABAZIONE = re.compile(r"(?<=[a-zà-ù])-\s+(?=[a-zà-ù])")


def _normalizza(testo: str) -> str:
    return testo.lower().replace("’", "'")


def _radici(testo: str, *, desillaba: bool = False) -> set[str]:
    t = _normalizza(testo)
    if desillaba:
        t = _SILLABAZIONE.sub("", t)
    # Le elisioni si spezzano sull'apostrofo: «dell'opera» → «opera».
    return {w[:_RADICE] for pezzo in t.split("'") for w in _TOKEN.findall(pezzo) if len(w) >= _MIN_LEN}


def content_terms(query: str) -> list[str]:
    """Termini di contenuto della domanda, ordinati, senza duplicati."""
    out = set()
    for pezzo in _normalizza(query).split("'"):
        for t in _TOKEN.findall(pezzo):
            if len(t) >= _MIN_LEN and t not in STOPWORDS:
                out.add(t)
    return sorted(out)


class TermStats:
    """Document frequency dei termini sui chunk del corpus, per pesare i mancanti via IDF.

    Persistita accanto agli indici (rigenerabile): calcolarla a ogni avvio costerebbe una
    passata su tutti i chunk, inaccettabile per un singolo ``ask``.
    """

    def __init__(self, n_chunks: int, df: dict[str, int]):
        self.n_chunks = max(1, n_chunks)
        self.df = df

    def idf(self, term: str) -> float:
        return math.log(self.n_chunks / (1 + self.df.get(term[:_RADICE], 0)))

    # --- costruzione / persistenza ---

    @classmethod
    def from_documents(cls, documents: list[str]) -> TermStats:
        df: Counter[str] = Counter()
        for doc in documents:
            df.update(_radici(doc, desillaba=True))
        return cls(len(documents), dict(df))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"n_chunks": self.n_chunks, "df": self.df}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> TermStats | None:
        p = Path(path)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return cls(int(d["n_chunks"]), dict(d["df"]))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None  # file corrotto: meglio ricalcolare che fidarsi


def missing_terms(query: str, passages: list[str]) -> list[str]:
    """Termini di contenuto della domanda assenti da **tutti** i passaggi recuperati."""
    presenti = _radici(" ".join(passages), desillaba=True)
    return [t for t in content_terms(query) if t[:_RADICE] not in presenti]


def abstention_signal(query: str, passages: list[str], stats: TermStats | None) -> tuple[float, list[str]]:
    """(segnale, termini mancanti). Segnale = IDF del mancante più raro; 0 se non ne mancano.

    Senza ``stats`` il segnale è 0: si preferisce non astenersi che astenersi su un peso
    inventato — il guardiano è una rete di sicurezza, non deve diventare la causa dei guasti.
    """
    mancanti = missing_terms(query, passages)
    if not mancanti or stats is None:
        return 0.0, mancanti
    return max(stats.idf(t) for t in mancanti), mancanti
