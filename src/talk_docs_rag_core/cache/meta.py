"""Cache persistente delle risposte meta: la prosa generata una volta, poi congelata.

## Perché

Da quando la risposta meta è **generata** invece che concatenata dalla scheda, il modello
la riscrive a ogni run anche a temperatura 0. Misurato su due coppie di run consecutive:
`meta` 6/6 e 5/6 risposte cambiate, contro 1/7 del ramo `structured`, che è a template. Non
sono ritocchi — somiglianza 0,42 su `bd-05`, 0,72 su `meta-03`.

La conseguenza costa: l'ereditarietà dei giudizi umani è governata dall'impronta
`sha256(domanda + risposta)`, quindi **le sei meta-domande tornano da rileggere a ogni
run**, per sempre, senza che sia cambiato niente nel sistema.

## Perché si può cachare qui e non altrove

Il ramo meta **non dipende dal retrieval**: i suoi passaggi sono le sezioni della scheda e
un blocco di statistiche calcolate sul manifest. A parità di corpus, di scheda e di modello
la risposta *dovrebbe* essere la stessa — la variazione è rumore del decoder, non
informazione. Congelarla non nasconde niente: rende il ramo meta stabile come lo è il ramo
strutturato, con la differenza che la prosa è vera prosa e non un template.

Sul ramo puntuale la stessa cosa sarebbe inaccettabile: lì la risposta dipende da quali
cinque passaggi il retrieval ha scelto, e congelarla vorrebbe dire servire una risposta
costruita su un contesto che non si sta più guardando.

## Che cosa invalida la cache

La chiave è (corpus_version, impronta della scheda, modello, domanda normalizzata).

La **scheda entra nella chiave a parte**: vive in `corpus/delibere/card/` ma è esclusa
dall'ingest, quindi `corpus_version` **non la copre**. Senza la sua impronta, correggere la
scheda — che è esattamente il lavoro di setup di un corpus — lascerebbe in circolo risposte
meta che descrivono la scheda di ieri. Sarebbe il guasto peggiore possibile qui: il
componente nato per dichiarare che cosa il sistema sa fare, che dichiara il falso.

## Effetto sull'A/B del caching

La cache è attiva in **entrambi i bracci**, come il ramo strutturato: se rispettasse
`use_cache` il braccio OFF rigenererebbe, e siccome è quello che si giudica non risolverebbe
niente. Le meta escono quindi dal conteggio delle chiamate al modello in tutte e due le
condizioni, e il report lo dichiara — non è un risparmio della cache semantica e non va
letto come tale.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


def _normalizza(query: str) -> str:
    """Domande che differiscono per spazi o maiuscole sono la stessa domanda.

    Nient'altro: due formulazioni diverse della stessa meta-domanda **non** condividono la
    voce. Allargare qui significherebbe servire la risposta a una domanda che non è stata
    posta, che è il guasto che il router esiste per evitare.
    """
    return " ".join(re.sub(r"\s+", " ", (query or "").strip().lower()).split())


def impronta_scheda(card) -> str:
    testo = getattr(card, "text", "") or ""
    return hashlib.sha256(testo.encode("utf-8")).hexdigest()[:16]


@dataclass
class VoceMeta:
    answer_text: str
    claims: list[dict]
    cited_passages: list[int]
    cited_chunk_ids: list[str]
    raw_output: str
    # Solo per leggere il file a occhio e capire da dove viene una risposta.
    query: str
    model: str


class CacheMeta:
    """Voci su file JSON. Piccola per costruzione: una per meta-domanda vista."""

    def __init__(self, path: str | Path, corpus_version: str, card_hash: str, model: str):
        self.path = Path(path)
        self.corpus_version = corpus_version
        self.card_hash = card_hash
        self.model = model
        self._voci: dict[str, dict] = {}
        self._carica()

    def _carica(self) -> None:
        if not self.path.exists():
            return
        try:
            self._voci = json.loads(self.path.read_text(encoding="utf-8")).get("voci", {})
        except (json.JSONDecodeError, OSError) as exc:
            # Una cache illeggibile non deve impedire di rispondere: si riparte da vuota.
            logger.warning("cache meta illeggibile (%s): si riparte da vuota", exc.__class__.__name__)
            self._voci = {}

    def chiave(self, query: str) -> str:
        materiale = "|".join([self.corpus_version, self.card_hash, self.model, _normalizza(query)])
        return hashlib.sha256(materiale.encode("utf-8")).hexdigest()[:24]

    def leggi(self, query: str) -> VoceMeta | None:
        v = self._voci.get(self.chiave(query))
        if v is None:
            return None
        return VoceMeta(
            answer_text=v["answer_text"],
            claims=v.get("claims") or [],
            cited_passages=v.get("cited_passages") or [],
            cited_chunk_ids=v.get("cited_chunk_ids") or [],
            raw_output=v.get("raw_output") or "",
            query=v.get("query") or query,
            model=v.get("model") or self.model,
        )

    def scrivi(self, query: str, voce: VoceMeta) -> None:
        self._voci[self.chiave(query)] = {
            "query": voce.query,
            "model": voce.model,
            "corpus_version": self.corpus_version,
            "card_hash": self.card_hash,
            "answer_text": voce.answer_text,
            "claims": voce.claims,
            "cited_passages": voce.cited_passages,
            "cited_chunk_ids": voce.cited_chunk_ids,
            "raw_output": voce.raw_output,
        }
        self._salva()

    def _salva(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps({"voci": self._voci}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            # Non poter scrivere la cache è un peccato, non un errore: la risposta c'è.
            logger.warning("cache meta non scritta (%s)", exc)

    def pulisci_obsolete(self) -> int:
        """Toglie le voci di un altro corpus, un'altra scheda o un altro modello.

        Non serve alla correttezza — la chiave le rende irraggiungibili — ma un file che
        accumula tutte le risposte di tutte le configurazioni passate diventa illeggibile,
        e questo file esiste anche per essere **letto**: è ciò che si serve agli utenti.
        """
        vive = {
            k: v
            for k, v in self._voci.items()
            if v.get("corpus_version") == self.corpus_version
            and v.get("card_hash") == self.card_hash
            and v.get("model") == self.model
        }
        tolte = len(self._voci) - len(vive)
        if tolte:
            self._voci = vive
            self._salva()
        return tolte
