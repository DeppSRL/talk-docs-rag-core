"""Scheda del corpus: il contesto semantico scritto a mano che accompagna l'indice.

È il prototipo, in forma di directory di file di testo, del «setup del corpus» che in
talk-docs diventerà UI di back office: chi conosce il corpus scrive che cosa è, come
sono fatti i documenti e che cosa il sistema sa calcolarci sopra. La scheda alimenta
due consumatori: la risposta alle **meta-domande** («di cosa parla questo corpus?») e
il prompt del **router agentico**.

Regola non negoziabile, fatta rispettare a monte (README della scheda) e non qui: la
scheda porta contesto, **mai numeri del corpus** — quelli li calcola lo
``StructuredStore``, con la query in audit. Una scheda con «511 delibere» scritte a
mano diverge dal corpus indicizzato alla prima run di ingest e mente proprio nel
componente nato per dichiarare i limiti.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CorpusCard:
    """Sezioni della scheda, in ordine di nome file. Immutabile come ``RagConfig``:
    si carica alla costruzione della pipeline, non si muta a run in corso."""

    sections: tuple[tuple[str, str], ...]  # (stem del file, testo)

    @property
    def text(self) -> str:
        """La scheda intera, concatenata: è ciò che entra nel prompt del router
        agentico e nella risposta meta."""
        return "\n\n".join(testo for _, testo in self.sections)

    @classmethod
    def load(cls, card_dir: str | Path) -> CorpusCard | None:
        """``None`` se la directory non esiste o non contiene sezioni: la pipeline
        deve degradare (niente route meta dalla scheda), non esplodere — stesso patto
        di ``StructuredStore.from_path``.

        Entrano solo i file ``NN-*.md`` (prefisso numerico): l'ordine delle sezioni è
        l'ordinamento dei nomi, e il README della directory — che documenta le regole
        di compilazione, non il corpus — resta fuori per costruzione.
        """
        p = Path(card_dir)
        if not p.is_dir():
            return None
        sezioni = []
        for f in sorted(p.glob("*.md")):
            if not f.name[:1].isdigit():
                continue
            testo = f.read_text(encoding="utf-8").strip()
            if testo:
                sezioni.append((f.stem, testo))
        if not sezioni:
            return None
        return cls(sections=tuple(sezioni))
