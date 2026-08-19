"""Riferimenti normativi nominati in un testo.

Serve a una cosa sola, ed è quella che chi legge chiede: **dire qual è la norma**. Quando
la risposta poggia su una frase che ricorre in centinaia di delibere, la fonte vera non è
la delibera che il retrieval ha pescato — è la legge che tutte richiamano. Il corpus non
la contiene (sono delibere, non leggi), ma la **nomina**, e un nome verificabile vale più
di un rimando a uno dei 289 posti in cui compare.

Il riconoscimento è deliberatamente conservativo: si estrae solo ciò che ha la forma piena
«tipo + data + n. numero». Le forme abbreviate di richiamo interno — «decreto legislativo
n. 163», 185 occorrenze nel corpus — rimandano a una norma introdotta per esteso altrove
nello stesso documento, e risolverle richiederebbe di indovinare quale. Nominare la norma
sbagliata sarebbe peggio che non nominarne nessuna: qui il punto è la verificabilità.
"""

from __future__ import annotations

import re

_TIPI = (
    r"legge costituzionale|legge|decreto[-\s]legge|decreto legislativo|"
    r"decreto del Presidente della Repubblica|decreto del Presidente del Consiglio dei ministri"
)
_MESI = (
    "gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre"
)
_RE_NORMA = re.compile(
    rf"\b({_TIPI})\s+(\d{{1,2}})\s+({_MESI})\s+(\d{{4}}),?\s*n\.?\s*(\d+)",
    re.IGNORECASE,
)


def estrai_norme(testo: str) -> list[str]:
    """Etichette canoniche delle norme citate per esteso, in ordine di comparsa e senza ripetizioni.

    Forma canonica: «legge 27 febbraio 1967, n. 48» — tipo in minuscolo (è così che si cita
    in italiano corrente), data e numero come nel testo.
    """
    viste: dict[str, None] = {}
    for m in _RE_NORMA.finditer(testo or ""):
        tipo = " ".join(m.group(1).split()).lower().replace("decreto legge", "decreto-legge")
        etichetta = f"{tipo} {int(m.group(2))} {m.group(3).lower()} {m.group(4)}, n. {m.group(5)}"
        viste.setdefault(etichetta, None)
    return list(viste)
