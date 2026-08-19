"""Semantic layer: insieme **chiuso** di intenti tipizzati.

Niente text-to-SQL. La letteratura dà 0,85–1,00 a un semantic layer con rifiuto esplicito
contro 0,20–0,50 al SQL generato su schemi reali; su quattro colonne la questione non si
pone nemmeno. Il modello non entra in questo ramo: qui non c'è nulla da generare.

Nessun testo dell'utente entra nella *stringa* SQL — i nomi di colonna vengono da questo
file, i valori passano come parametri.
"""

from __future__ import annotations

COUNT_DELIBERE = "count_delibere"
LIST_DELIBERE = "list_delibere"
COUNT_BY_YEAR = "count_by_year"

TUTTI = (COUNT_DELIBERE, LIST_DELIBERE, COUNT_BY_YEAR)

# Limite di guardia sulle righe restituite dallo store (l'audit le porta tutte, la vista
# le tronca a `structured_max_rows`). 1000 > dimensione del corpus: non taglia nulla oggi.
_LIMIT_DEFAULT = 1000


def _where(params: dict) -> tuple[str, list]:
    """Clausola WHERE dai soli filtri riconosciuti. `is_delibera` è sempre presente:
    un file del corpus senza codice non è una delibera e non entra in un conteggio di
    delibere.

    `anno` e `anno_da`/`anno_a` sono mutuamente esclusivi **per contratto del chiamante**
    (`rag.router._filtri`, rami disgiunti): qui non è verificato."""
    cond = ["is_delibera"]
    args: list = []
    if params.get("anno") is not None:
        cond.append("anno = ?")
        args.append(int(params["anno"]))
    if params.get("anno_da") is not None:
        cond.append("anno >= ?")
        args.append(int(params["anno_da"]))
    if params.get("anno_a") is not None:
        cond.append("anno <= ?")
        args.append(int(params["anno_a"]))
    if params.get("comitato"):
        cond.append("comitato = ?")
        args.append(str(params["comitato"]))
    return " AND ".join(cond), args


def build(intent: str, params: dict) -> tuple[str, list] | None:
    """``(sql, params)`` oppure ``None`` se l'intento non è coperto.

    ``None`` fa degradare la route a rifiuto dichiarato: meglio dire «non so contare
    questo» che eseguire una query su un filtro indovinato.
    """
    if intent not in TUTTI:
        return None
    where, args = _where(params)
    if intent == COUNT_DELIBERE:
        return (
            f"SELECT COUNT(*) AS n, MIN(numero) AS min_numero, MAX(numero) AS max_numero "
            f"FROM documenti WHERE {where}",
            args,
        )
    if intent == LIST_DELIBERE:
        limit = int(params.get("limit") or _LIMIT_DEFAULT)
        return (
            f"SELECT anno, numero, comitato, title, path FROM documenti WHERE {where} "
            f"ORDER BY anno, numero LIMIT ?",
            [*args, limit],
        )
    return (
        f"SELECT anno, COUNT(*) AS n FROM documenti WHERE {where} GROUP BY anno ORDER BY anno",
        args,
    )
