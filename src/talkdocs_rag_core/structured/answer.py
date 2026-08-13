"""Prosa della risposta calcolata. **Template, mai il modello.**

La pipeline ha già il numero: farlo riscrivere al modello aggiungerebbe soltanto una
superficie di riscrittura su un dato esatto, che è precisamente il guasto da cui nasce
questo incremento («almeno due» invece di 93).

Due elementi rendono la risposta difendibile invece che solo corretta: il **perimetro
dichiarato** (il conteggio è sul corpus indicizzato, non sull'archivio 1967-2026) e il
**controllo di completezza** (93 presenti, numerazione fino alla 95 → due mancano dal
corpus).

Il perimetro vale però solo se **nessuna frase lo sfonda**: il controllo di completezza
misura l'assenza dal corpus indicizzato, non l'inesistenza della delibera. Dire «non sono
presenti nell'archivio» sarebbe un'affermazione sul mondo che la query non sostiene, e
annullerebbe la dichiarazione fatta una frase prima.

Il perimetro si dichiara però **una volta sola**, in apertura, dove regge il numero: ripetere
«corpus indicizzato» a ogni frase non rafforza la cautela, la rende un tic e sminuisce proprio
la difendibilità che persegue. Le frasi successive vi si agganciano in forma breve («mancano
dal corpus»), che resta dentro il perimetro senza ridichiararlo.
"""

from __future__ import annotations

from talkdocs_rag_core.rag.outcomes import StructuredOutcome
from talkdocs_rag_core.structured import intents

# Vale per il conteggio, per l'elenco e per la distribuzione: sotto un elenco di titoli
# «il conteggio è calcolato» sarebbe la cosa sbagliata. Non ripete il perimetro, già
# dichiarato in apertura.
_NOTA = "Il risultato è calcolato sui metadati, non estratto dal testo dei documenti."

# Apertura di ogni risposta: il perimetro precede il numero invece di seguirlo. Posposto
# produceva il doppio «nel» di «del CIPESS nel 2024 nel corpus indicizzato».
_PERIMETRO = "Nel corpus indicizzato"


def _perimetro(params: dict) -> str:
    """Il filtro applicato, in italiano leggibile: **prima il comitato, poi il tempo**.

    L'ordine inverso produce «nel 2024 del CIPESS», che si legge come una svista e squalifica
    il numero che precede."""
    pezzi = []
    if params.get("comitato"):
        pezzi.append(f"del {params['comitato']}")
    if params.get("anno") is not None:
        pezzi.append(f"nel {params['anno']}")
    if params.get("anno_da") is not None and params.get("anno_a") is not None:
        pezzi.append(f"dal {params['anno_da']} al {params['anno_a']}")
    return " ".join(pezzi)


def _delibere(n: int) -> str:
    """«1 delibera» / «N delibere». L'accordo rotto costa credibilità al numero esatto."""
    return "1 delibera" if n == 1 else f"{n} delibere"


def _riga(r: dict) -> str:
    return f"- {r['title']} (`{r['path']}`)"


def _apertura(testa: str, perimetro: str) -> str:
    """Attacca il perimetro alla testa della frase e chiude gli spazi pendenti.

    Senza filtri `perimetro` è vuoto e l'interpolazione lascerebbe uno spazio prima del
    punto o dei due punti: qui il perimetro sta in coda alla testa, non a metà frase, e
    `rstrip()` basta (prima serviva sostituire il doppio spazio interno)."""
    return f"{testa} {perimetro}".rstrip()


def componi(
    *, intent: str, params: dict, rows: list[dict], sql: str, sql_params: list, max_rows: int
) -> tuple[str, StructuredOutcome]:
    """``(testo, esito)``. Il testo è ciò che vede l'utente; l'esito è ciò che va in audit."""
    perimetro = _perimetro(params)

    if intent == intents.COUNT_DELIBERE:
        r = rows[0] if rows else {"n": 0, "min_numero": None, "max_numero": None}
        n = int(r.get("n") or 0)
        completeness: dict = {}
        # `MAX(numero)` è confrontabile con il conteggio solo su un singolo anno: la
        # numerazione delle delibere riparte ogni anno. Con zero righe `MIN`/`MAX` sono
        # entrambi NULL e il controllo non si applica.
        if params.get("anno") is not None and r.get("min_numero") is not None and r.get("max_numero") is not None:
            minimo, massimo = int(r["min_numero"]), int(r["max_numero"])
            # L'ampiezza dell'intervallo osservato è `max - min + 1`, **misurata** e non
            # assunta: dare per scontata la base 1 sarebbe un'ipotesi sulla numerazione, non
            # un dato. Sui dati reali (che partono da 1) il risultato coincide.
            gap = max(0, massimo - minimo + 1 - n)
            completeness = {"count": n, "max_numero": massimo, "gap": gap}
        if n == 0:
            apertura = _apertura(f"{_PERIMETRO} non risulta nessuna delibera", perimetro)
            testo = f"{apertura}. {_NOTA}"
        else:
            verbo = "risulta" if n == 1 else "risultano"
            apertura = _apertura(f"{_PERIMETRO} {verbo} **{_delibere(n)}**", perimetro)
            testo = f"{apertura}."
            if completeness.get("gap"):
                gap = completeness["gap"]
                # Il fatto misurato è l'assenza dal corpus, non l'inesistenza: «mancano dal
                # corpus» resta dentro il perimetro dichiarato in apertura senza ridichiararlo,
                # e senza dire *perché* mancano (mai pubblicate, saltate dall'ingest, ritirate:
                # la query non distingue).
                verbo_gap = "manca" if gap == 1 else "mancano"
                testo += (
                    f" La numerazione dell'anno arriva alla n. {completeness['max_numero']}: "
                    f"{_delibere(gap)} {verbo_gap} dal corpus."
                )
            testo += f" {_NOTA}"
        return testo, StructuredOutcome(
            intent=intent,
            sql=sql,
            params=sql_params,
            rows=rows,
            n_rows=len(rows),
            computed_value=n,
            completeness=completeness,
            cited_doc_ids=[],
        )

    if intent == intents.LIST_DELIBERE:
        n = len(rows)
        visibili = rows[:max_rows]
        elenco = "\n".join(_riga(r) for r in visibili)
        coda = ""
        if n > len(visibili):
            # «Nella tupla di audit» era gergo interno: chi legge non sa cosa sia né come
            # arrivarci. L'informazione che gli serve è che le righe non mostrate esistono e
            # non sono andate perse.
            coda = (
                f"\n\n(…altre {n - len(visibili)} non mostrate qui; "
                "l'elenco completo è registrato insieme alla risposta.)"
            )
        if n == 0:
            # Senza righe i due punti resterebbero pendenti su un elenco vuoto: qui la frase
            # si chiude, come nel ramo del conteggio.
            apertura = _apertura(f"{_PERIMETRO} non risulta nessuna delibera", perimetro)
            testo = f"{apertura}. {_NOTA}"
        else:
            verbo = "risulta" if n == 1 else "risultano"
            apertura = _apertura(f"{_PERIMETRO} {verbo} **{_delibere(n)}**", perimetro)
            testo = f"{apertura}:\n\n{elenco}{coda}\n\n{_NOTA}"
        return testo, StructuredOutcome(
            intent=intent,
            sql=sql,
            params=sql_params,
            rows=rows,
            n_rows=n,
            computed_value=n,
            completeness={},
            cited_doc_ids=[r["path"] for r in rows],
        )

    if intent == intents.COUNT_BY_YEAR:
        righe = "\n".join(f"- {r['anno']}: {r['n']}" for r in rows)
        apertura = _apertura(f"{_PERIMETRO} le delibere", perimetro)
        testo = f"{apertura} si distribuiscono per anno:\n\n{righe}\n\n{_NOTA}"
        return testo, StructuredOutcome(
            intent=intent,
            sql=sql,
            params=sql_params,
            rows=rows,
            n_rows=len(rows),
            computed_value=None,
            completeness={},
            cited_doc_ids=[],
        )

    # La distribuzione per anno era il ramo di fall-through: ci finiva dentro *qualunque*
    # intento non riconosciuto, in silenzio, producendo la prosa sbagliata su righe che non
    # hanno le colonne che quel ramo legge. `structured.service` non può arrivare qui —
    # `intents.build` filtra già sull'insieme chiuso `TUTTI` e degrada a rifiuto — ma un
    # intento aggiunto a `TUTTI` e dimenticato qui deve fare rumore, non prosa plausibile.
    raise ValueError(f"intento non gestito da componi(): {intent!r}. Attesi: {', '.join(intents.TUTTI)}.")
