"""Rilegge il modulo di giudizio compilato e calcola il tasso di **fedeltà**.

Chiude il giro che le metriche automatiche non possono chiudere: `citation_validity`
verifica che i marcatori `[n]` siano in range, non che il passaggio citato sostenga
l'affermazione. La fedeltà è l'unica metrica che separa una risposta corretta da una
risposta sicura e falsa, ed è quella su cui si decide se promuovere il prototipo.

Uso:
    uv run python -m app spot-check  --run <run_id> --condition off   # genera il modulo
    # ... si compila il CSV a mano ...
    uv run python -m app eval-human  --run <run_id> --condition off   # rilegge e calcola
"""

from __future__ import annotations

import csv
from pathlib import Path

VERO = {"sì", "si", "s", "1", "true", "vero", "y", "yes", "ok"}
FALSO = {"no", "n", "0", "false", "falso"}

# Causa di un'infedeltà. `retrieval` = la risposta non era in nessuno dei passaggi (difetto a
# monte: reranker, top_k, arm keyword). `generazione` = c'era e il modello l'ha mancata.
# Determina dove investire, e sono rimedi completamente diversi.
CAUSE = {
    "retrieval": {"retrieval", "retr", "r", "recupero"},
    "generazione": {"generazione", "gen", "g", "modello"},
    # Terza causa, emersa dal giudizio su `bd-03` («quante delibere nel 2024?» → «almeno due»,
    # quando nel corpus ce ne sono 93): la domanda richiede una vista d'insieme che un RAG su
    # k chunk non può avere. Non è né retrieval né modello: è disallineamento di architettura,
    # e il rimedio è una query strutturata sui metadati.
    "strutturale": {"strutturale", "struttura", "s", "aggregazione", "architettura"},
}

# Bande di decisione sul tasso di fedeltà (soglie incluse nel limite inferiore).
BANDA_ALTA = 0.95
BANDA_BASSA = 0.85


def parse_bool(raw: str | None) -> bool | None:
    """``'sì'`` → True, ``'no'`` → False, vuoto/ignoto → None (non giudicato).

    Tollerante di proposito: il file lo compila una persona a mano, in LibreOffice, e
    ``Sì``/``SI``/``x`` non devono costare una rilettura.
    """
    if raw is None:
        return None
    v = raw.strip().lower()
    if v in VERO:
        return True
    if v in FALSO:
        return False
    return None


def parse_causa(raw: str | None) -> str | None:
    """``'retrieval'``/``'r'`` → ``'retrieval'``; ``'gen'`` → ``'generazione'``; altro → None."""
    if raw is None:
        return None
    v = raw.strip().lower()
    for canonica, sinonimi in CAUSE.items():
        if v in sinonimi:
            return canonica
    return None


def parse_voto(raw: str | None) -> int | None:
    """Voto 1-5. Fuori scala o non numerico → None: meglio non contarlo che falsarlo."""
    if raw is None:
        return None
    v = raw.strip().replace(",", ".")
    if not v:
        return None
    try:
        n = int(round(float(v)))
    except ValueError:
        return None
    return n if 1 <= n <= 5 else None


def banda(fedelta: float | None) -> tuple[str, str]:
    """(etichetta, raccomandazione) dal tasso di fedeltà."""
    if fedelta is None:
        return ("INSUFFICIENTE", "Nessun giudizio compilato: non si può decidere.")
    if fedelta >= BANDA_ALTA:
        return ("ALTA", "Promuovi. L'architettura attuale basta.")
    if fedelta >= BANDA_BASSA:
        return (
            "MEDIA",
            "Promuovi, ma l'applicazione deve avere uno strato di verifica "
            "affermazione↔passaggio: è ciò che separa una risposta corretta da una sicura e falsa.",
        )
    return ("BASSA", "Non promuovere su questo modello. Rimisura su un modello più grande prima di decidere.")


def aggregate(rows: list[dict]) -> dict:
    """Metriche di fedeltà dal modulo compilato. Funzione pura: nessun I/O."""
    n_totali = len(rows)
    fedeli, infedeli, cit_ok, cit_no, voti = 0, 0, 0, 0, []
    giudicati = 0
    # Quanti giudizi sono stati RIPORTATI da una run precedente invece che riletti adesso.
    # Un 92% con trenta giudizi su trentaquattro ereditati non è la stessa affermazione di
    # un 92% appena misurato: la prima dice che il sistema non è cambiato dove era già
    # stato verificato, la seconda che qualcuno ha letto tutto. Senza questo numero le due
    # si confondono, ed è esattamente il tipo di confusione che questo banco esiste per
    # evitare.
    ereditati = 0
    infedeli_ids: list[str] = []
    cause = {"retrieval": 0, "generazione": 0, "strutturale": 0}
    infedeli_senza_causa: list[str] = []

    for r in rows:
        f = parse_bool(r.get("fedele"))
        c = parse_bool(r.get("citazione_corretta"))
        v = parse_voto(r.get("italiano_1_5"))
        if f is None and c is None and v is None:
            continue
        giudicati += 1
        if (r.get("ereditato_da") or "").strip():
            ereditati += 1
        etichetta = r.get("id") or r.get("domanda", "")[:40]
        if f is True:
            fedeli += 1
        elif f is False:
            infedeli += 1
            infedeli_ids.append(etichetta)
            causa = parse_causa(r.get("causa"))
            if causa is None:
                infedeli_senza_causa.append(etichetta)
            else:
                cause[causa] += 1
        if c is True:
            cit_ok += 1
        elif c is False:
            cit_no += 1
        if v is not None:
            voti.append(v)

    giudicati_fedelta = fedeli + infedeli
    fedelta = (fedeli / giudicati_fedelta) if giudicati_fedelta else None
    giudicati_cit = cit_ok + cit_no
    return {
        "n_totali": n_totali,
        "n_giudicati": giudicati,
        "n_ereditati": ereditati,
        "n_riletti": giudicati - ereditati,
        "copertura": round(giudicati / n_totali, 3) if n_totali else 0.0,
        "n_fedeli": fedeli,
        "n_infedeli": infedeli,
        "infedeli_ids": infedeli_ids,
        "cause": cause,
        "infedeli_senza_causa": infedeli_senza_causa,
        "fedelta": round(fedelta, 4) if fedelta is not None else None,
        "citazione_corretta": round(cit_ok / giudicati_cit, 4) if giudicati_cit else None,
        "n_citazioni_sbagliate": cit_no,
        "italiano_medio": round(sum(voti) / len(voti), 2) if voti else None,
        "n_voti_italiano": len(voti),
    }


def build_report(run_id: str, condition: str | None, agg: dict) -> str:
    etichetta, raccomandazione = banda(agg["fedelta"])
    n = agg["n_fedeli"] + agg["n_infedeli"]
    passo = (1 / n * 100) if n else 0.0

    def pct(x):
        return "—" if x is None else f"{x * 100:.1f}%"

    lines = [
        f"# Fedeltà — giudizio umano su `{run_id}`",
        "",
        f"- condizione: **{condition or 'tutte'}**",
        f"- item da giudicare: **{agg['n_totali']}**  ·  giudicati: **{agg['n_giudicati']}**  "
        f"(copertura {pct(agg['copertura'])})",
        (
            f"- di cui **riletti in questa tornata: {agg['n_riletti']}**, ereditati da run "
            f"precedenti su risposta identica: {agg['n_ereditati']}"
            if agg["n_ereditati"]
            else "- tutti i giudizi sono stati dati su questa run"
        ),
        "",
        "| Metrica | Valore |",
        "|---|---|",
        f"| **Fedeltà** (affermazione sostenuta dal passaggio citato) | **{pct(agg['fedelta'])}** "
        f"({agg['n_fedeli']}/{n}) |",
        f"| Citazione corretta (rimando al documento giusto) | {pct(agg['citazione_corretta'])} |",
        f"| Italiano (media 1-5, su {agg['n_voti_italiano']} voti) | {agg['italiano_medio'] or '—'} |",
        f"| Risposte infedeli | {agg['n_infedeli']} |",
        f"| Citazioni al documento sbagliato | {agg['n_citazioni_sbagliate']} |",
        "",
        f"## Verdetto: fascia **{etichetta}**",
        "",
        raccomandazione,
        "",
    ]

    if agg["infedeli_ids"]:
        lines += ["**Item giudicati infedeli** (da leggere prima di decidere):", ""]
        lines += [f"- `{i}`" for i in agg["infedeli_ids"]]
        lines += [""]

    if agg["n_infedeli"]:
        r, g = agg["cause"]["retrieval"], agg["cause"]["generazione"]
        s = agg["cause"]["strutturale"]
        lines += [
            "## Dove intervenire",
            "",
            "| Causa dell'infedeltà | Item | Cosa significa |",
            "|---|---|---|",
            f"| `retrieval` | {r} | La risposta non era in **nessuno** dei passaggi: il modello "
            "non poteva saperla. Rimedio a monte — reranker, `top_k` più alto, arm keyword. |",
            f"| `generazione` | {g} | La risposta **c'era** e il modello ha estratto o attribuito "
            "male. Rimedio sul modello, o verifica affermazione↔passaggio. |",
            f"| `strutturale` | {s} | La domanda richiede una vista d'insieme (contare, sommare, "
            "elencare) che un RAG su k chunk non ha. Nessun modello e nessun reranker la risolve: "
            "serve una query sui metadati. |",
            "",
        ]
        if agg["infedeli_senza_causa"]:
            lines += [
                f"> ⚠️ {len(agg['infedeli_senza_causa'])} infedeltà senza `causa` compilata "
                f"({', '.join('`' + i + '`' for i in agg['infedeli_senza_causa'])}): senza quella "
                "colonna non si sa se investire sul retrieval o sul modello.",
                "",
            ]
        if agg["n_infedeli"] < 4:
            lines += [
                f"> ⚠️ Solo {agg['n_infedeli']} infedeltà: la ripartizione fra le cause è **indicativa**, "
                "non una priorità di investimento. Serve un campione più ampio (o un eval set più "
                "grande) prima di spostare budget su una delle tre.",
                "",
            ]
        elif s and s >= max(r, g):
            lines += [
                "**Lettura:** la causa prevalente è **strutturale**. Queste domande non si "
                "risolvono migliorando il RAG: vanno riconosciute e instradate su una query "
                "strutturata (il manifest ha già anno, numero e titolo di ogni atto), oppure "
                "rifiutate esplicitamente. Prima di comprare un modello più grande, decidi questo.",
                "",
            ]
        elif r or g:
            prevalente = "retrieval" if r > g else ("generazione" if g > r else None)
            if prevalente == "retrieval":
                lines += [
                    "**Lettura:** il collo di bottiglia è il **retrieval**, non il modello. Cambiare "
                    "LLM non sposta questi casi: nessun modello risponde correttamente da passaggi "
                    "che non contengono la risposta. Investi prima sul recupero.",
                    "",
                ]
            elif prevalente == "generazione":
                lines += [
                    "**Lettura:** il retrieval consegna le prove giuste e il **modello** le usa male. "
                    "Qui ha senso misurare un modello più grande sullo stesso set, e mettere a "
                    "preventivo lo strato di verifica affermazione↔passaggio.",
                    "",
                ]
            else:
                lines += [
                    "**Lettura:** le due cause si bilanciano: servono entrambi gli interventi, e il "
                    "campione è troppo piccolo per dare una priorità.",
                    "",
                ]

    if n:
        lines += [
            "## Quanto è solido questo numero",
            "",
            f"Su **{n}** item giudicati, **un solo item** vale {passo:.1f} punti percentuali.",
            f"Le soglie delle fasce (85% / 95%) distano quindi ~{max(1, round(0.10 * n))} item: "
            "questo campione dice **in quale fascia** siamo, non un valore preciso.",
            "",
        ]
        if n < 20:
            lines += [
                f"> ⚠️ Solo {n} item giudicati: troppo pochi perché la fascia sia affidabile. "
                "Completa il modulo prima di decidere.",
                "",
            ]

    if agg["copertura"] < 1.0:
        lines += [
            f"> ⚠️ Modulo compilato al {pct(agg['copertura'])}: le righe vuote sono **escluse** dal "
            "calcolo, non contate come corrette.",
            "",
        ]

    return "\n".join(lines) + "\n"


def load_form(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"modulo di giudizio non trovato: {path}\n"
            f"Generalo prima con: uv run python -m app spot-check --run <run_id>"
        )
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main(run_id: str, condition: str | None = None, out_dir: str | None = None) -> int:
    base = Path(out_dir or "eval/reports")
    suffix = f"-{condition}" if condition else ""
    rows = load_form(base / f"{run_id}-giudizi{suffix}.csv")
    agg = aggregate(rows)
    report = build_report(run_id, condition, agg)

    dest = base / f"{run_id}-fedelta{suffix}.md"
    dest.write_text(report, encoding="utf-8")

    etichetta, _ = banda(agg["fedelta"])
    fed = "—" if agg["fedelta"] is None else f"{agg['fedelta'] * 100:.1f}%"
    print(f"[eval-human] giudicati {agg['n_giudicati']}/{agg['n_totali']}  ·  fedeltà {fed}  ·  fascia {etichetta}")
    print(f"[eval-human] report: {dest}")
    return 0
