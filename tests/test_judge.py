"""UI di giudizio: il contratto che conta è il modulo CSV.

La UI è un'interfaccia di scrittura su `<run>-giudizi-<cond>.csv`, non un formato nuovo:
`eval-human` deve continuare a rileggerlo senza sapere che esiste, e un modulo compilato
a mano in LibreOffice deve restare valido. Qui si verifica proprio questo — più la cosa
che non si può sbagliare due volte: **un giudizio già dato non si perde**.
"""

import json
import re
from pathlib import Path

import pytest

from app import judge
from scripts.spot_check import COLONNE_FORM


def _rec(query, ident_noto=True, **kw):
    base = {
        "query": query,
        "answer_text": "risposta",
        "refused": False,
        "uncertain": False,
        "from_cache": False,
        "cache_enabled": False,
        "claims": [{"statement": "afferma X", "passages": [1], "verbatim": "le parole esatte"}],
        "cited_chunk_ids": ["c1"],
        "retrieved_chunk_ids": ["c1", "c2"],
        "route": "pointwise",
        "router_source": "lexical",
        "verbatim": {
            "valid_ratio": 1.0,
            "per_claim": [
                {"statement": "afferma X", "passages": [1], "verbatim": "le parole esatte",
                 "esito": "valido", "matched_passage": 1}
            ],
        },
    }
    base.update(kw)
    return base


CHUNKS = {
    "c1": {"text": "Testo del passaggio con le parole esatte dentro.", "meta": {"title": "Delibera n. 1/2024"}},
    "c2": {"text": "Altro passaggio non citato.", "meta": {"title": "Delibera n. 2/2024"}},
}
IDX = {"domanda uno": ("ic-01", "in_corpus"), "domanda due": ("ic-02", "in_corpus")}


def test_item_porta_passaggi_marcati_e_esito_verbatim():
    it = judge.costruisci_item(_rec("domanda uno"), IDX, CHUNKS)
    assert it["id"] == "ic-01"
    assert [p["citato"] for p in it["passaggi"]] == [True, False]
    assert it["passaggi"][0]["testo"].startswith("Testo del passaggio")
    # L'esito verbatim viene dall'audit, non ricalcolato: la UI mostra ciò che la
    # pipeline HA deciso, non una seconda opinione.
    assert it["claims"][0]["esito"] == "valido"


def test_chunk_mancante_e_dichiarato_non_finto():
    it = judge.costruisci_item(_rec("domanda uno"), IDX, {})
    assert all(p["mancante"] for p in it["passaggi"])
    assert it["passaggi"][0]["testo"] == ""


def _form(tmp_path):
    return [
        {**{c: "" for c in COLONNE_FORM}, "run_id": "r1", "condition": "off", "id": "ic-01", "domanda": "domanda uno"},
        {**{c: "" for c in COLONNE_FORM}, "run_id": "r1", "condition": "off", "id": "ic-02", "domanda": "domanda due"},
    ]


def test_salvataggio_scrive_le_colonne_attese(tmp_path):
    p = tmp_path / "giudizi.csv"
    judge.salva_giudizio(p, _form(tmp_path), "ic-01", {"fedele": "SI", "italiano_1_5": "5"})
    righe = judge.leggi_modulo(p)
    assert list(righe[0]) == COLONNE_FORM      # `eval-human` legge per nome di colonna
    assert righe[0]["fedele"] == "SI" and righe[0]["italiano_1_5"] == "5"
    assert righe[1]["fedele"] == ""            # le altre righe restano intatte


def test_un_giudizio_gia_dato_non_si_perde(tmp_path):
    """Il caso che costa caro: si apre la UI su un modulo compilato a metà. Il lavoro
    umano è l'unica cosa non rigenerabile del banco."""
    p = tmp_path / "giudizi.csv"
    judge.salva_giudizio(p, _form(tmp_path), "ic-02", {"fedele": "NO", "causa": "retrieval", "note": "a mano"})
    judge.salva_giudizio(p, _form(tmp_path), "ic-01", {"fedele": "SI"})
    righe = {r["id"]: r for r in judge.leggi_modulo(p)}
    assert righe["ic-02"]["fedele"] == "NO" and righe["ic-02"]["note"] == "a mano"
    assert righe["ic-01"]["fedele"] == "SI"


def test_deselezionare_svuota_invece_di_lasciare_il_vecchio(tmp_path):
    p = tmp_path / "giudizi.csv"
    judge.salva_giudizio(p, _form(tmp_path), "ic-01", {"fedele": "SI"})
    judge.salva_giudizio(p, _form(tmp_path), "ic-01", {"fedele": ""})
    assert judge.leggi_modulo(p)[0]["fedele"] == ""


def test_riga_inesistente_e_un_errore_non_una_riga_nuova(tmp_path):
    """Silenziosamente aggiungere una riga produrrebbe un modulo con più giudizi che
    risposte, e un tasso di fedeltà calcolato su un denominatore inventato."""
    p = tmp_path / "giudizi.csv"
    with pytest.raises(KeyError):
        judge.salva_giudizio(p, _form(tmp_path), "non-esiste", {"fedele": "SI"})


def test_stato_modulo_permette_di_riprendere(tmp_path):
    p = tmp_path / "giudizi.csv"
    judge.salva_giudizio(p, _form(tmp_path), "ic-01", {"fedele": "SI"})
    stato = judge.stato_modulo(p, _form(tmp_path))
    assert stato["ic-01"]["fedele"] == "SI" and stato["ic-02"]["fedele"] == ""


def test_bundle_roundtrip(tmp_path):
    bundle = {"run_id": "r1", "condition": "off", "items": [], "form": []}
    path = judge.percorso_bundle("r1", "off", tmp_path)
    path.write_text(json.dumps(bundle), encoding="utf-8")
    assert judge.carica_bundle("r1", "off", tmp_path)["run_id"] == "r1"
    trovati = judge.bundle_disponibili(tmp_path)
    assert trovati == [{"run_id": "r1", "condition": "off", "path": str(path)}]


def test_le_run_di_singole_ask_non_finiscono_nel_menu(tmp_path):
    """In `logs/` si accumulano decine di `run-*` (una `ask` da CLI o dalla console) e
    `web-session`. Giudicarle non produce un tasso di fedeltà — il denominatore è l'eval
    set — e in un menu a tendina seppelliscono le run che si vogliono davvero giudicare."""
    for nome in ("eval-20260808T152700Z", "eval-20260807T181416Z", "run-20260807T162233Z", "web-session"):
        (tmp_path / f"{nome}.jsonl").write_text("{}\n", encoding="utf-8")
    assert judge.run_disponibili(tmp_path) == ["eval-20260808T152700Z", "eval-20260807T181416Z"]
    assert len(judge.run_disponibili(tmp_path, solo_eval=False)) == 4


def test_la_pagina_di_giudizio_e_una_sola():
    """La serve FastAPI in locale ed è la stessa che Vercel pubblica statica. Una copia in
    `app/static/` divergerebbe da quella in `web/public/` nel giro di poche modifiche, e
    la peggiore delle due sarebbe quella su cui si giudica.

    Il controllo NON importa `app.web`: quel modulo tira dentro FastAPI, e
    `test_vendor_import` verifica proprio che la suite non se lo trascini. Si legge il
    sorgente, che per questa invariante è sufficiente.
    """
    radice = Path(__file__).resolve().parent.parent
    assert (radice / "web" / "public" / "index.html").exists()
    assert not (radice / "app" / "static" / "judge.html").exists(), "copia superata della pagina"
    sorgente = (radice / "app" / "web.py").read_text(encoding="utf-8")
    assert '"web" / "public" / "index.html"' in sorgente


def test_le_colonne_del_csv_scritto_dal_browser_combaciano_con_quelle_di_python():
    """La pagina costruisce il CSV in JavaScript quando gira su Vercel: se le due liste
    divergono, `eval-human` legge colonne che non esistono e la fedeltà esce vuota — senza
    che nulla fallisca."""
    pagina = Path(__file__).resolve().parent.parent / "web" / "public" / "index.html"
    blocco = re.search(r"const COLONNE = \[(.*?)\];", pagina.read_text(encoding="utf-8"), re.DOTALL)
    assert blocco, "lista COLONNE non trovata nella pagina"
    assert re.findall(r'"([a-z_0-9]+)"', blocco.group(1)) == COLONNE_FORM
