"""C1 — titolazione dei documenti del corpus delibere CIPE/CIPESS.

Il nome file è l'unica fonte di metadati disponibile (il DB ``db-delibere`` non è una
dipendenza del PoC): ``E{YY}{NNNN}`` → numero e anno della delibera. Il titolo finisce
nei metadati del chunk e quindi **nelle citazioni**, per cui va costruito in modo
deterministico e verificabile: è materiale della metrica "correttezza citazioni".

Denominazione: il comitato è **CIPE** fino al 2020 e **CIPESS** dal 2021 (Comitato
interministeriale per la programmazione economica *e lo sviluppo sostenibile*).
Citare "CIPE n. 75/2021" sarebbe un errore di merito, non di forma.
"""

import pytest

from talkdocs_rag_core.ingest.parsers import metadati_delibera, titolo_delibera


@pytest.mark.parametrize(
    ("stem", "atteso"),
    [
        # Anni CIPE (fino al 2020 incluso).
        ("E670001", "Delibera CIPE n. 1/1967"),
        ("E920057", "Delibera CIPE n. 57/1992"),
        ("E190004", "Delibera CIPE n. 4/2019"),
        ("E200080", "Delibera CIPE n. 80/2020"),
        # Anni CIPESS (dal 2021).
        ("E210075", "Delibera CIPESS n. 75/2021"),
        ("E240093", "Delibera CIPESS n. 93/2024"),
        ("E260014", "Delibera CIPESS n. 14/2026"),
    ],
)
def test_titolo_da_codice_canonico(stem, atteso):
    assert titolo_delibera(stem) == atteso


def test_pivot_secolo_sul_1967():
    """L'archivio parte dal 1967: YY>=67 è Novecento, YY<67 è Duemila."""
    assert titolo_delibera("E660001") == "Delibera CIPESS n. 1/2066"  # improbabile, ma coerente
    assert titolo_delibera("E670001") == "Delibera CIPE n. 1/1967"
    assert titolo_delibera("E990001") == "Delibera CIPE n. 1/1999"
    assert titolo_delibera("E000001") == "Delibera CIPE n. 1/2000"


def test_codice_minuscolo():
    """Nell'archivio 20 file hanno il prefisso minuscolo."""
    assert titolo_delibera("e260014") == "Delibera CIPESS n. 14/2026"


def test_varianti_di_lavorazione_conservano_il_codice():
    """I file non canonici (bozze, post-CdC) restano titolabili dal prefisso."""
    assert titolo_delibera("E190004finale_1") == "Delibera CIPE n. 4/2019"
    assert titolo_delibera("E200067_Patuanelli") == "Delibera CIPE n. 67/2020"
    assert titolo_delibera("E210075-DEL-Delibera-75-ASPI-post-MEF-corretta_3") == "Delibera CIPESS n. 75/2021"


def test_numero_senza_zeri_di_riempimento():
    assert titolo_delibera("E240007") == "Delibera CIPESS n. 7/2024"
    assert titolo_delibera("E960301") == "Delibera CIPE n. 301/1996"


@pytest.mark.parametrize("stem", ["dati-bene-comune", "openpolis-monitoraggio", "E1900", "X190004", "190004"])
def test_nessun_codice_ritorna_none(stem):
    """Senza codice riconoscibile il parser deve ricadere sul nome file, non inventare."""
    assert titolo_delibera(stem) is None


def test_parse_file_usa_il_titolo_delibera(tmp_path):
    """Integrazione: un .txt di corpus con nome canonico riceve il titolo della delibera."""
    from talkdocs_rag_core.ingest.parsers import parse_file

    p = tmp_path / "E210075.txt"
    p.write_text("IL COMITATO INTERMINISTERIALE\n\nVISTA la legge...", encoding="utf-8")
    text, title = parse_file(p)
    assert title == "Delibera CIPESS n. 75/2021"
    assert "COMITATO INTERMINISTERIALE" in text


def test_parse_file_txt_generico_resta_sul_nome(tmp_path):
    from talkdocs_rag_core.ingest.parsers import parse_file

    p = tmp_path / "nota-di-metodo.txt"
    p.write_text("contenuto", encoding="utf-8")
    _, title = parse_file(p)
    assert title == "nota-di-metodo"


def test_metadati_delibera_campi_tipizzati():
    """Lo store strutturato ha bisogno dei campi, non del titolo composto."""
    assert metadati_delibera("E210075") == {
        "codice": "E210075",
        "numero": 75,
        "anno": 2021,
        "comitato": "CIPESS",
    }


def test_metadati_delibera_normalizza_il_codice_minuscolo():
    """Nell'archivio 20 file hanno il prefisso minuscolo: il codice esce sempre maiuscolo."""
    assert metadati_delibera("e190004finale_1")["codice"] == "E190004"


def test_metadati_delibera_none_se_non_e_un_codice():
    assert metadati_delibera("relazione-annuale") is None


def test_metadati_delibera_e_titolo_condividono_la_regola():
    """Se le due funzioni divergono, le citazioni e i conteggi raccontano cose diverse."""
    for stem in ("E200080", "E210001", "E670001", "E000001"):
        meta = metadati_delibera(stem)
        assert titolo_delibera(stem) == f"Delibera {meta['comitato']} n. {meta['numero']}/{meta['anno']}"
