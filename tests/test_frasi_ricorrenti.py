"""Frasi ricorrenti e riferimenti normativi: il materiale della nota di provenienza.

Le due trappole che questi test bloccano sono entrambe emerse misurando sul corpus vero,
non ragionando: il taglio della frase sulle abbreviazioni («…, n.» come frase a sé, 216
volte) e la sillabazione della Gazzetta («Attri- buzioni»), che rendeva diverse fra loro
frasi identiche e teneva nascosto tutto il fenomeno.
"""

from talkdocs_rag_core.ingest.frasi import (
    IndiceFrasi,
    costruisci_indice,
    dividi_in_frasi,
    normalizza,
    ricongiungi_sillabazione,
)
from talkdocs_rag_core.rag.norme import estrai_norme

PREMESSA = (
    "VISTA la legge 27 febbraio 1967, n. 48, recante «Attribuzioni e ordinamento del "
    "Ministero del bilancio e della programmazione economica e istituzione del CIPE»; "
    "VISTO il decreto-legge 14 ottobre 2019, n. 111, convertito dalla legge 12 dicembre 2019, n. 141."
)


def test_il_punto_di_una_abbreviazione_non_chiude_la_frase():
    frasi = dividi_in_frasi(PREMESSA)
    assert any("n. 48" in f and "istituzione del CIPE" in f for f in frasi), frasi
    # Il difetto che si evita: «VISTA la legge 27 febbraio 1967, n.» come frase a sé.
    assert not any(f.rstrip().endswith("n.") for f in frasi)


def test_la_sillabazione_della_gazzetta_non_fa_due_frasi_di_una():
    spezzata = "VISTA la legge 27 febbraio 1967, n. 48, recan- te «Attri- buzioni e ordinamento»;"
    intera = "VISTA la legge 27 febbraio 1967, n. 48, recante «Attribuzioni e ordinamento»;"
    assert ricongiungi_sillabazione(spezzata).count("recante") == 1
    assert normalizza(spezzata) == normalizza(intera)


def test_i_numeri_sono_mascherati_ma_le_parole_no():
    """Due premesse identiche con importi diversi sono la stessa frase ricorrente; due
    frasi che differiscono in una parola non lo sono. Il rischio da evitare è il secondo:
    far coincidere due importi diversi renderebbe «boilerplate» un numero, che non lo è mai."""
    assert normalizza("assegna 1.000 euro al Fondo") == normalizza("assegna 2.500 euro al Fondo")
    assert normalizza("assegna 1.000 euro al Fondo") != normalizza("revoca 1.000 euro al Fondo")


def test_conta_i_documenti_non_le_occorrenze():
    """Una frase ripetuta tre volte nella stessa delibera non è boilerplate del corpus: è
    la struttura di quel documento. La soglia si applica ai documenti."""
    doc_a = PREMESSA + " " + PREMESSA
    idx = costruisci_indice([("a.txt", doc_a), ("b.txt", PREMESSA)], soglia=2)
    ricorrenti = list(idx.frasi.values())
    assert ricorrenti, "nessuna frase ricorrente riconosciuta"
    prima = max(ricorrenti, key=lambda f: f.n_documenti)
    assert prima.n_documenti == 2
    assert prima.n_occorrenze == 3

    # A soglia 3 non resta niente: i documenti sono due.
    assert costruisci_indice([("a.txt", doc_a), ("b.txt", PREMESSA)], soglia=3).frasi == {}


def test_la_norma_si_estrae_solo_in_forma_piena():
    norme = estrai_norme(PREMESSA)
    assert "legge 27 febbraio 1967, n. 48" in norme
    assert "decreto-legge 14 ottobre 2019, n. 111" in norme
    assert "legge 12 dicembre 2019, n. 141" in norme
    # I richiami interni senza data («decreto legislativo n. 163», 185 occorrenze nel
    # corpus) rimandano a una norma introdotta altrove: nominarne una sbagliata sarebbe
    # peggio che non nominarne nessuna.
    assert estrai_norme("ai sensi del decreto legislativo n. 163 si provvede") == []


def test_indice_assente_non_e_un_errore(tmp_path):
    """La provenienza è un di più dichiarativo: un corpus indicizzato prima di questo
    incremento deve continuare a rispondere, senza nota."""
    vuoto = IndiceFrasi.carica(tmp_path / "non-esiste.json")
    assert vuoto.frasi == {} and vuoto.in_passaggio(PREMESSA) == []


def test_indice_roundtrip(tmp_path):
    idx = costruisci_indice([("a.txt", PREMESSA), ("b.txt", PREMESSA)], soglia=2)
    p = idx.salva(tmp_path / "frasi.json")
    riletto = IndiceFrasi.carica(p)
    assert riletto.n_documenti_corpus == 2 and riletto.soglia == 2
    assert set(riletto.frasi) == set(idx.frasi)
    trovate = riletto.in_passaggio(PREMESSA)
    assert trovate and trovate[0].n_documenti == 2
    assert "legge 27 febbraio 1967, n. 48" in trovate[0].norme
