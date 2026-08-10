"""La nota di provenienza: dichiarare che la fonte citata è una fra molte equivalenti.

Ogni test qui blocca una decisione presa **misurando sugli item che il giudizio umano
aveva segnalato**, non ragionando a tavolino. In particolare le due che decidono se il
meccanismo è utile o rumoroso: l'ancora è lo span del claim (non il passaggio), e uno span
che porta una cifra propria non è mai boilerplate.
"""

from ingest.frasi import costruisci_indice
from rag import provenienza as P

PREMESSA = (
    "VISTA la legge 27 febbraio 1967, n. 48, recante «Attribuzioni e ordinamento del Ministero "
    "del bilancio e della programmazione economica e istituzione del Comitato dei ministri per "
    "la programmazione economica»; "
)
SPECIFICA = "Al Fondo sport e periferie sono riassegnati 7.526.144,01 euro dal programma Cantieri in Comune. "


def _indice(n_documenti: int = 30, soglia: int = 3):
    docs = [
        (f"d{i}.txt", PREMESSA + f"DELIBERA di assegnare {i * 1000} euro al programma. ")
        for i in range(n_documenti)
    ]
    docs.append(("specifica.txt", PREMESSA + SPECIFICA))
    return costruisci_indice(docs, soglia=soglia)


def _claim(verbatim, passaggi=(1,)):
    return {"statement": "…", "verbatim": verbatim, "passages": list(passaggi)}


def test_la_nota_nomina_la_legge_e_dice_quanto_e_diffusa():
    idx = _indice()
    prov = P.calcola([_claim(PREMESSA[:130])], {1: PREMESSA}, idx, min_documenti=3)
    assert not prov.vuota
    nota = P.rendi(prov)
    assert "legge 27 febbraio 1967, n. 48" in nota
    assert "31 delle 31 delibere indicizzate" in nota
    # Le tre cose chieste, e nell'ordine in cui servono a chi legge.
    assert nota.index("ricorre in") < nota.index("Il passaggio citato")
    assert "la risposta non cambia" in nota


def test_uno_span_con_una_cifra_propria_non_e_boilerplate():
    """La premessa che gli sta attorno ricorre in tutto il corpus, ma il fatto affermato è
    un importo: la nota lì sarebbe rumore, e direbbe a chi legge che una cifra unica è una
    formula ripetuta. Misurato su `ic-06-bis` e `ic-07`, dove infatti tace."""
    idx = _indice()
    passaggio = PREMESSA + SPECIFICA
    claim = _claim("sono riassegnati 7.526.144,01 euro dal programma Cantieri in Comune")
    prov = P.calcola([claim], {1: passaggio}, idx, 3)
    assert prov.vuota
    assert P.rendi(prov) == ""


def test_il_numero_della_legge_non_conta_come_cifra_propria():
    """«legge 27 febbraio 1967, n. 48» è fatta di cifre, ma appartengono al *nome* della
    norma, non all'affermazione. Contarle spegnerebbe la nota proprio sui richiami
    normativi, cioè sul caso per cui esiste."""
    assert P._numero_proprio("VISTA la legge 27 febbraio 1967, n. 48, recante «Attribuzioni»") is False
    assert P._numero_proprio("sono assegnati 295.178.000 euro alle quote premiali") is True


def test_un_refuso_nella_trascrizione_non_fa_tacere_la_nota():
    """Lo span è trascritto dal modello e la trascrizione sbaglia: su `ic-03` — l'item più
    emblematico — dice «Atribuzioni» dove il passaggio dice «Attribuzioni». Con il
    confronto per sottostringa la nota taceva proprio dove serviva."""
    idx = _indice()
    span = "VISTA la legge 27 febbraio 1967, n. 48, recante «Atribuzioni e ordinamento del Ministero del bilancio»"
    prov = P.calcola([_claim(span)], {1: PREMESSA}, idx, 3)
    assert not prov.vuota and prov.fonti[0].norme == ("legge 27 febbraio 1967, n. 48",)


def test_sotto_soglia_non_si_dichiara_niente():
    idx = _indice(n_documenti=30)
    assert P.calcola([_claim(PREMESSA[:130])], {1: PREMESSA}, idx, min_documenti=100).vuota


def test_indice_vuoto_lascia_la_risposta_intatta():
    from ingest.frasi import IndiceFrasi

    prov = P.calcola([_claim(PREMESSA[:130])], {1: PREMESSA}, IndiceFrasi.vuoto(), 3)
    assert prov.vuota
    assert P.applica("La risposta.", prov) == "La risposta."


def test_la_nota_si_aggiunge_in_coda_senza_toccare_la_risposta():
    idx = _indice()
    prov = P.calcola([_claim(PREMESSA[:130])], {1: PREMESSA}, idx, 3)
    testo = P.applica("Il CIPE è stato istituito nel 1967 [1].", prov)
    assert testo.startswith("Il CIPE è stato istituito nel 1967 [1].")
    assert "**Provenienza.**" in testo


def test_piu_passaggi_citati_concordano_al_plurale():
    idx = _indice()
    prov = P.calcola([_claim(PREMESSA[:130], passaggi=(1, 2))], {1: PREMESSA, 2: PREMESSA}, idx, 3)
    nota = P.rendi(prov)
    assert "I passaggi citati in [1][2] sono fra quelli" in nota


def test_uno_span_troppo_corto_non_si_prova_a_localizzare():
    """Meglio nessuna nota che una nota agganciata alla frase sbagliata: con poche parole
    la sovrapposizione non identifica niente."""
    idx = _indice()
    assert P.calcola([_claim("il CIPE")], {1: PREMESSA}, idx, 3).vuota
