"""Guardia verbatim: le parole dichiarate esistono davvero dove il modello dice.

Il precedente che la giustifica è misurato, non teorico: in `db-delibere/apps/ai_analysis`
il prompt chiedeva di citare le delibere e nell'output generato le citazioni sono **zero**.
La provenienza chiesta non arriva: va verificata.

**Cosa NON cattura**, e va tenuto presente leggendo i numeri: il caso `ic-07-bis`. Se il
modello riporta una cifra che *esiste* nel passaggio ma si riferisce ad altra spesa, la
substring c'è e il controllo passa. Quel guasto lo prende il guardiano IDF a monte, e la
sua causa vera è il retrieval.
"""

from talkdocs_rag_core.rag import guard
from talkdocs_rag_core.rag.generation import Passage
from talkdocs_rag_core.rag.verbatim import (
    ESITO_MISATTRIBUITO,
    ESITO_NON_TROVATO,
    ESITO_TROPPO_CORTO,
    ESITO_VALIDO,
    normalizza,
    ripulisci_span,
    verifica,
)

TESTO_1 = (
    "Il Comitato assegna in via programmatica la somma di euro 295.178.000 "
    "a valere sulle risorse del Fondo sviluppo e coesione per l'anno 2020."
)
TESTO_2 = "Le quote premiali sono accantonate presso il Ministero dell'economia e delle finanze."

# Ricalca il chunk reale `delibere/2025/E250021.txt::22` su cui la guardia ha segnato 0/2
# (run-20260807T162233Z): sillabazione da PDF e virgolette caporali **legittime**, parte del
# testo della delibera.
TESTO_3 = (
    "Visto il decreto del Presidente del Consiglio dei mini-\nstri 25 novembre 2022, con il quale "
    "il senatore Alessandro Morelli è stato nominato Segretario del Comitato, e gli è stata "
    "assegnata, tra le altre, la delega ad esercitare le funzioni spettanti al Presidente del "
    "Consiglio dei ministri in materia di coordinamento della politica economi-\nca e di "
    "programmazione e monitoraggio degli investimenti pubblici;\n"
    "Considerato che ai sensi dell'art. 16, comma 3, della legge 27 febbraio 1967, n. 48, «In "
    "caso di assenza o impedimento tempo-\nraneo del Presidente del Consiglio dei ministri, il "
    "Comitato è presieduto dal Ministro dell’economia e delle finanze in qualità di vice "
    "presidente del Comitato stesso»;"
)

SPAN_TRONCATO = (
    "e gli è stata assegnata, tra le altre, la delega ad esercitare le funzioni spettanti al "
    "Presidente del Consiglio dei ministri in materia di coordinamento della politica economica"
)
SPAN_CITATO = (
    "In caso di assenza o impedimento temporaneo del Presidente del Consiglio dei ministri, il "
    "Comitato è presieduto dal Ministro dell’economia e delle finanze"
)


def _passaggi():
    return [
        Passage(n=1, chunk_id="c1", source="delibere/2020/E200001.txt", content=TESTO_1),
        Passage(n=2, chunk_id="c2", source="delibere/2020/E200002.txt", content=TESTO_2),
    ]


# Ricalca il chunk reale `delibere/2023/E230007.txt::21` (eval-20260807T164834Z, domanda sui
# lotti costruttivi della Torino-Lione): è il passaggio su cui il modello ha appiccicato il
# marcatore `[2]` **dentro** il campo `verbatim`.
TESTO_4 = (
    "VISTA la delibera CIPE 15 febbraio 2022, n. 3, con la quale questo Comitato ha autorizzato il 4° "
    "lotto costruttivo, ha modificato la prescrizione n. 9 della delibera CIPE n. 39 del 2018 ed ha "
    "autorizzato la rimodulazione della ripartizione degli interventi fra il 3°, il 4° e il 5° lotto "
    "costruttivo, evidenziando nelle disponibilità del quadro economico l’importo di 51,32 milioni di "
    "euro, quali “Ulteriori risorse disponibili da assegnare” derivanti dalla legge di bilancio 2022;"
)

# Ricalcano i chunk reali `delibere/2019/E190007.txt::20` e `delibere/2019/E190002.txt::7`
# (eval-20260807T164834Z, domanda «Dove deve essere evidenziato il CUP?»): il modello ha
# concatenato con uno slash due citazioni prese da passaggi diversi.
TESTO_5 = (
    "2.4 Ai sensi della delibera n. 24 del 2004, il CUP assegnato all’opera dovrà essere evidenziato "
    "in tutta la documentazione amministrativa e contabile. Roma, 4 aprile 2019"
)
TESTO_6 = (
    "questo Comitato ha definito il sistema per l’attribuzione del CUP e ha stabilito che il CUP deve "
    "essere riportato su tutti i documenti amministrativi e contabili, cartacei ed informatici, "
    "relativi a progetti di investimento pubblico, e deve essere utilizzato nelle banche dati dei vari "
    "sistemi informativi, comunque interessati ai suddetti progetti;"
)


def _passaggi_delibera():
    return [Passage(n=5, chunk_id="delibere/2025/E250021.txt::22", source="delibere/2025/E250021.txt",
                    content=TESTO_3)]


def _passaggi_lotti():
    return [Passage(n=2, chunk_id="delibere/2023/E230007.txt::21", source="delibere/2023/E230007.txt",
                    content=TESTO_4)]


def _passaggi_cup():
    return [
        Passage(n=1, chunk_id="delibere/2019/E190007.txt::20", source="delibere/2019/E190007.txt",
                content=TESTO_5),
        Passage(n=3, chunk_id="delibere/2019/E190002.txt::7", source="delibere/2019/E190002.txt",
                content=TESTO_6),
    ]


def _claim(verbatim: str) -> list[dict]:
    return [{"statement": "…", "passages": [5], "verbatim": verbatim}]


def test_normalizza_collassa_gli_a_capo_del_pdf():
    """Il testo estratto da PDF ha a capo arbitrari a metà frase."""
    assert normalizza("euro   295.178.000\n a valere") == "euro 295.178.000 a valere"


def test_normalizza_riunisce_la_sillabazione():
    """«intel- ligenza» dall'estrazione PDF: stessa regola già usata dal guardiano IDF."""
    assert normalizza("l'intel- ligenza artificiale") == "l'intelligenza artificiale"


def test_normalizza_unifica_gli_apostrofi():
    assert normalizza("dell’economia") == normalizza("dell'economia")


def test_span_presente_nel_passaggio_citato_e_valido():
    claims = [{"statement": "Assegna 295 milioni", "passages": [1],
               "verbatim": "assegna in via programmatica la somma di euro 295.178.000"}]
    out = verifica(claims, _passaggi(), min_chars=40)
    assert out.n_valid == 1 and out.valid_ratio == 1.0
    assert out.per_claim[0]["esito"] == ESITO_VALIDO


def test_span_in_un_altro_passaggio_e_misattribuito():
    """Difetto diverso dalla fabbricazione, e oggi invisibile: va contato a parte."""
    claims = [{"statement": "Le quote premiali sono accantonate", "passages": [1],
               "verbatim": "Le quote premiali sono accantonate presso il Ministero dell'economia"}]
    out = verifica(claims, _passaggi(), min_chars=40)
    assert out.n_misattributed == 1 and out.n_valid == 0
    assert out.per_claim[0]["esito"] == ESITO_MISATTRIBUITO
    assert out.per_claim[0]["matched_passage"] == 2


def test_span_inventato_non_e_trovato():
    claims = [{"statement": "Stanzia 32,5 milioni", "passages": [1],
               "verbatim": "la somma di euro 32.500.000 destinata alle quote premiali"}]
    out = verifica(claims, _passaggi(), min_chars=40)
    assert out.n_not_found == 1 and out.valid_ratio == 0.0
    assert out.per_claim[0]["esito"] == ESITO_NON_TROVATO


def test_span_troppo_corto_non_e_valido_anche_se_esiste():
    """Sotto una certa lunghezza il test di appartenenza è banalmente soddisfacibile."""
    claims = [{"statement": "…", "passages": [1], "verbatim": "il Comitato"}]
    out = verifica(claims, _passaggi(), min_chars=40)
    assert out.n_too_short == 1 and out.n_valid == 0
    assert out.per_claim[0]["esito"] == ESITO_TROPPO_CORTO


def test_verbatim_vuoto_conta_come_non_valido():
    """Campo vuoto peggiora le statistiche, campo assente sparirebbe. È giusto così."""
    claims = [{"statement": "…", "passages": [1], "verbatim": ""}]
    out = verifica(claims, _passaggi(), min_chars=40)
    assert out.n_valid == 0 and out.valid_ratio == 0.0


def test_nessun_claim_lascia_il_rapporto_indefinito():
    """Succede col salvage della prosa da JSON troncato: la guardia non deve scattare."""
    out = verifica([], _passaggi(), min_chars=40)
    assert out.n_claims == 0 and out.valid_ratio is None


def test_confine_di_frase_misurato_non_bloccante():
    """Diagnostica: su testo da PDF la punteggiatura è troppo inaffidabile per farci una guardia."""
    interi = [{"statement": "…", "passages": [2], "verbatim": TESTO_2}]
    out = verifica(interi, _passaggi(), min_chars=40)
    assert out.span_boundary_ratio == 1.0
    assert out.n_valid == 1  # il confine non incide sulla validità


# --- La normalizzazione è condivisa, non ricopiata ---


def test_normalizza_delega_al_guardiano(monkeypatch):
    """Se `rag.guard` cambia nozione di «stessa stringa», `rag.verbatim` la segue.

    Ricopiare i passi base è il guasto che si manifesta mesi dopo, quando uno dei due moduli
    impara una variante di apostrofo e l'altro no.
    """
    monkeypatch.setattr(guard, "normalizza", lambda t: t.replace("œ", "oe"))
    assert normalizza("cœsione   territoriale") == "coesione territoriale"


# --- Uno span vuoto non è valido, qualunque sia la soglia ---


def test_span_vuoto_non_e_valido_nemmeno_con_min_chars_zero():
    """`"" in qualunque_testo` è sempre True: con soglia 0 ogni claim senza verbatim passerebbe.

    La metrica diventerebbe silenziosamente ottimistica per configurazione, non per merito.
    """
    for vuoto in ("", "   ", "…", "..."):
        out = verifica(_claim(vuoto), _passaggi_delibera(), min_chars=0)
        assert out.n_valid == 0, vuoto
        assert out.valid_ratio == 0.0, vuoto
        assert out.per_claim[0]["esito"] == ESITO_TROPPO_CORTO, vuoto


# --- Artefatti di formattazione del modello: la guardia misura la fabbricazione ---


def test_ellissi_finale_del_modello_non_falsifica_uno_span_corretto():
    """Caso reale (run-20260807T162233Z): il modello tronca lo span con `...`. Le parole ci sono."""
    out = verifica(_claim(SPAN_TRONCATO + "..."), _passaggi_delibera(), min_chars=40)
    assert out.per_claim[0]["esito"] == ESITO_VALIDO
    assert out.valid_ratio == 1.0


def test_ellissi_tipografica_e_iniziale_sono_equivalenti():
    out = verifica(_claim("… " + SPAN_TRONCATO + " …"), _passaggi_delibera(), min_chars=40)
    assert out.per_claim[0]["esito"] == ESITO_VALIDO


def test_virgolette_dritte_aggiunte_dal_modello_non_falsificano_lo_span():
    """Caso reale (run-20260807T162233Z): il modello racchiude lo span fra virgolette sue."""
    out = verifica(_claim(f'"{SPAN_CITATO}"'), _passaggi_delibera(), min_chars=40)
    assert out.per_claim[0]["esito"] == ESITO_VALIDO


def test_virgolette_caporali_di_contorno_non_falsificano_lo_span():
    out = verifica(_claim(f"«{SPAN_CITATO}»"), _passaggi_delibera(), min_chars=40)
    assert out.per_claim[0]["esito"] == ESITO_VALIDO


def test_virgolette_curve_inglesi_di_contorno_non_falsificano_lo_span():
    out = verifica(_claim(f"“{SPAN_CITATO}”"), _passaggi_delibera(), min_chars=40)
    assert out.per_claim[0]["esito"] == ESITO_VALIDO


def test_virgolette_e_ellissi_annidate_si_tolgono_insieme():
    out = verifica(_claim(f'"{SPAN_TRONCATO}…"'), _passaggi_delibera(), min_chars=40)
    assert out.per_claim[0]["esito"] == ESITO_VALIDO


def test_ellissi_in_mezzo_resta_non_trovata():
    """Il modello ha saltato del testo: lo span non è più letterale, e non deve risultare valido."""
    saltato = ("il Comitato è presieduto dal Ministro dell’economia … in qualità di vice presidente "
               "del Comitato stesso")
    out = verifica(_claim(saltato), _passaggi_delibera(), min_chars=40)
    assert out.per_claim[0]["esito"] == ESITO_NON_TROVATO
    assert out.n_valid == 0


def test_le_virgolette_interne_al_passaggio_non_vengono_toccate():
    """Si ripulisce lo **span**, mai il passaggio: le caporali della delibera sono testo suo."""
    con_caporale = "della legge 27 febbraio 1967, n. 48, «In caso di assenza o impedimento temporaneo"
    out = verifica(_claim(con_caporale), _passaggi_delibera(), min_chars=40)
    assert out.per_claim[0]["esito"] == ESITO_VALIDO


def test_una_virgoletta_spaiata_non_e_di_contorno():
    """Solo le coppie che *racchiudono* lo span sono artefatto: il resto è testo."""
    inventato = f'"{SPAN_CITATO} e delle politiche di coesione territoriale'
    out = verifica(_claim(inventato), _passaggi_delibera(), min_chars=40)
    assert out.per_claim[0]["esito"] == ESITO_NON_TROVATO


# --- I marcatori di citazione `[n]` finiti dentro il campo verbatim ---


def _claim_lotti(verbatim: str) -> list[dict]:
    return [{"statement": "…", "passages": [2], "verbatim": verbatim}]


def test_marcatore_di_citazione_in_coda_non_falsifica_lo_span():
    """Caso reale (eval-20260807T164834Z): il modello chiude lo span con `[2]`.

    Le parole ci sono, nel passaggio dichiarato: a fallire è la sintassi del contratto finita
    nel campo sbagliato. La guardia misura la fabbricazione, non la formattazione.
    """
    span = ("ha modificato la prescrizione n. 9 della delibera CIPE n. 39 del 2018 ed ha autorizzato "
            "la rimodulazione della ripartizione degli interventi fra il 3°, il 4° e il 5° lotto "
            "costruttivo [2]")
    out = verifica(_claim_lotti(span), _passaggi_lotti(), min_chars=40)
    assert out.per_claim[0]["esito"] == ESITO_VALIDO
    assert out.per_claim[0]["matched_passage"] == 2


def test_marcatore_in_mezzo_allo_span_si_toglie_e_non_lascia_spazi_doppi():
    """Nella prosa del modello il marcatore non sta solo in coda: va tolto **ovunque**.

    E lo span registrato è quello confrontato: se restasse lo spazio doppio, chi rilegge
    l'audit vedrebbe una stringa che non ha mai fatto da termine di paragone.
    """
    span = ("ha modificato la prescrizione n. 9 della delibera CIPE n. 39 del 2018 [2] ed ha "
            "autorizzato la rimodulazione della ripartizione degli interventi")
    assert "  " not in ripulisci_span(span)
    out = verifica(_claim_lotti(span), _passaggi_lotti(), min_chars=40)
    assert out.per_claim[0]["esito"] == ESITO_VALIDO


def test_marcatori_consecutivi_si_tolgono_tutti():
    """Caso reale: `[3][4][5]` in coda a uno span altrimenti letterale."""
    span = ("deve essere utilizzato nelle banche dati dei vari sistemi informativi, comunque "
            "interessati ai suddetti progetti [3][4][5]")
    claims = [{"statement": "…", "passages": [3, 4, 5], "verbatim": span}]
    out = verifica(claims, _passaggi_cup(), min_chars=40)
    assert out.per_claim[0]["esito"] == ESITO_VALIDO
    assert out.per_claim[0]["matched_passage"] == 3


def test_ellissi_in_parentesi_quadre_non_e_un_marcatore():
    """Caso reale (eval-20260807T164834Z): `[...]` dice che il modello ha **saltato** del testo.

    Solo cifre, mai puntini: lì lo span non è letterale e deve restare non valido.
    """
    span = ("ha modificato [...] la ripartizione degli interventi fra il 3°, il 4° e il 5° lotto "
            "costruttivo")
    out = verifica(_claim_lotti(span), _passaggi_lotti(), min_chars=40)
    assert out.per_claim[0]["esito"] == ESITO_NON_TROVATO
    assert out.n_valid == 0


def test_span_composto_da_due_citazioni_resta_non_valido():
    """Caso reale: due passaggi diversi concatenati con uno slash.

    Tolti i marcatori resta la concatenazione, che letterale non è in nessuno dei due: la
    ripulitura non deve renderla valida per sbaglio.
    """
    span = ("il CUP assegnato all’opera dovrà essere evidenziato in tutta la documentazione "
            "amministrativa e contabile riguardante l’opera stessa [1][2] / deve essere riportato su "
            "tutti i documenti amministrativi e contabili, cartacei ed informatici, relativi a "
            "progetti di investimento pubblico [3][4][5]")
    claims = [{"statement": "…", "passages": [1, 3], "verbatim": span}]
    out = verifica(claims, _passaggi_cup(), min_chars=40)
    assert out.per_claim[0]["esito"] == ESITO_NON_TROVATO
    assert out.n_valid == 0


def test_il_marcatore_si_toglie_allo_span_e_mai_al_passaggio():
    """Il passaggio non si ripulisce: nelle delibere `[n]` può essere testo della delibera.

    Il prezzo è dichiarato, non nascosto: se le quadre col numero sono testo *del passaggio*,
    lo span che le riporta fedelmente risulta non valido. Ripulire anche il passaggio
    renderebbe indistinguibile un marcatore inventato da uno trascritto — un falso negativo
    della guardia, che è l'errore peggiore dei due. Il prezzo è misurato e piccolo: nel corpus
    le quadre con dentro solo cifre sono **3 occorrenze in 1 delibera su 511** (`E200064`).
    """
    contenuto = "il quadro economico [2] di cui alla tabella allegata alla presente delibera"
    passaggi = [Passage(n=2, chunk_id="c", source="s", content=contenuto)]
    out = verifica([{"statement": "…", "passages": [2], "verbatim": contenuto}], passaggi, min_chars=20)
    assert out.per_claim[0]["esito"] == ESITO_NON_TROVATO
