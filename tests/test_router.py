"""Router deterministico: quale via risponde a questa domanda.

Il rischio non è «riconosce le domande facili». Sono due, opposti:

1. **fall-through silenzioso** — un'aggregativa fuori dai pattern non finisce nel ramo di
   rifiuto, cade in POINTWISE e ottiene un numero inventato con una citazione ben formata;
2. **falso positivo** — una domanda puntuale che comincia per «quanto» viene dirottata e
   rifiutata, su un sistema che già tace troppo.

La regola che disinnesca il secondo caso: il riferimento a una delibera **specifica** vince
su tutto. I test in coda documentano i limiti noti: falliscono per progetto se qualcuno
crede di averli risolti senza misurarli.
"""

import pytest

from rag import router
from structured import intents


@pytest.mark.parametrize(
    ("domanda", "atteso"),
    [
        ("Quante delibere ha adottato il CIPESS nel 2024?", intents.COUNT_DELIBERE),
        ("Quante delibere ci sono in tutto?", intents.COUNT_DELIBERE),
        ("Elenca le delibere del 2021", intents.LIST_DELIBERE),
        ("Quali sono le delibere del 2021?", intents.LIST_DELIBERE),
        ("Quante delibere per anno ha adottato il Comitato?", intents.COUNT_BY_YEAR),
    ],
)
def test_aggregative_coperte(domanda, atteso):
    r = router.classify(domanda)
    assert r.route == router.STRUCTURED
    assert r.intent == atteso


def test_estrae_anno_e_comitato():
    r = router.classify("Quante delibere ha adottato il CIPESS nel 2024?")
    assert r.params == {"anno": 2024, "comitato": "CIPESS"}


def test_estrae_intervallo_di_anni():
    r = router.classify("Quante delibere dal 2019 al 2021?")
    assert r.params["anno_da"] == 2019
    assert r.params["anno_a"] == 2021
    assert "anno" not in r.params


def test_un_anno_solo_resta_strutturata():
    """Regressione sul caso principale: il filtro anno è il motivo per cui il ramo esiste."""
    r = router.classify("Quante delibere del CIPE nel 2019?")
    assert r.route == router.STRUCTURED
    assert r.params == {"anno": 2019, "comitato": "CIPE"}
    assert r.signals["anni_multipli"] is False
    assert r.reason is None


def test_piu_anni_senza_intervallo_sono_fuori_copertura():
    """La classe di guasto peggiore, e il motivo per cui l'incremento 1 esiste.

    Con due anni sciolti il filtro non è esprimibile negli intenti chiusi di oggi. Lasciarlo
    cadere in silenzio non produce un rifiuto né un'allucinazione: produce il **conteggio
    globale** — un numero *calcolato*, e quindi credibile, alla domanda sbagliata. Meglio
    dichiarare di non saper contare *questo* che eseguire una query su un filtro amputato.
    """
    r = router.classify("Quante delibere del CIPE nel 2019 e nel 2021?")
    assert r.route == router.UNCOVERED
    assert r.reason == router.MOTIVO_ANNI_MULTIPLI
    assert r.signals["anni_multipli"] is True


def test_intervallo_riconosciuto_non_e_multi_anno():
    """«dal 2019 al 2021» contiene due anni ma li lega: il filtro c'è ed è esprimibile.

    Il segnale distingue *due anni sciolti* da *un intervallo*, non «quante cifre a quattro
    caratteri compaiono»: altrimenti la correzione al multi-anno spegnerebbe il caso coperto.
    """
    r = router.classify("Quante delibere dal 2019 al 2021?")
    assert r.route == router.STRUCTURED
    assert r.params["anno_da"] == 2019
    assert r.params["anno_a"] == 2021
    assert r.signals["anni_multipli"] is False
    assert r.reason is None


def test_un_anno_oltre_l_intervallo_e_fuori_copertura():
    """La stessa classe di guasto, nella variante che l'intervallo mascherava.

    Con «dal 2019 al 2021 e nel 2024» il range è riconosciuto e il 2024 resta fuori da
    qualunque filtro esprimibile: il conteggio uscirebbe **monco**, calcolato sui soli
    2019-2021, senza motivo e senza segnale in audit. Un numero monco è indistinguibile da
    uno giusto — è esattamente il guasto che il multi-anno esiste per chiudere, non un
    limite diverso. Perciò l'intervallo non basta a spegnere il segnale: conta se esiste un
    anno citato che non sia uno dei due estremi.
    """
    r = router.classify("Quante delibere del CIPE dal 2019 al 2021 e nel 2024?")
    assert r.route == router.UNCOVERED
    assert r.reason == router.MOTIVO_ANNI_MULTIPLI
    assert r.signals["anni_multipli"] is True


def test_un_estremo_ripetuto_non_e_un_anno_in_piu():
    """Il criterio è «anni fuori dagli estremi», non «quante volte compare una data».

    In «dal 2019 al 2021, nel 2021» il 2021 è già un estremo del range: non c'è nulla che
    il filtro non sappia esprimere, e il rifiuto sarebbe un falso positivo — il secondo dei
    due errori che questo router deve tenere separati.
    """
    r = router.classify("Quante delibere dal 2019 al 2021, nel 2021?")
    assert r.route == router.STRUCTURED
    assert r.params["anno_da"] == 2019
    assert r.params["anno_a"] == 2021
    assert r.signals["anni_multipli"] is False
    assert r.reason is None


def test_delibera_specifica_vince_anche_con_due_anni():
    """L'ordine della cascata *è* il progetto: la regola 1 resta la prima.

    Un atto nominato per numero è puntuale e rispondibile dal RAG, anche se nella frase
    compaiono due anni: il multi-anno non deve trasformarlo in un rifiuto.
    """
    r = router.classify("Che cosa prevede la delibera 75 del 2021 rispetto al 2019?")
    assert r.route == router.POINTWISE
    assert r.reason is None


@pytest.mark.parametrize(
    "domanda",
    [
        "Quante delibere ha adottato il CIPESS nel 2024?",             # strutturata
        "Che cosa prevede la delibera 75/2021?",                       # puntuale
        "Quanto è stato speso per le ferrovie?",                       # rifiuto per forma di massa
    ],
)
def test_motivo_assente_dove_vale_il_default(domanda):
    """`reason` è un'eccezione, non un campo da riempire sempre.

    `None` significa «il motivo di default» (nessuna tabella degli importi): il rifiuto per
    forma di massa lo usa, e le rotte non-uncovered non hanno un motivo da portare.
    """
    assert router.classify(domanda).reason is None


@pytest.mark.parametrize(
    "domanda",
    [
        "Quanto è stato speso per le ferrovie?",                       # non abbiamo importi
        "Qual è la dotazione complessiva del Fondo sviluppo e coesione?",  # bd-01
        "Quanti chilometri di rete ferroviaria sono stati realizzati?",    # bd-04
        "Quanti dipendenti ha ANAS?",                                  # bd-06
    ],
)
def test_aggregative_fuori_copertura_sono_rifiutate_dichiarando_il_motivo(domanda):
    assert router.classify(domanda).route == router.UNCOVERED


@pytest.mark.parametrize(
    "domanda",
    [
        "Quanto vale il fondo previsto dalla delibera 47 del 2024?",   # delibera specifica
        "Quali sono gli obiettivi del Fondo sviluppo e coesione?",     # «quali sono» senza oggetto
        "Che cosa prevede la delibera 75/2021?",
        "Quali interventi ha approvato il CIPE nel 1985?",             # bd-02: la prende il guardiano IDF
    ],
)
def test_puntuali_non_vengono_dirottate(domanda):
    assert router.classify(domanda).route == router.POINTWISE


def test_i_segnali_grezzi_finiscono_nel_risultato():
    """Servono in audit: un fall-through si conta, non si perde nel silenzio.

    L'uguaglianza è esatta di proposito: chi aggiunge un segnale senza portarlo nell'audit
    rompe questo test, e lo scopre qui invece che in una run di misura.
    """
    r = router.classify("Quanti dipendenti ha ANAS?")
    assert r.signals == {"forma_conteggio": True, "forma_massa": False, "oggetto_coperto": False,
                         "delibera_specifica": False, "anni_multipli": False,
                         "forma_meta": False, "oggetto_corpus": False}


@pytest.mark.xfail(
    reason="limite noto (spec §7): il richiamo del router lessicale è la metrica, non l'assunzione",
    strict=False,
)
@pytest.mark.parametrize(
    "domanda",
    [
        "Nel 2024 il Comitato quante ne ha approvate?",
        "Me ne fai l'elenco per il 2021?",
        "Di atti del 2024 ce ne sono molti?",
    ],
)
def test_riformulazioni_colloquiali_limite_noto(domanda):
    """Documenta il buco invece di nasconderlo. `strict=False`: se una inizia a passare,
    la suite non si rompe — ma il numero da guardare resta `router_recall` nell'eval."""
    assert router.classify(domanda).route == router.STRUCTURED
