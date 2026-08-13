"""Route META: la meta-domanda è sulla collezione, non su un contenuto.

I pattern sono stretti di proposito (forma meta E riferimento alla collezione): la
risposta — la scheda del corpus — è giusta solo se la domanda è davvero sull'archivio.
Il richiamo sulle formulazioni colloquiali è lavoro del router agentico, non delle regex.
"""

import pytest
from talkdocs_rag_core.rag import router


@pytest.mark.parametrize(
    "domanda",
    [
        "Di cosa parla questo corpus di documenti?",
        "Che tipo di documenti contiene l'archivio e che periodo copre?",
        "Cosa posso chiederti su questo archivio?",
        "Che cos'è questa raccolta?",
        "Quanti documenti contiene il corpus?",
    ],
)
def test_meta_domande_riconosciute(domanda):
    r = router.classify(domanda)
    assert r.route == router.META
    assert r.signals["forma_meta"] is True and r.signals["oggetto_corpus"] is True


@pytest.mark.parametrize(
    ("domanda", "attesa"),
    [
        # Tematica, non meta: la risposta starebbe nei testi, non nella scheda.
        ("Di cosa parlano le delibere del 2020?", router.POINTWISE),
        # L'atto nominato per numero resta puntuale, qualunque forma abbia la domanda.
        ("Di cosa parla la delibera 47 del 2024?", router.POINTWISE),
        # Aggregativa vera: l'oggetto coperto vince, il corpus citato non la rende meta.
        ("Quante delibere ci sono in tutto nel corpus?", router.STRUCTURED),
        # Contenuto di un documento: «cosa deve contenere» non è «cosa contiene il corpus».
        ("Cosa deve contenere il Piano sviluppo e coesione?", router.POINTWISE),
    ],
)
def test_non_meta_non_dirottate(domanda, attesa):
    assert router.classify(domanda).route == attesa


def test_colloquiale_fuori_pattern_e_un_limite_dichiarato():
    """`meta-04` dell'eval set: col solo lessicale finisce POINTWISE. È il caso che
    misura il richiamo aggiunto dal router agentico — qui si documenta, non si nasconde."""
    r = router.classify("Ma qui dentro che roba c'è?")
    assert r.route == router.POINTWISE
