"""Prosa della risposta calcolata: template, mai il modello.

Due proprietà da fissare. La risposta deve dichiarare il **perimetro** (il conteggio è sul
corpus indicizzato, non sull'archivio completo), e deve fare il **controllo di
completezza**: 93 delibere presenti ma numerazione fino alla 95 significa che due mancano,
ed è la differenza fra un numero e un numero difendibile.
"""

import pytest

from talkdocs_rag_core.structured import intents
from talkdocs_rag_core.structured.answer import componi


def test_count_dichiara_il_perimetro_e_la_provenienza_del_numero():
    testo, out = componi(
        intent=intents.COUNT_DELIBERE,
        params={"anno": 2024, "comitato": "CIPESS"},
        rows=[{"n": 93, "min_numero": 1, "max_numero": 95}],
        sql="SELECT COUNT(*) …",
        sql_params=[2024, "CIPESS"],
        max_rows=20,
    )
    assert "93" in testo
    assert "corpus indicizzato" in testo
    assert "calcolato sui metadati" in testo
    assert out.computed_value == 93


def test_controllo_di_completezza_quando_mancano_delibere():
    testo, out = componi(
        intent=intents.COUNT_DELIBERE,
        params={"anno": 2024},
        rows=[{"n": 93, "min_numero": 1, "max_numero": 95}],
        sql="…",
        sql_params=[2024],
        max_rows=20,
    )
    assert out.completeness == {"count": 93, "max_numero": 95, "gap": 2}
    assert "95" in testo


def test_nessun_gap_nessuna_frase_sulle_mancanti():
    testo, out = componi(
        intent=intents.COUNT_DELIBERE,
        params={"anno": 2021},
        rows=[{"n": 91, "min_numero": 1, "max_numero": 91}],
        sql="…",
        sql_params=[2021],
        max_rows=20,
    )
    assert out.completeness["gap"] == 0
    # Il lessico vivo della frase sulle mancanti è «N delibere mancano dal corpus»: agganciare
    # l'assert a una formulazione che nessun ramo produce più lo renderebbe vero a prescindere.
    assert "manca" not in testo


def test_senza_anno_niente_controllo_di_completezza():
    """`MAX(numero)` non ha senso su più anni: la numerazione riparte ogni anno."""
    _, out = componi(
        intent=intents.COUNT_DELIBERE,
        params={},
        rows=[{"n": 511, "min_numero": 1, "max_numero": 95}],
        sql="…",
        sql_params=[],
        max_rows=20,
    )
    assert out.completeness == {}


def test_list_tronca_la_vista_ma_non_l_audit():
    righe = [
        {
            "anno": 2021,
            "numero": i,
            "comitato": "CIPESS",
            "title": f"Delibera CIPESS n. {i}/2021",
            "path": f"delibere/2021/E2100{i:02d}.txt",
        }
        for i in range(1, 31)
    ]
    testo, out = componi(
        intent=intents.LIST_DELIBERE,
        params={"anno": 2021},
        rows=righe,
        sql="…",
        sql_params=[2021],
        max_rows=5,
    )
    assert out.n_rows == 30
    assert len(out.rows) == 30  # l'audit le porta tutte
    assert testo.count("Delibera CIPESS n.") == 5  # la vista ne mostra 5
    assert "altre 25" in testo
    assert out.computed_value == 30
    assert len(out.cited_doc_ids) == 30


def test_count_by_year_elenca_gli_anni():
    testo, out = componi(
        intent=intents.COUNT_BY_YEAR,
        params={},
        rows=[{"anno": 2019, "n": 84}, {"anno": 2020, "n": 80}],
        sql="…",
        sql_params=[],
        max_rows=20,
    )
    assert "2019" in testo and "84" in testo
    assert out.computed_value is None  # non è un numero singolo


def test_count_su_zero_righe_lo_dice():
    testo, out = componi(
        intent=intents.COUNT_DELIBERE,
        params={"anno": 1985},
        rows=[{"n": 0, "min_numero": None, "max_numero": None}],
        sql="…",
        sql_params=[1985],
        max_rows=20,
    )
    assert out.computed_value == 0
    assert "nessuna delibera" in testo.lower()


def test_count_accorda_il_singolare():
    """«Risultano 1 delibere» è una frase che squalifica il numero che la precede."""
    testo, _ = componi(
        intent=intents.COUNT_DELIBERE,
        params={"anno": 2024, "comitato": "CIPESS"},
        rows=[{"n": 1, "min_numero": 1, "max_numero": 2}],
        sql="…",
        sql_params=[2024, "CIPESS"],
        max_rows=20,
    )
    assert "risulta **1 delibera**" in testo
    assert "1 delibere" not in testo


def test_list_accorda_il_singolare():
    testo, _ = componi(
        intent=intents.LIST_DELIBERE,
        params={"anno": 2024},
        rows=[{"anno": 2024, "numero": 1, "comitato": "CIPESS", "title": "Delibera CIPESS n. 1/2024", "path": "a.txt"}],
        sql="…",
        sql_params=[2024],
        max_rows=20,
    )
    assert "risulta **1 delibera**" in testo
    assert "1 delibere" not in testo


def test_list_su_zero_righe_chiude_la_frase():
    """Zero righe non deve produrre due punti pendenti seguiti da un elenco vuoto."""
    testo, out = componi(
        intent=intents.LIST_DELIBERE,
        params={"anno": 1985},
        rows=[],
        sql="…",
        sql_params=[1985],
        max_rows=20,
    )
    assert out.n_rows == 0
    assert "nessuna delibera" in testo.lower()
    assert ":" not in testo
    assert "0 delibere" not in testo


def test_perimetro_mette_il_comitato_prima_del_tempo():
    """«nel 2024 del CIPESS» è italiano legnoso: l'ordine naturale è «del CIPESS nel 2024»."""
    testo, _ = componi(
        intent=intents.COUNT_DELIBERE,
        params={"anno": 2024, "comitato": "CIPESS"},
        rows=[{"n": 93, "min_numero": 1, "max_numero": 95}],
        sql="…",
        sql_params=[2024, "CIPESS"],
        max_rows=20,
    )
    assert "del CIPESS nel 2024" in testo
    assert "nel 2024 del CIPESS" not in testo


def test_gap_misurato_sull_intervallo_osservato_non_sulla_base_uno():
    """Numerazione da 3 a 95 con 93 delibere: l'intervallo è completo, il gap è 0.

    Assumere la base 1 conterebbe come mancanti due delibere che semplicemente non
    esistono con quei numeri — un'affermazione sul mondo, non una misura."""
    testo, out = componi(
        intent=intents.COUNT_DELIBERE,
        params={"anno": 2024},
        rows=[{"n": 93, "min_numero": 3, "max_numero": 95}],
        sql="…",
        sql_params=[2024],
        max_rows=20,
    )
    assert out.completeness == {"count": 93, "max_numero": 95, "gap": 0}
    assert "manca" not in testo


def test_la_frase_sulle_mancanti_non_afferma_nulla_fuori_dal_corpus():
    """Il fatto misurato è l'assenza dal corpus, non l'inesistenza della delibera.

    Dichiarare il perimetro nella prima frase e sfondarlo nella seconda («non sono
    presenti nell'archivio») annulla il perimetro."""
    testo, _ = componi(
        intent=intents.COUNT_DELIBERE,
        params={"anno": 2024},
        rows=[{"n": 93, "min_numero": 1, "max_numero": 95}],
        sql="…",
        sql_params=[2024],
        max_rows=20,
    )
    assert "archivio" not in testo
    assert "2 delibere mancano dal corpus" in testo


def test_la_frase_sulle_mancanti_accorda_il_singolare():
    testo, _ = componi(
        intent=intents.COUNT_DELIBERE,
        params={"anno": 2024},
        rows=[{"n": 94, "min_numero": 1, "max_numero": 95}],
        sql="…",
        sql_params=[2024],
        max_rows=20,
    )
    assert "1 delibera manca dal corpus" in testo
    assert "1 delibere" not in testo


def test_il_perimetro_si_dichiara_una_volta_sola():
    """Tre «corpus indicizzato» in tre frasi sminuiscono la difendibilità che perseguono.

    Il perimetro si dichiara una volta, in apertura, e vale per tutta la risposta: la frase
    sulle mancanti vi si aggancia («mancano dal corpus»), la nota finale non lo ripete."""
    testo, _ = componi(
        intent=intents.COUNT_DELIBERE,
        params={"anno": 2024, "comitato": "CIPESS"},
        rows=[{"n": 93, "min_numero": 1, "max_numero": 95}],
        sql="…",
        sql_params=[2024, "CIPESS"],
        max_rows=20,
    )
    assert testo.count("corpus indicizzato") == 1
    assert "nel 2024 nel corpus" not in testo  # il doppio «nel» consecutivo


def test_il_perimetro_si_dichiara_una_volta_sola_anche_in_elenco_e_distribuzione():
    elenco, _ = componi(
        intent=intents.LIST_DELIBERE,
        params={"anno": 2021, "comitato": "CIPESS"},
        rows=[{"anno": 2021, "numero": 1, "comitato": "CIPESS", "title": "Delibera CIPESS n. 1/2021", "path": "a.txt"}],
        sql="…",
        sql_params=[2021, "CIPESS"],
        max_rows=20,
    )
    distribuzione, _ = componi(
        intent=intents.COUNT_BY_YEAR,
        params={},
        rows=[{"anno": 2019, "n": 84}, {"anno": 2020, "n": 80}],
        sql="…",
        sql_params=[],
        max_rows=20,
    )
    assert elenco.count("corpus indicizzato") == 1
    assert distribuzione.count("corpus indicizzato") == 1
    assert "nel 2021 nel corpus" not in elenco


def test_la_nota_finale_non_parla_di_conteggio_sotto_un_elenco():
    """Sotto un elenco di titoli e sotto una distribuzione per anno «il conteggio» è la cosa
    sbagliata: la nota dice della provenienza del dato, e vale per tutti e tre i rami."""
    elenco, _ = componi(
        intent=intents.LIST_DELIBERE,
        params={"anno": 2021},
        rows=[{"anno": 2021, "numero": 1, "comitato": "CIPESS", "title": "Delibera CIPESS n. 1/2021", "path": "a.txt"}],
        sql="…",
        sql_params=[2021],
        max_rows=20,
    )
    distribuzione, _ = componi(
        intent=intents.COUNT_BY_YEAR,
        params={},
        rows=[{"anno": 2019, "n": 84}],
        sql="…",
        sql_params=[],
        max_rows=20,
    )
    for testo in (elenco, distribuzione):
        assert "conteggio" not in testo
        assert "calcolato sui metadati" in testo


def test_la_coda_dell_elenco_non_parla_in_gergo():
    """«La tupla di audit» è gergo interno: il committente non sa cosa sia. L'informazione da
    dare è un'altra — le righe non mostrate esistono e restano registrate."""
    righe = [
        {
            "anno": 2021,
            "numero": i,
            "comitato": "CIPESS",
            "title": f"Delibera CIPESS n. {i}/2021",
            "path": f"delibere/2021/E2100{i:02d}.txt",
        }
        for i in range(1, 31)
    ]
    testo, _ = componi(
        intent=intents.LIST_DELIBERE,
        params={"anno": 2021},
        rows=righe,
        sql="…",
        sql_params=[2021],
        max_rows=5,
    )
    assert "tupla" not in testo
    assert "audit" not in testo
    assert "altre 25 non mostrate" in testo
    assert "registrato" in testo  # le righe non mostrate esistono e sono conservate


def test_intento_non_riconosciuto_fa_rumore_invece_di_prosa():
    """La distribuzione per anno era il fall-through: ci cadeva dentro *qualunque* intento.

    Un intento ignoto non produceva un errore, produceva la prosa della distribuzione su
    righe che non hanno le colonne che quel ramo legge — cioè, nel caso fortunato, un
    `KeyError` lontano dalla causa; nel caso sfortunato, prosa plausibile e sbagliata.
    `serve_structured` non può arrivarci (`intents.build` filtra già su `TUTTI`), ma un
    guasto futuro deve fare rumore e nominare l'intento che non conosce.
    """
    with pytest.raises(ValueError, match="somma_importi"):
        componi(intent="somma_importi", params={}, rows=[], sql="…", sql_params=[], max_rows=20)
