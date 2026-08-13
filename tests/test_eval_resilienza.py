"""Un 429 sull'ultimo item non deve buttare via i 104 già pagati.

Misurato il 9 agosto: la run `eval-20260809T130308Z` è morta all'item 104 su 110 con un
`RateLimitError` risalito fino a `run_eval`. Nessun report, nessun bundle, nessun numero —
il criterio di successo del banco è produrre numeri, e una run che ne produce zero perché
l'ultimo item ha trovato la quota esaurita è un difetto del banco, non del provider.

Due proprietà, e servono entrambe:
1. si **ritenta** con pausa lunga (il retry dell'SDK copre il 429 istantaneo, non la
   finestra di quota esaurita);
2. esaurititi i tentativi si scrive una **riga di errore** e si prosegue — e quella riga
   sta fuori da ogni denominatore di merito, altrimenti misurerebbe la rete.
"""

import asyncio

import pytest
from openai import RateLimitError
from talkdocs_rag_core.config import RagConfig
from talkdocs_rag_core.eval.runner import EvalItem, _aggregate, _ask_con_ripresa, _row, _run_condition


def _rate_limit() -> RateLimitError:
    class _Resp:
        status_code = 429
        headers = {}
        request = None

    return RateLimitError("Rate limit exceeded", response=_Resp(), body=None)


class _PipelineFinta:
    """Fallisce le prime `n_fallimenti` chiamate, poi (se `guarisce`) risponde."""

    def __init__(self, cfg, n_fallimenti: int, guarisce: bool = True):
        self.cfg = cfg
        self.n_fallimenti = n_fallimenti
        self.guarisce = guarisce
        self.chiamate = 0
        self.corpus_version = "test"
        self.semantic_cache = type("C", (), {"clear": lambda self: None})()

    async def ask(self, query, use_cache=False, provider_cache_key=None):
        self.chiamate += 1
        if self.chiamate <= self.n_fallimenti or not self.guarisce:
            raise _rate_limit()
        from talkdocs_rag_core.rag.generation import RagResult

        return RagResult(
            query=query, answer_text="ok [1]", refused=False, refusal_reason=None,
            support_score=0.9, cited_passages=[1], cited_chunk_ids=["c1"], invalid_citations=[],
            claims=[], passages=[], usage={"prompt_tokens": 10, "completion_tokens": 5},
            raw_output="{}", model="finto", params={}, latency_s=0.1,
        )


class _WriterFinto:
    def __init__(self):
        self.registrate = 0

    def record(self, *a, **kw):
        self.registrate += 1


@pytest.fixture(autouse=True)
def _niente_attese(monkeypatch):
    """La pausa è reale in produzione e inutile nel test: si misura che ci sia, non quanto duri."""
    attese = []

    async def _sleep(s):
        attese.append(s)

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    return attese


ITEM = EvalItem("x-01", "in_corpus", False, "domanda?")


def test_ritenta_e_recupera(_niente_attese):
    cfg = RagConfig(eval_item_max_retries=4, eval_item_backoff_s=20.0)
    pipeline = _PipelineFinta(cfg, n_fallimenti=2)
    res, errore = asyncio.run(_ask_con_ripresa(pipeline, ITEM, "off", 0))
    assert errore == "" and res.answer_text == "ok [1]"
    assert pipeline.chiamate == 3
    # Backoff esponenziale, non una pausa fissa: una quota esaurita non si libera a ritmo.
    assert _niente_attese == [20.0, 40.0]


def test_esauriti_i_tentativi_la_run_prosegue(_niente_attese):
    cfg = RagConfig(eval_item_max_retries=2, eval_item_backoff_s=1.0)
    pipeline = _PipelineFinta(cfg, n_fallimenti=0, guarisce=False)
    res, errore = asyncio.run(_ask_con_ripresa(pipeline, ITEM, "off", 0))
    assert errore == "RateLimitError"
    # Non è un rifiuto: un rifiuto è una decisione del sistema, questo è un buco.
    assert res.refused is False and res.usage == {}
    assert pipeline.chiamate == 3  # 1 tentativo + 2 ripetizioni


def test_una_domanda_morta_non_uccide_la_condizione():
    cfg = RagConfig(eval_item_max_retries=0)
    items = [EvalItem(f"x-{i:02d}", "in_corpus", False, "q?") for i in range(4)]

    class _MuoreAllaTerza(_PipelineFinta):
        async def ask(self, query, use_cache=False, provider_cache_key=None):
            self.chiamate += 1
            if self.chiamate == 3:
                raise _rate_limit()
            return await super().ask(query, use_cache, provider_cache_key)

    pipeline = _MuoreAllaTerza(cfg, n_fallimenti=0)
    pipeline.chiamate = 0
    writer = _WriterFinto()
    rows = asyncio.run(_run_condition(pipeline, items, "off", writer))

    assert len(rows) == 4  # tutte le domande hanno una riga
    morte = [r for r in rows if r["errore"]]
    assert [r["id"] for r in morte] == ["x-01"]  # la terza chiamata è il secondo item
    # L'audit registra ciò che è stato servito: una risposta che non c'è non si audita.
    assert writer.registrate == 3


def test_la_riga_di_errore_sta_fuori_dai_denominatori():
    cfg = RagConfig()
    ok = _PipelineFinta(cfg, n_fallimenti=0)
    buona = asyncio.run(ok.ask("q?"))
    vuota = _row(cfg, EvalItem("m-01", "in_corpus", False, "q?"), "off", buona)
    morta = _row(
        cfg, EvalItem("m-02", "out_of_corpus", True, "q?"), "off",
        asyncio.run(_ask_con_ripresa(_PipelineFinta(RagConfig(eval_item_max_retries=0), 0, False),
                                     EvalItem("m-02", "out_of_corpus", True, "q?"), "off", 0))[0],
        errore="RateLimitError",
    )
    # `expect_refuse=True` + `refused=0` darebbe «rifiuto mancato»: sarebbe la rete a
    # rispondere al posto del sistema. La colonna resta vuota, non a 0.
    assert morta["refusal_correct"] == "" and vuota["refusal_correct"] == 1
    assert morta["source_id_ok"] == ""

    agg = _aggregate(cfg, [vuota, morta], "off")
    assert agg["n_errori"] == 1
    assert agg["n"] == 1  # il denominatore dichiarato è quello vero
    assert agg["refusal_accuracy"] == 1.0  # non 0.5: la morta non è un errore di rifiuto
    assert agg["n_answered"] == 1
