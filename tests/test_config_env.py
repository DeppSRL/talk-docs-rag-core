"""Lettura dei parametri di run dall'ambiente.

Trappola reale, incontrata copiando ``.env.example`` in ``.env``: ``python-dotenv``
rimuove il commento inline **solo se la variabile ha un valore**. Su una riga
``SUPPORT_THRESHOLD=          # da tarare`` il commento *diventa* il valore, e il banco
esplode con un ``ValueError`` a metà configurazione. Sulla chiave API è peggio: una key
vuota diventerebbe la stringa ``# SECRET → …`` e partirebbe verso l'API, restituendo un
401 invece di un chiaro "chiave assente".

Nessuno di questi parametri (soglie, modelli, path, prezzi, chiavi) può legittimamente
iniziare con ``#``: un valore così è un residuo di commento e va trattato come assente.
"""

import pytest

from talk_docs_rag_core.config import RagConfig, _get, _get_bool, _get_float, _get_int


class TestCommentoResiduo:
    def test_valore_che_e_solo_un_commento_vale_assente(self, monkeypatch):
        monkeypatch.setenv("X_TEST", "# sotto → rifiuto deterministico (C3), da tarare")
        assert _get("X_TEST", "fallback") == "fallback"

    def test_float_cade_sul_default_invece_di_esplodere(self, monkeypatch):
        monkeypatch.setenv("X_TEST", "# da tarare")
        assert _get_float("X_TEST", 0.55) == 0.55

    def test_int_cade_sul_default(self, monkeypatch):
        monkeypatch.setenv("X_TEST", "# ~300–500")
        assert _get_int("X_TEST", 400) == 400

    def test_commento_con_spazi_iniziali(self, monkeypatch):
        monkeypatch.setenv("X_TEST", "   # commento indentato")
        assert _get("X_TEST", "fallback") == "fallback"

    def test_api_key_residua_non_arriva_al_client(self, monkeypatch):
        """Meglio 'chiave assente' che un 401 con una chiave spazzatura."""
        monkeypatch.setenv("MISTRAL_API_KEY", "# SECRET → impostare come secret dell'environment")
        assert RagConfig.from_env().mistral_api_key == ""


class TestValoriLegittimi:
    def test_valore_normale_passa(self, monkeypatch):
        monkeypatch.setenv("X_TEST", "0.78")
        assert _get_float("X_TEST", 0.55) == 0.78

    def test_valore_con_commento_inline_gia_ripulito_da_dotenv(self, monkeypatch):
        """Con un valore presente dotenv toglie il commento: qui arriva già pulito."""
        monkeypatch.setenv("X_TEST", "0.55")
        assert _get_float("X_TEST", 0.99) == 0.55

    def test_stringa_vuota_vale_assente(self, monkeypatch):
        monkeypatch.setenv("X_TEST", "")
        assert _get_int("X_TEST", 400) == 400

    def test_variabile_non_impostata(self, monkeypatch):
        monkeypatch.delenv("X_TEST", raising=False)
        assert _get("X_TEST", "fallback") == "fallback"

    def test_valore_con_cancelletto_interno_resta_intatto(self, monkeypatch):
        """Il cancelletto conta solo in apertura: un path o un token può contenerlo."""
        monkeypatch.setenv("X_TEST", "chiave#con#cancelletti")
        assert _get("X_TEST") == "chiave#con#cancelletti"


class TestBoolFailLoud:
    """Un flag scritto male deve fare rumore, non spegnere in silenzio il meccanismo.

    ``_get_float``/``_get_int`` esplodono su un valore non interpretabile; un `_get_bool`
    che degradasse a ``False`` renderebbe ``ROUTER_ENABLED=tru`` indistinguibile da uno
    spegnimento voluto. È la classe di guasto di ``SUPPORT_THRESHOLD`` lasciata vuota.
    """

    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", "Yes", "on", "si", "sì", "SÌ"])
    def test_forme_vere(self, monkeypatch, raw):
        monkeypatch.setenv("X_TEST", raw)
        assert _get_bool("X_TEST", False) is True

    @pytest.mark.parametrize("raw", ["0", "false", "FALSE", "No", "off", "Off"])
    def test_forme_false(self, monkeypatch, raw):
        monkeypatch.setenv("X_TEST", raw)
        assert _get_bool("X_TEST", True) is False

    def test_valore_non_riconosciuto_esplode_nominando_la_variabile(self, monkeypatch):
        monkeypatch.setenv("ROUTER_ENABLED", "tru")
        with pytest.raises(ValueError) as exc:
            _get_bool("ROUTER_ENABLED", True)
        assert "ROUTER_ENABLED" in str(exc.value)
        assert "tru" in str(exc.value)

    def test_variabile_assente_usa_il_default(self, monkeypatch):
        monkeypatch.delenv("X_TEST", raising=False)
        assert _get_bool("X_TEST", True) is True
        assert _get_bool("X_TEST", False) is False


def test_router_e_verbatim_hanno_default_espliciti():
    """La guardia nasce SPENTA: prima si misura, poi si tara — come per la soglia IDF."""
    cfg = RagConfig()
    assert cfg.router_enabled is True
    assert cfg.verbatim_enabled is True
    assert cfg.verbatim_min_valid_ratio == 0.0
    assert cfg.verbatim_min_chars == 40
    assert cfg.structured_max_rows == 20


def test_router_disattivabile_da_env(monkeypatch):
    """Serve a rigiocare le run precedenti con la pipeline di prima."""
    monkeypatch.setenv("ROUTER_ENABLED", "0")
    assert RagConfig.from_env().router_enabled is False
    monkeypatch.setenv("ROUTER_ENABLED", "true")
    assert RagConfig.from_env().router_enabled is True


def test_bool_vuoto_vale_il_default(monkeypatch):
    """Stessa trappola già coperta per le soglie: valore vuoto o residuo di commento."""
    monkeypatch.setenv("VERBATIM_ENABLED", "   # da decidere")
    assert RagConfig.from_env().verbatim_enabled is True
