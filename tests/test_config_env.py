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

from config import RagConfig, _get, _get_float, _get_int


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
