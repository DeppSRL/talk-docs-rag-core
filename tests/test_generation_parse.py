"""Degrado del parsing quando il modello non produce JSON valido.

Difetto misurato su `va-05` (run eval-20260805T111224Z, domanda vaga «Cosa hanno combinato
con l'ANAS?»): la risposta ha sbattuto contro il tetto `max_output_tokens=512`, il JSON è
rimasto tagliato a metà dell'array `claims`, `json.loads` è fallito e il fallback riversava il
**JSON grezzo** — graffe, virgolette, `"claims": [` — nel campo mostrato all'utente. Peggio: il
regex dei marcatori trovava comunque i `[1][3]` dentro quel testo, quindi le metriche
automatiche registravano «5 citazioni valide, 0 invalide» su un output inutilizzabile.
"""

import json

from talkdocs_rag_core.rag.generation import _salvage_answer


def test_json_valido_non_passa_dal_recupero():
    """Il recupero serve solo al ramo degradato; qui si controlla che sappia comunque
    estrarre la prosa senza portarsi dietro la sintassi."""
    raw = json.dumps({"answer": "Il CIPESS è presieduto dal Presidente [1].", "claims": []})
    assert _salvage_answer(raw) == "Il CIPESS è presieduto dal Presidente [1]."


def test_recupera_la_prosa_da_json_troncato():
    raw = (
        '{\n  "answer": "L\'ANAS è stata coinvolta in accordi per accelerare le opere [1][3].",\n'
        '  "claims": [\n    {\n      "statement": "L\'ANAS accelera le procedure.",\n'
        '      "passages": ['
    )
    fuori = _salvage_answer(raw)
    assert fuori == "L'ANAS è stata coinvolta in accordi per accelerare le opere [1][3]."
    for spia in ('"claims"', "{", "passages"):
        assert spia not in fuori


def test_recupera_anche_se_answer_stesso_e_tagliato():
    """Taglio *dentro* la prosa: si tiene ciò che c'è, senza chiusura di virgolette."""
    raw = '{"answer": "Il Comitato ha approvato l\'aggiornamento del contratto'
    assert _salvage_answer(raw) == "Il Comitato ha approvato l'aggiornamento del contratto"


def test_scioglie_le_sequenze_di_escape():
    raw = '{"answer": "Riparto \\"quote premiali\\" 2020:\\neuro 295.178.000", "claims": ['
    fuori = _salvage_answer(raw)
    assert '"quote premiali"' in fuori
    assert "\\n" not in fuori
    assert "\n" in fuori


def test_senza_campo_answer_restituisce_il_grezzo():
    """Meglio un grezzo visibile che una stringa vuota che nasconde il guasto."""
    assert _salvage_answer("non è affatto JSON") == "non è affatto JSON"
    assert _salvage_answer('{"altro": "campo"}') == '{"altro": "campo"}'


def test_stringa_vuota():
    assert _salvage_answer("") == ""
