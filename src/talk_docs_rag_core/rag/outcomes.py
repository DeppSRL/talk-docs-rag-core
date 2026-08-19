"""Esiti dei due meccanismi nuovi, tenuti fuori da ``RagResult``.

``RagResult`` ha già venticinque campi: appenderne altri dodici lo renderebbe illeggibile.
Audit ed eval leggono ``res.structured.sql`` e ``res.verbatim.valid_ratio`` invece di
navigare un dizionario generico.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StructuredOutcome:
    """Il ramo aggregativo. La citazione di questa risposta è ``sql`` + ``rows``."""

    intent: str
    sql: str
    params: list
    rows: list[dict]
    n_rows: int
    # Il valore scalare che la risposta asserisce: il numero contro cui l'eval confronta
    # `expected_value`. `None` quando la risposta non è un numero singolo (distribuzione).
    computed_value: int | None = None
    # {"count": 93, "max_numero": 95, "gap": 2} — vuoto quando il controllo non si applica.
    completeness: dict = field(default_factory=dict)
    cited_doc_ids: list[str] = field(default_factory=list)


@dataclass
class VerbatimOutcome:
    """La verifica degli span. Registrata SEMPRE, anche a guardia spenta: le soglie si
    ritarano su run passate senza rigiocarle, com'è già per ``abstention_signal``."""

    n_claims: int
    n_valid: int
    n_misattributed: int
    n_not_found: int
    n_too_short: int
    valid_ratio: float | None = None  # None se n_claims == 0: il rapporto non è definito
    span_boundary_ratio: float | None = None
    per_claim: list[dict] = field(default_factory=list)
