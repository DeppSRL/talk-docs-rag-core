"""C3 — contratto di output strutturato.

Ogni claim porta i numeri di passaggio ``[n]`` che lo sostengono. Il modello cita per
**numero di passaggio** (non per chunk_id, che allucinerebbe): la mappa n→chunk_id la
tiene il codice, e la verifica a valle controlla che ogni ``n`` esista fra i chunk
recuperati.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# JSON schema passato a La Plateforme (response_format json_schema). Deliberatamente
# minimale e byte-stabile: fa parte del prefisso cache-friendly.
STRUCTURED_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "grounded_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Risposta sintetica in italiano con citazioni inline [n].",
                },
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "statement": {"type": "string"},
                            "passages": {"type": "array", "items": {"type": "integer"}},
                        },
                        "required": ["statement", "passages"],
                    },
                },
            },
            "required": ["answer", "claims"],
        },
    },
}


class Claim(BaseModel):
    statement: str
    passages: list[int] = Field(default_factory=list)


class StructuredAnswer(BaseModel):
    answer: str
    claims: list[Claim] = Field(default_factory=list)
