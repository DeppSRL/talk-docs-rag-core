"""Tabella strutturata dei documenti del corpus, per le domande aggregative.

DuckDB **in memoria**, popolata dal manifest alla costruzione della pipeline: 511 righe,
nessun file di database da versionare o invalidare. Non serve alla performance — serve
perché la query eseguita è **l'artefatto di citazione**: una risposta aggregativa non si
cita con un ``[n]``, si cita con la query più le righe che concorrono al totale.

I campi ``codice``/``numero``/``anno``/``comitato`` vengono da
``ingest.parsers.metadati_delibera``: la regola sta lì e non si riscrive qui.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from ingest.parsers import metadati_delibera

DDL = """
CREATE TABLE documenti (
    path TEXT,
    title TEXT,
    content_hash TEXT,
    n_chunks INTEGER,
    codice TEXT,
    numero INTEGER,
    anno INTEGER,
    comitato TEXT,
    is_delibera BOOLEAN
)
"""


class StructuredStore:
    """Sola lettura. La sorgente di verità resta il manifest scritto dall'ingest."""

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn

    @classmethod
    def from_manifest(cls, manifest: dict) -> StructuredStore:
        conn = duckdb.connect(":memory:")
        conn.execute(DDL)
        righe = []
        for f in manifest.get("files", []):
            meta = metadati_delibera(Path(f["path"]).stem)
            righe.append(
                (
                    f["path"],
                    f.get("title", ""),
                    f.get("content_hash", ""),
                    int(f.get("n_chunks", 0)),
                    meta["codice"] if meta else None,
                    meta["numero"] if meta else None,
                    meta["anno"] if meta else None,
                    meta["comitato"] if meta else None,
                    meta is not None,
                )
            )
        if righe:
            conn.executemany("INSERT INTO documenti VALUES (?,?,?,?,?,?,?,?,?)", righe)
        return cls(conn)

    @classmethod
    def from_path(cls, path: str | Path) -> StructuredStore | None:
        """``None`` se il manifest non esiste: senza corpus indicizzato il router va spento,
        non deve far fallire la costruzione della pipeline."""
        p = Path(path)
        if not p.exists():
            return None
        return cls.from_manifest(json.loads(p.read_text(encoding="utf-8")))

    def query(self, sql: str, params: list) -> list[dict]:
        cur = self.conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]
