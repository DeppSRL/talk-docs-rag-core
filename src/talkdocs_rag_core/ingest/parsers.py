"""C1 — parsing dei formati del corpus (pdf/md/html/txt) → testo pulito + titolo.

Ogni parser ritorna ``(text, title)``. Il titolo è euristico (prima intestazione md,
``<title>``/primo ``<h1>`` html, nome file altrimenti) e finisce nei metadati del chunk.

Caso speciale del corpus delibere: il nome file porta il codice ``E{YY}{NNNN}``, unica
fonte di metadati disponibile (il DB ``db-delibere`` non è una dipendenza del PoC). Da
lì si ricava un titolo citabile — vedi ``titolo_delibera``.
"""

from __future__ import annotations

import re
from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".html", ".htm", ".txt"}

# Codice delibera: 'E' + anno a 2 cifre + numero a 4 cifre, eventualmente seguito da
# suffissi di lavorazione ('E190004finale_1', 'E200067_Patuanelli').
CODICE_DELIBERA = re.compile(r"^[Ee](\d{2})(\d{4})")

# L'archivio parte dal 1967: YY>=67 è Novecento, YY<67 è Duemila.
PIVOT_SECOLO = 67

# Il comitato è CIPE fino al 2020, CIPESS (…e lo sviluppo sostenibile) dal 2021.
PRIMO_ANNO_CIPESS = 2021


def titolo_delibera(stem: str) -> str | None:
    """``'E210075'`` → ``'Delibera CIPESS n. 75/2021'``. ``None`` se non è un codice.

    Il titolo entra nelle citazioni: deve essere deterministico e nominare il comitato
    con la denominazione vigente nell'anno dell'atto.
    """
    m = CODICE_DELIBERA.match(stem)
    if not m:
        return None
    yy, numero = int(m.group(1)), int(m.group(2))
    anno = 1900 + yy if yy >= PIVOT_SECOLO else 2000 + yy
    comitato = "CIPESS" if anno >= PRIMO_ANNO_CIPESS else "CIPE"
    return f"Delibera {comitato} n. {numero}/{anno}"


def parse_file(path: Path) -> tuple[str, str]:
    """Dispatch per estensione. Ritorna (testo, titolo)."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(path)
    if suffix in (".md", ".markdown"):
        return _parse_markdown(path)
    if suffix in (".html", ".htm"):
        return _parse_html(path)
    if suffix == ".txt":
        return _parse_txt(path)
    raise ValueError(f"Formato non supportato: {path.suffix} ({path})")


def _parse_txt(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text, titolo_delibera(path.stem) or path.stem


def _parse_markdown(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    # Titolo: primo heading '# ...'
    title = path.stem
    for line in text.splitlines():
        m = re.match(r"^#\s+(.+)", line.strip())
        if m:
            title = m.group(1).strip()
            break
    return text, title


def _parse_html(path: Path) -> tuple[str, str]:
    from bs4 import BeautifulSoup

    raw = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")

    # Titolo: <title> o primo <h1>
    title = path.stem
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    elif soup.h1 and soup.h1.get_text(strip=True):
        title = soup.h1.get_text(strip=True)

    # Rimuovi script/style, estrai testo preservando i break di paragrafo
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return text, title


def _parse_pdf(path: Path) -> tuple[str, str]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages)

    # Titolo: metadati PDF se presenti, altrimenti nome file
    title = path.stem
    try:
        meta_title = reader.metadata.title if reader.metadata else None
        if meta_title:
            title = str(meta_title).strip()
    except Exception:
        pass
    return text, title
