from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import List, Optional, Tuple


MAX_INSTRUCTIONS_CHARS = 30_000


def read_series_instructions(path: Optional[str]) -> str:
    """Read an optional series terminology/instructions Markdown file."""
    if not path:
        return ""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Instructions Markdown file not found: {path}")
    return file_path.read_text(encoding="utf-8-sig").strip()[:MAX_INSTRUCTIONS_CHARS]


def markdown_terminology_pairs(markdown: str) -> List[Tuple[str, str]]:
    """Extract conservative term/translation pairs from Markdown.

    Supported forms are Markdown tables and bullets such as ``- term -> translation``.
    Free-form prose is intentionally ignored when building a DeepL glossary.
    """
    pairs: List[Tuple[str, str]] = []
    seen = set()
    for raw_line in (markdown or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or set(line) <= {"|", "-", ":", " ", "\t"}:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) >= 2 and not all(re.fullmatch(r"[-:\s]+", cell or "") for cell in cells[:2]):
            candidate = (cells[0], cells[1])
        else:
            match = re.match(r"^(?:[-*+]\s*)?(.+?)\s*(?:->|=>|:|→)\s*(.+?)\s*$", line)
            if not match:
                continue
            candidate = (match.group(1).strip(" `"), match.group(2).strip(" `"))
        if not all(candidate) or candidate[0].lower() in {"source", "term", "english", "original"}:
            continue
        if candidate not in seen:
            pairs.append(candidate)
            seen.add(candidate)
    return pairs


def terminology_pairs_csv(markdown: str) -> str:
    pairs = markdown_terminology_pairs(markdown)
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(pairs)
    return buf.getvalue().strip()
