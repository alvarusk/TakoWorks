#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
TakoWorks - Corrector (ASS -> LanguageTool -> HTML)

v0.473 (repack)
Principales funciones:
- Regla CARTEL:
  - CARTEL si Name/Actor contiene "CARTEL" (case-insensitive)
  - O si Style empieza por "Cart_" (case-insensitive)
- Reordena [Events] en dos bloques:
  1) CARTEL (ordenados cronológicamente por Start/End)
  2) resto (orden original)
- Convierte "-" -> "—" SOLO en diálogos de 2 personajes (campo Text):
  - un "-" al inicio (tras posibles tags/espacios)
  - y otro "-" tras "\\N" o "\\n" (tras posibles tags/espacios)

LanguageTool:
- /v2/check con opciones: level, motherTongue, enabledRules/disabledRules, enabledCategories/disabledCategories, enabledOnly.
- Reporte HTML con filtros y botón "➕ Diccionario" para misspellings:
  - En modo normal (file://): intenta 127.0.0.1:8765; si no, se queda "en cola"
  - En modo --serve: el script sirve el HTML por localhost y el botón añade directamente a LT vía /api/add-word

Outputs (por defecto solo HTML):
- *_cartelFirst.ass
- *_dialogo.txt
- *_dialogo_merged.txt
- *_LT_raw.json
- *_LT_report.html
"""

from __future__ import annotations

import argparse
import html
import json
import os
import tempfile
import re
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
import unicodedata
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PUNCT_ENDINGS = (".", "…", ":", "?", "!")
TAG_RE = re.compile(r"\{[^}]*\}")
ASS_NL_RE = re.compile(r"\\[Nn]")
ASS_HARD_SPACE_RE = re.compile(r"\\h")

# Two-speaker dash rules (in ASS Text):
_START_DASH_RE = re.compile(r"^(?P<prefix>(?:\{[^}]*\})*\s*)-(?=\s*\S)")
_AFTER_NL_DASH_RE = re.compile(r"(\\[Nn])(?P<mid>(?:\{[^}]*\})*\s*)-(?=\s*\S)")

@dataclass
class Event:
    kind: str
    fields: List[str]
    style: str
    name: str
    text: str
    is_cartel: bool
    start_s: Optional[float]
    end_s: Optional[float]
    original_index: int

@dataclass
class Settings:
    # Auth / endpoint
    lt_base_url: str = "https://api.languagetoolplus.com/v2"
    lt_username: str = ""
    lt_api_key: str = ""

    # Check params
    language: str = "es-ES"
    level: str = "default"  # "default" | "picky"
    mother_tongue: str = ""  # e.g. "es"
    enabled_only: bool = False

    # Advanced filters (comma-separated IDs)
    enabled_rules: str = ""
    disabled_rules: str = ""
    enabled_categories: str = ""
    disabled_categories: str = ""

    # HTML defaults (omit by default)
    suppress_rules: str = ""       # comma-separated rule IDs to hide by default in HTML
    suppress_categories: str = ""  # comma-separated category IDs to hide by default in HTML

    # UX
    open_html: bool = True

    # Personal dictionary (LT Plus)
    apply_personal_dict_filter: bool = True
    personal_dict_cache_ttl_s: int = 0  # 0 = always refresh
    personal_dict_name: str = ""  # optional dict name

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        return f.read()

def _write_text(path: str, s: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)

def _as_csv(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        cleaned = [str(x).strip() for x in value if str(x).strip()]
        return ",".join(cleaned)
    return str(value).strip()

def _csv_set(s: str) -> Set[str]:
    if not s:
        return set()
    return {p.strip() for p in str(s).split(",") if p.strip()}

def _load_settings(settings_path: Optional[str], script_dir: str, input_dir: str) -> Settings:
    candidates: List[str] = []
    if settings_path:
        candidates.append(settings_path)
    candidates.append(os.path.join(script_dir, "corrector_settings.json"))
    candidates.append(os.path.join(input_dir, "corrector_settings.json"))

    data: Dict[str, Any] = {}
    used = None
    for p in candidates:
        if p and os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                used = p
                break
            except Exception as e:
                print(f"[WARN] No se pudo leer settings '{p}': {e}")

    s = Settings()
    if data:
        s.lt_base_url = str(data.get("lt_base_url", s.lt_base_url)).rstrip("/")
        s.lt_username = str(data.get("lt_username", s.lt_username)).strip()
        s.lt_api_key = str(data.get("lt_api_key", s.lt_api_key)).strip()

        s.language = str(data.get("language", s.language)).strip() or s.language
        s.level = str(data.get("level", s.level)).strip() or s.level
        s.mother_tongue = str(data.get("mother_tongue", data.get("motherTongue", s.mother_tongue))).strip()
        s.enabled_only = bool(data.get("enabled_only", data.get("enabledOnly", s.enabled_only)))

        s.enabled_rules = _as_csv(data.get("enabled_rules", data.get("enabledRules")))
        s.disabled_rules = _as_csv(data.get("disabled_rules", data.get("disabledRules")))
        s.enabled_categories = _as_csv(data.get("enabled_categories", data.get("enabledCategories")))
        s.disabled_categories = _as_csv(data.get("disabled_categories", data.get("disabledCategories")))

        s.suppress_rules = _as_csv(data.get("suppress_rules", data.get("suppressRules")))
        s.suppress_categories = _as_csv(data.get("suppress_categories", data.get("suppressCategories")))

        
        s.apply_personal_dict_filter = bool(data.get("apply_personal_dict_filter", data.get("applyPersonalDictFilter", s.apply_personal_dict_filter)))
        try:
            s.personal_dict_cache_ttl_s = int(data.get("personal_dict_cache_ttl_s", data.get("personalDictCacheTtlS", s.personal_dict_cache_ttl_s)) or 0)
        except Exception:
            s.personal_dict_cache_ttl_s = s.personal_dict_cache_ttl_s
        s.personal_dict_name = str(data.get("personal_dict_name", data.get("personalDictName", s.personal_dict_name))).strip()
        if "open_html" in data:
            s.open_html = bool(data.get("open_html"))

    if used:
        print(f"[+] Settings cargados desde: {used}")
    else:
        print("[i] No se encontró settings; usaré valores por defecto (sin credenciales).")

    # Debug útil: confirmar endpoint y si el /check va autenticado (necesario para que aplique el Diccionario Personal)
    print(f"[i] LT base URL: {s.lt_base_url}")
    auth_on = bool(s.lt_username and s.lt_api_key)
    print(f"[i] LT auth: {'ON' if auth_on else 'OFF'}")
    if auth_on and 'languagetool.org' in s.lt_base_url and 'languagetoolplus' not in s.lt_base_url:
        print("[WARN] Estás usando api.languagetool.org (free). El diccionario personal de tu cuenta NO se aplicará; usa api.languagetoolplus.com.")

    return s

def _parse_events_section(ass_text: str) -> Tuple[str, str, str]:
    t = ass_text.replace("\r\n", "\n").replace("\r", "\n")
    m = re.search(r"(?im)^\[events\]\s*$", t)
    if not m:
        return t, "", ""
    start = m.start()
    m2 = re.search(r"(?im)^\[[^\]]+\]\s*$", t[m.end():])
    if m2:
        end = m.end() + m2.start()
        return t[:start], t[start:end], t[end:]
    return t[:start], t[start:], ""

def _parse_format_line(events_block: str) -> List[str]:
    for ln in events_block.split("\n"):
        if ln.strip().lower().startswith("format:"):
            fmt = ln.split(":", 1)[1]
            return [f.strip() for f in fmt.split(",")]
    return []

def _indices(fmt_fields: List[str]) -> Dict[str, int]:
    def idx_of(name: str) -> int:
        for i, f in enumerate(fmt_fields):
            if f.strip().lower() == name.lower():
                return i
        return -1

    if not fmt_fields:
        return {"Start": 1, "End": 2, "Style": 3, "Name": 4, "Text": 9}

    out = {
        "Start": idx_of("Start"),
        "End": idx_of("End"),
        "Style": idx_of("Style"),
        "Name": idx_of("Name"),
        "Text": idx_of("Text"),
    }
    if out["Start"] < 0: out["Start"] = 1
    if out["End"] < 0: out["End"] = 2
    if out["Name"] < 0: out["Name"] = 4
    if out["Text"] < 0: out["Text"] = len(fmt_fields) - 1
    return out

def _split_event_fields(line: str, fmt_fields: List[str]) -> Optional[Tuple[str, List[str]]]:
    stripped = line.lstrip()
    if not (stripped.lower().startswith("dialogue:") or stripped.lower().startswith("comment:")):
        return None
    kind = "Dialogue" if stripped.lower().startswith("dialogue:") else "Comment"
    payload = stripped.split(":", 1)[1].lstrip()
    if not fmt_fields:
        parts = payload.split(",", 9)
        return kind, parts
    parts = payload.split(",", len(fmt_fields) - 1)
    if len(parts) < len(fmt_fields):
        parts += [""] * (len(fmt_fields) - len(parts))
    return kind, parts

def parse_ass_time(t: str) -> Optional[float]:
    t = (t or "").strip()
    m = re.match(r"^(?P<h>\d+):(?P<m>\d{1,2}):(?P<s>\d{1,2})[.](?P<cs>\d{1,2})$", t)
    if not m:
        return None
    h = int(m.group("h"))
    mm = int(m.group("m"))
    ss = int(m.group("s"))
    cs = int(m.group("cs"))
    return h * 3600.0 + mm * 60.0 + ss + cs / 100.0

def convert_two_speaker_dashes_to_em_dash(text_field: str) -> str:
    if not text_field:
        return text_field
    if not _START_DASH_RE.search(text_field):
        return text_field
    if not _AFTER_NL_DASH_RE.search(text_field):
        return text_field

    def repl_start(m: re.Match) -> str:
        return m.group("prefix") + "—"

    def repl_after(m: re.Match) -> str:
        return m.group(1) + m.group("mid") + "—"

    out = _START_DASH_RE.sub(repl_start, text_field, count=1)
    out = _AFTER_NL_DASH_RE.sub(repl_after, out)
    return out

def render_event(kind: str, fields: List[str]) -> str:
    return f"{kind}: " + ",".join(fields)

def is_cartel_event(name_val: str, style_val: str) -> bool:
    if name_val and ("cartel" in name_val.lower()):
        return True
    if style_val and style_val.lower().startswith("cart_"):
        return True
    return False

def reorder_cartel_in_events_and_apply_dashes(ass_text: str) -> Tuple[str, List[Event], int]:
    before, events_block, after = _parse_events_section(ass_text)
    if not events_block:
        return ass_text, [], 0

    fmt_fields = _parse_format_line(events_block)
    idx = _indices(fmt_fields)
    style_idx = idx["Style"]
    name_idx = idx["Name"]
    text_idx = idx["Text"]
    start_idx = idx["Start"]
    end_idx = idx["End"]

    lines = events_block.split("\n")
    prefix: List[str] = []
    suffix: List[str] = []

    events: List[Event] = []
    seen_event = False
    event_counter = 0

    for ln in lines:
        is_event = bool(re.match(r"(?i)^\s*(dialogue|comment):", ln))
        if not seen_event and not is_event:
            prefix.append(ln)
            continue

        if is_event:
            seen_event = True
            parsed = _split_event_fields(ln, fmt_fields)
            if not parsed:
                suffix.append(ln)
                continue
            kind, fields = parsed

            # Two-speaker dash conversion (Text)
            if kind == "Dialogue" and 0 <= text_idx < len(fields):
                fields[text_idx] = convert_two_speaker_dashes_to_em_dash(fields[text_idx])

            style_val = fields[style_idx] if (0 <= style_idx < len(fields)) else ""
            name_val = fields[name_idx] if (0 <= name_idx < len(fields)) else ""
            text_val = fields[text_idx] if (0 <= text_idx < len(fields)) else (fields[-1] if fields else "")
            start_s = parse_ass_time(fields[start_idx]) if (0 <= start_idx < len(fields)) else None
            end_s = parse_ass_time(fields[end_idx]) if (0 <= end_idx < len(fields)) else None

            cartel = is_cartel_event(name_val, style_val)

            events.append(Event(
                kind=kind,
                fields=fields,
                style=style_val,
                name=name_val,
                text=text_val,
                is_cartel=cartel,
                start_s=start_s,
                end_s=end_s,
                original_index=event_counter
            ))
            event_counter += 1
        else:
            suffix.append(ln)

    cartel_events = [e for e in events if e.is_cartel]
    other_events = [e for e in events if not e.is_cartel]

    def cartel_key(e: Event) -> Tuple[float, float, int]:
        s = e.start_s if e.start_s is not None else 1e18
        en = e.end_s if e.end_s is not None else 1e18
        return (s, en, e.original_index)

    cartel_sorted = sorted(cartel_events, key=cartel_key)

    new_events = cartel_sorted + other_events

    rebuilt_lines = (
        prefix
        + [render_event(e.kind, e.fields) for e in new_events]
        + suffix
    )
    rebuilt = "\n".join(rebuilt_lines)
    return before + rebuilt + after, new_events, len(cartel_sorted)

def clean_ass_text(s: str) -> str:
    s = TAG_RE.sub("", s)
    s = ASS_HARD_SPACE_RE.sub(" ", s)
    s = ASS_NL_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s, flags=re.UNICODE).strip()
    return s

def extract_dialogue_lines(events: List[Event]) -> Tuple[List[str], List[bool]]:
    out_lines: List[str] = []
    flags: List[bool] = []
    for ev in events:
        if ev.kind != "Dialogue":
            continue
        cleaned = clean_ass_text(ev.text)
        if cleaned == "":
            continue
        out_lines.append(cleaned)
        flags.append(bool(ev.is_cartel))
    return out_lines, flags

def merge_lines(lines: List[str], is_cartel: List[bool]) -> List[str]:
    merged: List[str] = []
    buf = ""
    for ln, cartel in zip(lines, is_cartel):
        if cartel:
            if buf.strip():
                merged.append(buf.strip())
                buf = ""
            merged.append(ln.strip())
            continue

        if not buf:
            buf = ln.strip()
        else:
            buf = (buf.rstrip() + " " + ln.strip()).strip()

        if buf.endswith(PUNCT_ENDINGS):
            merged.append(buf.strip())
            buf = ""
    if buf.strip():
        merged.append(buf.strip())
    return merged


def split_long_line_to_fit(line: str, max_chars: int) -> List[str]:
    """
    Ensure no single line exceeds max_chars by splitting it into smaller pieces.
    Tries to split at natural boundaries (sentence punctuation/spaces) near max_chars.
    """
    line = (line or "").strip()
    if not line:
        return [""]
    if len(line) <= max_chars:
        return [line]

    seps = ["。", "！", "？", ".", "!", "?", "…", ";", ":", "—", " ", ","]
    parts: List[str] = []
    s = line
    lookback = 400  # search window for a nicer split point
    while len(s) > max_chars:
        cut = max_chars
        window = s[:max_chars]
        start = max(0, max_chars - lookback)
        tail = window[start:]
        best = -1
        for sep in seps:
            i = tail.rfind(sep)
            if i != -1:
                best = max(best, start + i + 1)  # include separator
        if best > 0:
            cut = best
        part = s[:cut].strip()
        if part:
            parts.append(part)
        s = s[cut:].lstrip()
        if not s:
            break
    if s.strip():
        parts.append(s.strip())
    return parts

def split_into_chunks(lines: List[str], max_chars: int) -> List[List[str]]:
    chunks: List[List[str]] = []
    cur: List[str] = []
    cur_len = 0

    for ln in lines:
        for part in split_long_line_to_fit(ln, max_chars):
            # Preserve empty lines as separators
            if part == "" and not cur:
                continue
            add_len = len(part) + (1 if cur else 0)
            if cur and (cur_len + add_len) > max_chars:
                chunks.append(cur)
                cur = [part]
                cur_len = len(part)
            else:
                cur.append(part)
                cur_len += add_len

    if cur:
        chunks.append(cur)
    return chunks

def lt_check(text: str, settings: Settings) -> Dict[str, Any]:
    """Calls /v2/check (form-urlencoded)."""
    base = settings.lt_base_url.rstrip("/")
    url = base + "/check"
    data: Dict[str, Any] = {"text": text, "language": settings.language}

    # Premium auth (optional)
    if settings.lt_username and settings.lt_api_key:
        data["username"] = settings.lt_username
        data["apiKey"] = settings.lt_api_key

    if settings.level and settings.level != "default":
        data["level"] = settings.level  # "picky"

    if settings.mother_tongue:
        data["motherTongue"] = settings.mother_tongue

    if settings.enabled_rules:
        data["enabledRules"] = settings.enabled_rules
    if settings.disabled_rules:
        data["disabledRules"] = settings.disabled_rules
    if settings.enabled_categories:
        data["enabledCategories"] = settings.enabled_categories
    if settings.disabled_categories:
        data["disabledCategories"] = settings.disabled_categories

    if settings.enabled_only:
        data["enabledOnly"] = "true"

    encoded = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(url, data=encoded, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded; charset=utf-8")
    req.add_header("User-Agent", "TakoWorks-Corrector/0.473")

    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)

def lt_add_word(word: str, settings: Settings, dict_name: str = "") -> Tuple[bool, str]:
    """
    Add a single word to LT Plus personal dictionary via /v2/words/add.
    Returns (ok, detail).
    """
    if not settings.lt_username or not settings.lt_api_key:
        return False, "missing_credentials"

    base = settings.lt_base_url.rstrip("/")
    url = base + "/words/add"
    data: Dict[str, Any] = {
        "word": word,
        "username": settings.lt_username,
        "apiKey": settings.lt_api_key,
    }
    if dict_name:
        data["dict"] = dict_name

    encoded = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(url, data=encoded, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded; charset=utf-8")
    req.add_header("User-Agent", "TakoWorks-Corrector/0.473")

    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8", errors="replace")

    try:
        js = json.loads(raw)
        # LT Plus returns {"added": true} when newly added and {"added": false} when it already existed.
        if "added" in js:
            return True, "added" if bool(js.get("added")) else "already_present"
        if bool(js.get("ok")):
            return True, "ok"
        return False, raw
    except Exception:
        if "true" in raw.lower() or "added" in raw.lower() or "ok" in raw.lower():
            return True, raw
        return False, raw




def _norm_word(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    # Normalize Unicode (e.g., full-width) and collapse spaces
    s = unicodedata.normalize("NFKC", s)
    s = s.casefold()  # case-insensitive matching
    s = re.sub(r"\s+", " ", s).strip()
    # Trim common surrounding punctuation so tokens like "Akari," match "Akari"
    s = re.sub(r"^[\"'“”‘’«»¿¡\(\)\[\]{}<>.,;:!?…—–-]+", "", s)
    s = re.sub(r"[\"'“”‘’«»¿¡\(\)\[\]{}<>.,;:!?…—–-]+$", "", s)
    return s.strip()

# --- Local fallback word list (immediate effect + resilience) ---

def _get_local_dict_path(script_dir: str, settings: "Settings") -> str:
    """Per-user local wordlist stored in AppData so it works across multiple copies (TakoWorks/SE/Aegisub)."""
    user = re.sub(r"[^a-zA-Z0-9]", "_", (settings.lt_username or "nouser"))[:24]
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    base = os.path.join(base, "TakoWorks", "Corrector")
    try:
        os.makedirs(base, exist_ok=True)
    except Exception:
        base = script_dir
    return os.path.join(base, f"local_words_{user}.txt")


def load_local_word_set(path: str) -> set[str]:
    out: set[str] = set()
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                for ln in f:
                    w = _norm_word(ln)
                    if w:
                        out.add(w)
    except Exception:
        pass
    return out


def load_local_words_raw(path: str) -> list[str]:
    """Load local word list preserving original casing (for snapshot/debug)."""
    out: list[str] = []
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                for ln in f:
                    w = (ln or "").strip()
                    if w:
                        out.append(w)
    except Exception:
        pass
    return out


def _get_snapshot_dict_path(script_dir: str, settings: "Settings") -> str:
    """Per-user snapshot path for downloaded LT personal dictionary (stored in AppData)."""
    user = re.sub(r"[^a-zA-Z0-9]", "_", (settings.lt_username or "nouser"))[:24]
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    base = os.path.join(base, "TakoWorks", "Corrector")
    try:
        os.makedirs(base, exist_ok=True)
    except Exception:
        base = script_dir
    return os.path.join(base, f"lt_personal_snapshot_{user}.txt")



def append_local_word(path: str, word: str) -> None:
    """Append a word to the local word list, preserving original casing.
    Deduping is done case-insensitively using _norm_word().
    """
    raw = (word or "").strip().replace("\r", "").replace("\n", "")
    key = _norm_word(raw)
    if not key:
        return
    try:
        # Avoid duplicates with a quick scan (file is usually small)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                for ln in f:
                    if _norm_word(ln) == key:
                        return
        with open(path, "a", encoding="utf-8") as f:
            f.write(raw + "\n")
    except Exception:
        pass

def lt_get_words(settings: Settings, dict_name: str = "", limit: int = 20000) -> List[str]:
    """Fetch user's personal dictionary words from LT Plus via /v2/words (paged by offset/limit)."""
    if not settings.lt_username or not settings.lt_api_key:
        return []

    base = settings.lt_base_url.rstrip("/")
    url = base + "/words"

    out: List[str] = []
    offset = 0
    while True:
        params = {
            "username": settings.lt_username,
            "apiKey": settings.lt_api_key,
            "offset": str(offset),
            "limit": str(limit),
        }
        if dict_name:
            params["dicts"] = dict_name

        full = url + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(full, method="GET")
        req.add_header("User-Agent", "TakoWorks-Corrector/0.473")

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            # Some deployments cap "limit"; if so, retry with a smaller page size once.
            if e.code in (400, 413) and limit > 2000:
                return lt_get_words(settings, dict_name=dict_name, limit=2000)
            raise

        js = json.loads(raw)
        words = js.get("words") or []
        if not isinstance(words, list):
            break
        out.extend([str(w) for w in words if w])
        got = len(words)
        if got == 0:
            break
        # Some deployments cap the actual returned page size below 'limit'.
        # Advance by what we actually received so we don't skip entries.
        offset += got

    return out


def load_personal_word_set(settings: Settings, cache_path: str) -> Set[str]:
    """Load personal words (optionally cached) and return a normalized set."""
    ttl = int(getattr(settings, "personal_dict_cache_ttl_s", 0) or 0)
    dict_name = getattr(settings, "personal_dict_name", "") or ""

    # Try cache
    if ttl > 0 and os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            ts = float(cached.get("ts", 0))
            if ts and (time.time() - ts) <= ttl:
                words = cached.get("words") or []
                return { _norm_word(w) for w in words if _norm_word(w) }
        except Exception:
            pass

    words = lt_get_words(settings, dict_name=dict_name)
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"ts": time.time(), "words": words, "dict": dict_name}, f, ensure_ascii=False)
    except Exception:
        pass

    return { _norm_word(w) for w in words if _norm_word(w) }


def filter_rows_by_personal_dict(rows: List[Dict[str, Any]], personal_set: Set[str]) -> List[Dict[str, Any]]:
    if not personal_set:
        return rows
    out = []
    for r in rows:
        token = _norm_word(str(r.get("token", "")))
        rid = str(r.get("rule_id", "") or "").strip()
        is_misspelling = bool(r.get("is_misspelling", False))
        if token and token in personal_set and (is_misspelling or rid == "PERSONALIZED_SPELLER_RULE"):
            continue
        out.append(r)
    return out


def map_matches_to_rows(chunk_lines: List[str], lt_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert LT matches to rows for HTML.
    """
    rows: List[Dict[str, Any]] = []
    matches = lt_json.get("matches", []) or []

    # Line start offsets in chunk text (joined with '\n')
    starts: List[int] = []
    pos = 0
    for i, ln in enumerate(chunk_lines):
        starts.append(pos)
        pos += len(ln)
        if i < len(chunk_lines) - 1:
            pos += 1

    def find_line_index(offset: int) -> int:
        idx = 0
        for i in range(len(starts)):
            if starts[i] <= offset:
                idx = i
            else:
                break
        return idx

    for match in matches:
        offset = int(match.get("offset", 0))
        length = int(match.get("length", 0))
        rule = match.get("rule", {}) or {}
        category = rule.get("category", {}) or {}
        if not category:
            category = match.get("category", {}) or {}

        cat_id = str(category.get("id", "") or "").strip()
        rule_id = str(rule.get("id", "") or "").strip()
        issue_type = str(rule.get("issueType", "") or "").strip()

        message = match.get("message", "") or ""
        short = match.get("shortMessage", "") or ""
        desc = rule.get("description", "") or ""
        err_title = short.strip() or desc.strip() or "Error"

        line_i = find_line_index(offset)
        line = chunk_lines[line_i] if 0 <= line_i < len(chunk_lines) else ""

        line_start = starts[line_i] if 0 <= line_i < len(starts) else 0
        in_line_off = max(0, offset - line_start)
        in_line_end = min(len(line), in_line_off + max(0, length))

        token = ""
        if 0 <= in_line_off < in_line_end <= len(line):
            token = line[in_line_off:in_line_end].strip()

        safe_line = html.escape(line)
        if 0 <= in_line_off < in_line_end <= len(line):
            pre = html.escape(line[:in_line_off])
            mid = html.escape(line[in_line_off:in_line_end])
            post = html.escape(line[in_line_end:])
            safe_line = pre + "<mark>" + mid + "</mark>" + post

        reps = match.get("replacements", []) or []
        sugg = [r.get("value", "") for r in reps if r.get("value")]
        if sugg:
            message2 = message.strip()
            message2 += " " if message2 else ""
            message2 += "Sugerencias: " + ", ".join(sugg[:8])
        else:
            message2 = message

        rows.append({
            "line_html": safe_line,
            "error": html.escape(err_title),
            "comment": html.escape(message2),
            "rule_id": rule_id,
            "cat_id": cat_id,
            "issue_type": issue_type,
            "token": token,
            "is_misspelling": (issue_type.lower() == "misspelling"),
        })
    return rows

def build_html_report(title: str,
                      rows: List[Dict[str, Any]],
                      meta: Dict[str, str],
                      suppress_rules: Set[str],
                      suppress_categories: Set[str]) -> str:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    total = len(rows)
    meta_lines = "".join(f"<li><b>{html.escape(k)}:</b> {html.escape(v)}</li>" for k, v in meta.items())

    cats: Set[str] = set()
    types: Set[str] = set()
    suppressed_count = 0
    for r in rows:
        cid = (r.get("cat_id", "") or "").strip()
        if cid:
            cats.add(cid)
        t = (r.get("issue_type", "") or "").strip()
        if t:
            types.add(t)
        rid = (r.get("rule_id", "") or "").strip()
        if (rid and rid in suppress_rules) or (cid and cid in suppress_categories):
            suppressed_count += 1

    cat_options = ['<option value="">Todas</option>'] + [
        f'<option value="{html.escape(cid)}">{html.escape(cid)}</option>' for cid in sorted(cats)
    ]
    type_options = ['<option value="">Todos</option>'] + [
        f'<option value="{html.escape(t)}">{html.escape(t)}</option>' for t in sorted(types)
    ]

    tr_html = []
    for r in rows:
        line_html = r.get("line_html", "")
        err = r.get("error", "")
        comment = r.get("comment", "")
        rule_id = (r.get("rule_id", "") or "").strip()
        cat_id = (r.get("cat_id", "") or "").strip()
        issue_type = (r.get("issue_type", "") or "").strip()
        token = (r.get("token", "") or "").strip()
        is_misspelling = bool(r.get("is_misspelling", False))

        suppressed = (rule_id and rule_id in suppress_rules) or (cat_id and cat_id in suppress_categories)
        tr_class = "hidden" if suppressed else ""

        add_btn = ""
        if is_misspelling and token and (" " not in token) and ("\t" not in token) and ("\n" not in token):
            add_btn = f' <button class="addword" type="button" data-word="{html.escape(token, quote=True)}">➕ Diccionario</button>'

        tr_html.append(
            f'<tr class="{tr_class}" data-cat="{html.escape(cat_id)}" data-type="{html.escape(issue_type)}" '
            f'data-rule="{html.escape(rule_id)}" data-suppressed="{("1" if suppressed else "0")}">'
            f'<td>{line_html}</td>'
            f'<td>{err}</td>'
            f'<td>{comment}{add_btn}</td>'
            f'<td><code>{html.escape(rule_id or "—")}</code></td>'
            f'<td><code>{html.escape(cat_id or "—")}</code></td>'
            f'</tr>'
        )

    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 20px; }}
    h1 {{ font-size: 20px; margin: 0 0 8px; }}
    .meta {{ color: #444; font-size: 13px; margin-bottom: 12px; }}
    .meta ul {{ margin: 6px 0 0 18px; padding: 0; }}

    .toolbar {{
      display: grid;
      grid-template-columns: minmax(220px, 1fr) minmax(180px, 260px) minmax(160px, 240px) auto;
      gap: 10px;
      align-items: end;
      margin: 14px 0 10px;
      padding: 12px;
      border: 1px solid #e5e5e5;
      border-radius: 10px;
      background: #fafafa;
    }}
    @media (max-width: 900px) {{
      .toolbar {{ grid-template-columns: 1fr 1fr; }}
    }}
    @media (max-width: 600px) {{
      .toolbar {{ grid-template-columns: 1fr; }}
    }}
    .toolbar label {{ display: block; font-size: 12px; color: #555; margin-bottom: 4px; }}
    .toolbar input[type="text"], .toolbar select {{
      width: 100%;
      max-width: 100%;
      padding: 8px 10px;
      border: 1px solid #d6d6d6;
      border-radius: 8px;
      font-size: 13px;
      background: #fff;
      box-sizing: border-box;
    }}
    .toolbar button {{
      padding: 9px 12px;
      border: 1px solid #d6d6d6;
      border-radius: 8px;
      background: #fff;
      cursor: pointer;
      font-size: 13px;
      white-space: nowrap;
    }}
    .counts {{ font-size: 12px; color: #666; margin-top: 6px; }}

    .chk {{
      display: flex;
      gap: 8px;
      align-items: center;
      padding: 8px 10px;
      border: 1px solid #d6d6d6;
      border-radius: 8px;
      background: #fff;
      font-size: 13px;
      height: 36px;
      box-sizing: border-box;
      white-space: nowrap;
    }}

    table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
    th, td {{ border: 1px solid #ddd; padding: 10px; vertical-align: top; word-wrap: break-word; }}
    th {{ position: sticky; top: 0; background: #f7f7f7; z-index: 1; }}
    tr:nth-child(even) td {{ background: #fcfcfc; }}
    mark {{ background: #fff3b0; padding: 0 2px; }}
    .hidden {{ display: none; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size: 12px; }}
    .addword {{ margin-left: 8px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>

  <div class="meta">
    <div><span class="small">Generado: {now} · Filas (errores): {total} · Omitidas por defecto: {suppressed_count}</span></div>
    <ul>{meta_lines}</ul>
  </div>

  <div class="toolbar">
    <div>
      <label for="q">Buscar</label>
      <input id="q" type="text" placeholder="Texto libre (línea, error, regla, categoría, comentario)…" />
      <div class="counts" id="counts"></div>
    </div>

    <div>
      <label for="cat">Categoría</label>
      <select id="cat">{''.join(cat_options)}</select>
    </div>

    <div>
      <label for="type">Tipo</label>
      <select id="type">{''.join(type_options)}</select>
    </div>

    <div>
      <label>&nbsp;</label>
      <div style="display:flex; gap:10px; align-items:center; justify-content:flex-end; flex-wrap: wrap;">
        <div class="chk" title="Mostrar también las reglas/categorías omitidas por defecto">
          <input id="showSuppressed" type="checkbox" />
          <span>Mostrar omitidas</span>
        </div>
        <button id="reset" type="button">Reset</button>
      </div>
    </div>
  </div>

  <table id="tbl">
    <thead>
      <tr>
        <th style="width: 40%;">Línea</th>
        <th style="width: 14%;">Error</th>
        <th style="width: 28%;">Comentario</th>
        <th style="width: 10%;">Regla</th>
        <th style="width: 8%;">Categoría</th>
      </tr>
    </thead>
    <tbody>
      {''.join(tr_html)}
    </tbody>
  </table>

<script>
(function() {{
  const q = document.getElementById('q');
  const cat = document.getElementById('cat');
  const type = document.getElementById('type');
  const reset = document.getElementById('reset');
  const showSuppressed = document.getElementById('showSuppressed');
  const tbody = document.querySelector('#tbl tbody');
  const counts = document.getElementById('counts');

  // Heartbeat: when served via --serve, keep pinging the server.
  // When you close the tab/window, pings stop and the server auto-closes after idle timeout.
  if (window.location.protocol === 'http:' || window.location.protocol === 'https:') {{
        setInterval(() => {{ fetch('/ping', {{ cache: 'no-store' }}).catch(() => {{}}); }}, 4000);
  }}

  // Row buttons: ➕ Diccionario
  async function pushWordToBridge(word) {{
    // If served via http(s), use same-origin relative endpoint.
    // If opened as file://, fall back to localhost:8765.
    const isHttp = (window.location.protocol === 'http:' || window.location.protocol === 'https:');
    const base = isHttp ? '' : 'http://127.0.0.1:8765';
    const url = base + '/api/add-word';

    try {{
      const resp = await fetch(url, {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{word: word}})
      }});
      const js = await resp.json().catch(() => ({{}}));
      return !!js.ok;
    }} catch (e) {{
      return false;
    }}
  }}

  tbody.addEventListener('click', async (ev) => {{
    const btn = ev.target.closest && ev.target.closest('button.addword');
    if (!btn) return;
    const w = (btn.getAttribute('data-word') || '').trim();
    if (!w) return;

    btn.disabled = true;
    btn.textContent = '⏳ Enviando…';

    const ok = await pushWordToBridge(w);
    if (ok) {{
      btn.textContent = '✅ En LT';
      return;
    }}

    // No local queue: show a warning and allow retry
    btn.textContent = '⚠️ Sin servidor';
    setTimeout(() => {{ btn.disabled = false; btn.textContent = '➕ Diccionario'; }}, 1200);
  }});



// Normalize text for searching (case-insensitive + Unicode NFKC)
function norm(s) {{
  try {{
    return (s == null ? '' : String(s))
      .toLowerCase()
      .normalize('NFKC')
      .replace(/\\s+/g, ' ')
      .trim();
  }} catch (e) {{
    // normalize() might not exist in very old browsers
    return (s == null ? '' : String(s)).toLowerCase().replace(/\\s+/g, ' ').trim();
  }}
}}
  function apply() {{
    const qv = norm(q.value);
    const cv = cat.value;
    const tv = type.value;
    const showSup = !!showSuppressed.checked;

    let visible = 0;
    const rows = tbody.querySelectorAll('tr');
    rows.forEach(tr => {{
      const trCat = tr.getAttribute('data-cat') || '';
      const trType = tr.getAttribute('data-type') || '';
      const trRule = tr.getAttribute('data-rule') || '';
      const trSupp = tr.getAttribute('data-suppressed') === '1';
      const text = norm(tr.innerText) + ' ' + norm(trRule);

      const okSup = showSup || !trSupp;
      const okCat = !cv || trCat === cv;
      const okType = !tv || trType === tv;
      const okQ = !qv || text.includes(qv);

      const show = okSup && okCat && okType && okQ;
      tr.classList.toggle('hidden', !show);
      if (show) visible++;
    }});
    counts.textContent = 'Mostrando ' + visible + ' de ' + rows.length;
  }}

  q.addEventListener('input', apply);
  cat.addEventListener('change', apply);
  type.addEventListener('change', apply);
  showSuppressed.addEventListener('change', apply);
  reset.addEventListener('click', () => {{
    q.value = '';
    cat.value = '';
    type.value = '';
    showSuppressed.checked = false;
    apply();
    q.focus();
  }});

  apply();
}})();
</script>
</body>
</html>
"""

def open_file(path: str) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}"')
    except Exception as e:
        print(f"[WARN] No pude abrir el HTML automáticamente: {e}")

def open_url(url: str) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(url)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            os.system(f'open "{url}"')
        else:
            os.system(f'xdg-open "{url}"')
    except Exception as e:
        print(f"[WARN] No pude abrir el navegador automáticamente: {e}")

def serve_report_with_bridge(root_dir: str, settings: Settings, local_dict_path: Optional[str] = None, report_path: Optional[str] = None, host: str = "127.0.0.1", port: int = 8765, idle_timeout_s: int = 45, cleanup_paths: Optional[List[str]] = None) -> None:
    """
    Serve the output folder over HTTP and expose:
      - GET  /health
      - POST /api/add-word  JSON: {"word":"...", "dict":""}
    """
    root = Path(root_dir).resolve()
    report_abs = None
    report_name = ""
    if report_path:
        try:
            report_abs = Path(report_path).resolve()
            report_name = report_abs.name
        except Exception:
            report_abs = None
            report_name = ""
    heartbeat = {"t": time.time()}


    class Handler(BaseHTTPRequestHandler):
        server_version = "TakoWorksCorrectorServe/0.472"

        def _set_headers(self, status: int = 200, content_type: str = "application/json; charset=utf-8"):
            self.send_response(status)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Type", content_type)
            self.end_headers()

        def do_OPTIONS(self):
            self._set_headers(204)
            return

        def do_GET(self):
            if self.path.startswith("/health"):
                self._set_headers(200)
                self.wfile.write(json.dumps({"ok": True}).encode("utf-8"))
                return
            if self.path.startswith("/ping"):
                heartbeat["t"] = time.time()
                self._set_headers(200)
                self.wfile.write(json.dumps({"ok": True}).encode("utf-8"))
                return


            rel = self.path.lstrip("/").split("?", 1)[0]
            if rel in ("", "report") or (report_name and rel == report_name):
                # Always serve the report, even if it's not located under root_dir
                if report_abs and report_abs.is_file():
                    heartbeat["t"] = time.time()
                    self._set_headers(200, "text/html; charset=utf-8")
                    self.wfile.write(report_abs.read_bytes())
                    return

            file_path = (root / rel).resolve()
            if not str(file_path).startswith(str(root)):
                self._set_headers(403, "text/plain; charset=utf-8")
                self.wfile.write(b"Forbidden")
                return

            if file_path.is_file():
                ct = "text/plain; charset=utf-8"
                suf = file_path.suffix.lower()
                if suf == ".html":
                    ct = "text/html; charset=utf-8"
                elif suf == ".js":
                    ct = "application/javascript; charset=utf-8"
                elif suf == ".css":
                    ct = "text/css; charset=utf-8"
                elif suf == ".json":
                    ct = "application/json; charset=utf-8"
                self._set_headers(200, ct)
                self.wfile.write(file_path.read_bytes())
                return

            self._set_headers(404, "text/plain; charset=utf-8")
            self.wfile.write(b"Not found")

        def do_POST(self):
            if not self.path.startswith("/api/add-word"):
                self._set_headers(404)
                self.wfile.write(json.dumps({"ok": False, "error": "not_found"}).encode("utf-8"))
                return

            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length) if length > 0 else b"{}"
            try:
                payload = json.loads(body.decode("utf-8", errors="replace"))
            except Exception:
                self._set_headers(400)
                self.wfile.write(json.dumps({"ok": False, "error": "bad_json"}).encode("utf-8"))
                return

            word = str(payload.get("word", "")).strip()
            dict_name = str(payload.get("dict", "")).strip()
            if not word or any(c.isspace() for c in word):
                self._set_headers(400)
                self.wfile.write(json.dumps({"ok": False, "error": "invalid_word"}).encode("utf-8"))
                return

            try:
                ok, detail = lt_add_word(word, settings, dict_name=dict_name)
                if ok and local_dict_path:
                    append_local_word(local_dict_path, word)
                self._set_headers(200 if ok else 502)
                self.wfile.write(json.dumps({"ok": ok, "word": word, "detail": detail}).encode("utf-8"))
            except Exception as e:
                self._set_headers(502)
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))

    httpd = ThreadingHTTPServer((host, int(port)), Handler)
    print(f"[+] Servidor local listo: http://{host}:{port}")
    print(f"[+] Sirviendo carpeta: {root}")
    print("[i] El servidor se cerrará solo cuando cierres el HTML (timeout de inactividad). Ctrl+C también funciona.")
    try:
        # idle-aware loop (closes when the HTML tab is closed and stops pinging)
        httpd.timeout = 1
        while True:
            httpd.handle_request()
            if idle_timeout_s > 0 and (time.time() - heartbeat['t']) > idle_timeout_s:
                print(f"[i] Sin actividad ({idle_timeout_s}s). Cerrando servidor…")
                break
    except KeyboardInterrupt:
        print("\n[i] Cerrando servidor…")

    # Cleanup temp snapshot files
    if cleanup_paths:
        for p in cleanup_paths:
            try:
                if p and os.path.isfile(p):
                    os.remove(p)
            except Exception:
                pass

def main() -> int:
    ap = argparse.ArgumentParser(prog="corrector", description="TakoWorks - Corrector (ASS -> LanguageTool -> HTML)")
    ap.add_argument("ass_path", help="Ruta del .ass")
    ap.add_argument("--settings", help="Ruta a corrector_settings.json", default=None)
    ap.add_argument("--no-open", action="store_true", help="No abrir el HTML al terminar")
    ap.add_argument("--max-chars", type=int, default=45000, help="Máximo de caracteres por request (por chunk)")
    ap.add_argument("--keep-intermediate", action="store_true", help="Guardar archivos intermedios (_cartelFirst.ass, _dialogo.txt, _dialogo_merged.txt, _LT_raw.json)")
    ap.add_argument("--serve", action="store_true", help="Sirve el HTML en localhost y habilita añadir palabras al diccionario con un clic")
    ap.add_argument("--serve-idle", type=int, default=45, help="Auto-cerrar servidor tras N segundos sin ping del HTML (default: 45)")
    ap.add_argument("--port", type=int, default=8765, help="Puerto para --serve (default: 8765)")
    ap.add_argument("--host", default="127.0.0.1", help="Host para --serve (default: 127.0.0.1)")
    args = ap.parse_args()

    ass_path = os.path.abspath(args.ass_path)
    if not os.path.isfile(ass_path):
        print(f"[ERROR] No existe el archivo: {ass_path}")
        return 2

    input_dir = os.path.dirname(ass_path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    settings = _load_settings(args.settings, script_dir, input_dir)

    local_dict_path = _get_local_dict_path(script_dir, settings)
    local_word_set = load_local_word_set(local_dict_path)
    if local_word_set:
        print(f"[+] Diccionario local cargado: {len(local_word_set)} palabras")
    if args.no_open:
        settings.open_html = False

    # Safety cap to avoid LT 413 (payload too large). LT API limits are plan-dependent.
    if settings.lt_username and settings.lt_api_key:
        # Premium API supports up to ~60k characters, but form-encoding adds overhead; keep some margin.
        if args.max_chars > 45000:
            print(f"[i] --max-chars demasiado alto ({args.max_chars}); usando 45000 para evitar 413.")
            args.max_chars = 45000
    else:
        # Free API limit is much lower; keep a conservative default.
        if args.max_chars > 18000:
            print(f"[i] Sin credenciales: bajando --max-chars de {args.max_chars} a 18000 para evitar 413.")
            args.max_chars = 18000

    suppress_rules_set = _csv_set(settings.suppress_rules)
    suppress_categories_set = _csv_set(settings.suppress_categories)

    # Personal dictionary snapshot (LT Plus) - always download fresh each run
    personal_word_set: Set[str] = set()
    cleanup_paths: List[str] = []

    if settings.apply_personal_dict_filter and settings.lt_username and settings.lt_api_key:
        try:
            dict_name = getattr(settings, "personal_dict_name", "") or ""
            mode = (getattr(settings, "personal_dict_snapshot_mode", "") or "keep").strip().lower()
            custom_path = (getattr(settings, "personal_dict_snapshot_path", "") or "").strip()
            if mode not in ("keep", "temp"):
                mode = "keep"

            snapshot_path = custom_path or _get_snapshot_dict_path(script_dir, settings)

            if mode == "temp":
                fd, tmp_path = tempfile.mkstemp(prefix="lt_personal_snapshot_", suffix=".txt")
                os.close(fd)
                snapshot_path = tmp_path
                cleanup_paths.append(tmp_path)

            print("[+] Descargando diccionario personal de LT (snapshot)…")
            remote_words_raw = lt_get_words(settings, dict_name=dict_name, limit=20000)
            remote_norm_set = { _norm_word(w) for w in remote_words_raw if _norm_word(w) }
            personal_word_set = set(remote_norm_set)

            # Merge local (added via HTML) to be resilient to any LT propagation delays
            if local_word_set:
                personal_word_set |= local_word_set

            # Build a *raw* snapshot file for verification (preserve casing as returned by LT)
            local_raw = load_local_words_raw(local_dict_path) if local_dict_path else []
            local_extra_raw: list[str] = []
            for w in local_raw:
                k = _norm_word(w)
                if k and k not in remote_norm_set:
                    local_extra_raw.append(w)

            snapshot_words_raw = [w for w in remote_words_raw if isinstance(w, str) and w.strip()]
            if local_extra_raw:
                snapshot_words_raw.extend([w for w in local_extra_raw if isinstance(w, str) and w.strip()])

            # Write snapshot for verification / debugging
            try:
                parent = os.path.dirname(snapshot_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
            except Exception:
                pass
            try:
                # Sort by normalized key for easier diff with LT's web export (keeps original casing)
                snapshot_words_raw_sorted = sorted(snapshot_words_raw, key=lambda x: _norm_word(x))
                with open(snapshot_path, "w", encoding="utf-8", newline="\n") as f:
                    for w in snapshot_words_raw_sorted:
                        f.write(w.replace("\r","").replace("\n","") + "\n")
                print(f"[+] Snapshot diccionario guardado: {snapshot_path} ({len(snapshot_words_raw_sorted)} entradas; LT={len(remote_words_raw)}; local_extra={len(local_extra_raw)}; unique_norm={len(personal_word_set)})")
            except Exception as e:
                print(f"[WARN] No se pudo guardar snapshot del diccionario: {e}")

            print(f"[+] Diccionario personal cargado: {len(personal_word_set)} palabras (LT_unique_norm: {len(remote_norm_set)}; LT_raw: {len(remote_words_raw)}; local_norm: {len(local_word_set)})")
        except Exception as e:
            print(f"[WARN] No se pudo cargar el diccionario personal: {e}")



    base_name = os.path.splitext(os.path.basename(ass_path))[0]
    out_dir = input_dir

    print("[+] Leyendo ASS…")
    ass_text = _read_text(ass_path)

    print("[+] Detectando CARTEL, moviéndolos arriba, y aplicando rayas…")
    ass_reordered, events_reordered, cartel_count = reorder_cartel_in_events_and_apply_dashes(ass_text)

    if args.keep_intermediate:
        out_ass = os.path.join(out_dir, f"{base_name}_cartelFirst.ass")
        _write_text(out_ass, ass_reordered)
    print(f"[+] CARTEL detectados y movidos: {cartel_count}")

    print("[+] Extrayendo diálogo y limpiando tags…")
    lines, flags = extract_dialogue_lines(events_reordered)
    if args.keep_intermediate:
        out_txt = os.path.join(out_dir, f"{base_name}_dialogo.txt")
        _write_text(out_txt, "\n".join(lines) + ("\n" if lines else ""))

    print("[+] Fusionando líneas según puntuación (respetando CARTEL)…")
    merged = merge_lines(lines, flags)
    if args.keep_intermediate:
        out_merged = os.path.join(out_dir, f"{base_name}_dialogo_merged.txt")
        _write_text(out_merged, "\n".join(merged) + ("\n" if merged else ""))

    print("[+] Enviando a LanguageTool…")
    rows_all: List[Dict[str, Any]] = []
    raw_responses: List[Dict[str, Any]] = []
    chunks = split_into_chunks(merged, max_chars=args.max_chars)

    for ci, chunk_lines in enumerate(chunks, start=1):
        chunk_text = "\n".join(chunk_lines)
        try:
            resp = lt_check(chunk_text, settings)
        except Exception as e:
            meta = {
                "Archivo": os.path.basename(ass_path),
                "Endpoint": settings.lt_base_url,
                "Idioma": settings.language,
                "Level": settings.level,
                "CARTEL movidos": str(cartel_count),
                "Chunk": f"{ci}/{len(chunks)}",
                "Error": str(e),
            }
            html_err = build_html_report(f"{base_name} — LanguageTool (ERROR)", [], meta, suppress_rules_set, suppress_categories_set)
            out_html = os.path.join(out_dir, f"{base_name}_LT_report.html")
            _write_text(out_html, html_err)
            print(f"[ERROR] Falló la llamada a LanguageTool: {e}")
            print(f"[+] HTML de error generado: {out_html}")
            if settings.open_html and not args.serve:
                open_file(out_html)
            return 3

        raw_responses.append(resp)
        rows = map_matches_to_rows(chunk_lines, resp)
        if not personal_word_set and local_word_set:
            personal_word_set = set(local_word_set)
        if personal_word_set:
            rows = filter_rows_by_personal_dict(rows, personal_word_set)
        rows_all.extend(rows)
        print(f"    - Chunk {ci}/{len(chunks)}: {len(rows)} errores")


    if args.keep_intermediate:
        out_json = os.path.join(out_dir, f"{base_name}_LT_raw.json")
        _write_text(out_json, json.dumps(raw_responses, ensure_ascii=False, indent=2))

    meta = {
        "Archivo": os.path.basename(ass_path),
        "Endpoint": settings.lt_base_url,
        "Idioma": settings.language,
        "Plan": "Plus (credenciales)" if (settings.lt_username and settings.lt_api_key) else "Sin credenciales",
        "Level": settings.level,
        "motherTongue": settings.mother_tongue or "(vacío)",
        "enabledOnly": str(settings.enabled_only),
        "enabledCategories": settings.enabled_categories or "(vacío)",
        "enabledRules": settings.enabled_rules or "(vacío)",
        "Omitir reglas (HTML)": settings.suppress_rules or "(vacío)",
        "Omitir categorías (HTML)": settings.suppress_categories or "(vacío)",
        "CARTEL movidos": str(cartel_count),
        "Líneas (merged)": str(len(merged)),
        "Chunks": str(len(chunks)),
    }

    out_html = os.path.join(out_dir, f"{base_name}_LT_report.html")
    html_doc = build_html_report(f"{base_name} — LanguageTool", rows_all, meta, suppress_rules_set, suppress_categories_set)
    _write_text(out_html, html_doc)

    print(f"[+] Listo. HTML: {out_html}")

    if args.serve:
        url = f"http://{args.host}:{int(args.port)}/report"
        print(f"[+] Abriendo: {url}")
        open_url(url)
        serve_report_with_bridge(out_dir, settings, local_dict_path=local_dict_path, report_path=out_html, host=args.host, port=int(args.port), idle_timeout_s=int(args.serve_idle), cleanup_paths=cleanup_paths)
        return 0

    if settings.open_html:
        open_file(out_html)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())