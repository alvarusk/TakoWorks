from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests

from ...config import load_config
from ..transferer.transferer import parse_ass

DEEPL_PRO_URL = "https://api.deepl.com"
DEEPL_FREE_URL = "https://api-free.deepl.com"
DEEPL_TRANSLATE_MAX_ITEMS = 50
DEEPL_TRANSLATE_MAX_BYTES = 120 * 1024
DEEPL_GLOSSARY_POLL_TIMEOUT_S = 30.0
DEEPL_GLOSSARY_POLL_INTERVAL_S = 1.0

QUOTE_TRANSLATION_TABLE = str.maketrans(
    {
        "«": '"',
        "»": '"',
        "‹": '"',
        "›": '"',
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "❝": '"',
        "❞": '"',
    }
)

HEADER_HINTS = {
    "source",
    "target",
    "source_lang",
    "target_lang",
    "english",
    "spanish",
    "en",
    "es",
    "src",
    "dst",
    "orig",
    "translated",
    "translation",
    "source text",
    "target text",
}

GLOSSARY_DELIMITERS = (",", ";", "\t", "|")


def _read_api_key() -> str:
    env = os.getenv("DEEPL_AUTH_KEY", "").strip() or os.getenv("DEEPL_API_KEY", "").strip()
    if env:
        return env
    try:
        cfg = load_config()
        api_keys = cfg.get("api_keys", {})
        if isinstance(api_keys, dict):
            return str(api_keys.get("deepl", "") or "").strip()
    except Exception:
        pass
    return ""


def _deepl_base_url(auth_key: str) -> str:
    override = os.getenv("DEEPL_BASE_URL", "").strip()
    if override:
        return override.rstrip("/")
    if auth_key.endswith(":fx"):
        return DEEPL_FREE_URL
    return DEEPL_PRO_URL


def _normalize_translated_text(text: str) -> str:
    text = (text or "").translate(QUOTE_TRANSLATION_TABLE)
    text = text.replace("...", "…")
    return text


def _make_output_path(src_path: str) -> str:
    src = Path(src_path)
    stem = src.stem
    new_stem = re.sub("(?i)en-us", "es-es", stem)
    new_stem = re.sub("(?i)full", "edited", new_stem)
    if new_stem == stem:
        new_stem = f"{stem}_es-es"
    return str(src.with_name(new_stem + src.suffix))


def _looks_like_header(row: Sequence[str]) -> bool:
    if len(row) < 2:
        return False
    a = row[0].strip().lower()
    b = row[1].strip().lower()
    return a in HEADER_HINTS and b in HEADER_HINTS


def _candidate_glossary_delimiters(csv_text: str) -> List[str]:
    first_non_empty = ""
    for line in csv_text.splitlines():
        if line.strip():
            first_non_empty = line
            break

    order = list(GLOSSARY_DELIMITERS)
    if ";" in first_non_empty and "," not in first_non_empty:
        order = [";"] + [d for d in order if d != ";"]
    elif "," in first_non_empty and ";" not in first_non_empty:
        order = [","] + [d for d in order if d != ","]
    else:
        try:
            sniffed = csv.Sniffer().sniff(csv_text, delimiters=",;\t|")
            if sniffed.delimiter in order:
                order = [sniffed.delimiter] + [d for d in order if d != sniffed.delimiter]
        except csv.Error:
            pass
    return order


def _load_glossary_csv(csv_path: str) -> Tuple[str, int]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        csv_text = fh.read()

    rows: List[Tuple[str, str]] = []
    last_error: Optional[ValueError] = None
    for delimiter in _candidate_glossary_delimiters(csv_text):
        candidate_rows: List[Tuple[str, str]] = []
        bad_row: Optional[List[str]] = None
        reader = csv.reader(io.StringIO(csv_text), delimiter=delimiter)
        for raw_row in reader:
            row = [cell.strip() for cell in raw_row]
            if not any(row):
                continue
            if not candidate_rows and _looks_like_header(row):
                continue
            if len(row) < 2:
                bad_row = raw_row
                break
            source = row[0].strip()
            target = row[1].strip()
            if not source or not target:
                continue
            candidate_rows.append((source, target))

        if bad_row is None and candidate_rows:
            rows = candidate_rows
            break
        if bad_row is not None:
            last_error = ValueError(
                f"Glossary CSV rows must have at least 2 columns. Bad row: {bad_row!r}"
            )

    if not rows:
        if last_error is not None:
            raise last_error
        raise ValueError("Glossary CSV does not contain usable entries.")

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerows(rows)
    return buf.getvalue().strip(), len(rows)


def _tokenize_ass_text(text: str, run_id: str) -> Tuple[str, Dict[str, str]]:
    encoded: List[str] = []
    mapping: Dict[str, str] = {}
    tag_idx = 0
    break_idx = 0
    i = 0
    raw = text or ""

    while i < len(raw):
        ch = raw[i]
        if ch == "{":
            end = raw.find("}", i + 1)
            if end != -1:
                token = f"__TW_{run_id}_TAG_{tag_idx}__"
                mapping[token] = raw[i : end + 1]
                encoded.append(token)
                tag_idx += 1
                i = end + 1
                continue
        if ch == "\\" and i + 1 < len(raw) and raw[i + 1] in "NnHh":
            token = f"__TW_{run_id}_BR_{break_idx}__"
            mapping[token] = raw[i : i + 2]
            encoded.append(token)
            break_idx += 1
            i += 2
            continue
        encoded.append(ch)
        i += 1

    return "".join(encoded), mapping


def _restore_ass_text(text: str, mapping: Dict[str, str]) -> str:
    out = text or ""
    for token in sorted(mapping, key=len, reverse=True):
        if token not in out:
            raise ValueError(f"DeepL translation lost placeholder token {token!r}.")
        out = out.replace(token, mapping[token])
    return out


def _chunk_texts(texts: Sequence[str], *, max_items: int, max_bytes: int) -> List[List[str]]:
    batches: List[List[str]] = []
    current: List[str] = []

    def payload_size(items: Sequence[str]) -> int:
        payload = {"text": list(items), "source_lang": "EN", "target_lang": "ES"}
        return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    for text in texts:
        candidate = current + [text]
        size = payload_size(candidate)
        if current and (len(candidate) > max_items or size > max_bytes):
            batches.append(current)
            current = [text]
            continue
        if not current and size > max_bytes:
            raise ValueError("A single subtitle line is too large for a DeepL request.")
        current = candidate

    if current:
        batches.append(current)
    return batches


@dataclass
class DeepLClient:
    auth_key: str
    base_url: str = ""
    timeout_s: float = 60.0
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self) -> None:
        if not self.auth_key:
            raise ValueError("DeepL authentication key is missing.")
        if not self.base_url:
            self.base_url = _deepl_base_url(self.auth_key)
        self.base_url = self.base_url.rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"DeepL-Auth-Key {self.auth_key}",
            "User-Agent": "TakoWorks-Translator/1.0",
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, *, json_body: Optional[dict] = None) -> requests.Response:
        url = f"{self.base_url}{path}"
        resp = self.session.request(
            method,
            url,
            headers=self._headers(),
            json=json_body,
            timeout=self.timeout_s,
        )
        if resp.status_code >= 400:
            detail = resp.text.strip()
            try:
                payload = resp.json()
                message = payload.get("message") or payload.get("detail") or detail
                detail = f"{message}" if not payload.get("detail") else f"{message}: {payload.get('detail')}"
            except Exception:
                pass
            raise RuntimeError(f"DeepL {method} {path} failed with HTTP {resp.status_code}: {detail}")
        return resp

    def create_glossary(
        self,
        *,
        name: str,
        entries_csv: str,
        source_lang: str = "en",
        target_lang: str = "es",
    ) -> str:
        payload = {
            "name": name,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "entries": entries_csv,
            "entries_format": "csv",
        }
        data = self._request("POST", "/v2/glossaries", json_body=payload).json()
        glossary_id = str(data.get("glossary_id", "")).strip()
        if not glossary_id:
            raise RuntimeError("DeepL did not return a glossary_id.")
        return glossary_id

    def get_glossary(self, glossary_id: str) -> dict:
        return self._request("GET", f"/v2/glossaries/{glossary_id}").json()

    def wait_glossary_ready(self, glossary_id: str, timeout_s: float = DEEPL_GLOSSARY_POLL_TIMEOUT_S) -> None:
        deadline = time.time() + timeout_s
        while True:
            data = self.get_glossary(glossary_id)
            if bool(data.get("ready")):
                return
            if time.time() >= deadline:
                raise RuntimeError(f"Glossary {glossary_id} did not become ready in time.")
            time.sleep(DEEPL_GLOSSARY_POLL_INTERVAL_S)

    def delete_glossary(self, glossary_id: str) -> None:
        self._request("DELETE", f"/v2/glossaries/{glossary_id}")

    def translate_batch(
        self,
        texts: Sequence[str],
        *,
        glossary_id: Optional[str] = None,
        source_lang: str = "EN",
        target_lang: str = "ES",
    ) -> List[str]:
        payload: Dict[str, object] = {
            "text": list(texts),
            "source_lang": source_lang,
            "target_lang": target_lang,
        }
        if glossary_id:
            payload["glossary_id"] = glossary_id
        data = self._request("POST", "/v2/translate", json_body=payload).json()
        translations = data.get("translations")
        if not isinstance(translations, list):
            raise RuntimeError("DeepL response does not contain a translations list.")
        out: List[str] = []
        for item in translations:
            if not isinstance(item, dict):
                raise RuntimeError("Unexpected DeepL translation item.")
            out.append(str(item.get("text", "")))
        if len(out) != len(texts):
            raise RuntimeError(
                f"DeepL returned {len(out)} translations for {len(texts)} input strings."
            )
        return out


def translate_ass_file(
    ass_path: str,
    glossary_csv_path: Optional[str] = None,
    out_path: Optional[str] = None,
    *,
    auth_key: Optional[str] = None,
    log=None,
    cancel_event=None,
) -> str:
    if log is None:
        log = print

    auth_key = (auth_key or _read_api_key()).strip()
    client = DeepLClient(auth_key=auth_key)

    src_lines, events = parse_ass(ass_path)
    dialogue_events = [ev for ev in events if str(ev.get("prefix", "")).lower() == "dialogue"]
    glossary_id: Optional[str] = None
    if not dialogue_events:
        out_path = out_path or _make_output_path(ass_path)
        with open(out_path, "w", encoding="utf-8-sig", errors="replace") as fh:
            fh.writelines(src_lines)
        log("[i] No Dialogue lines were found. The file was copied unchanged.")
        return out_path

    log(f"[i] DeepL endpoint: {client.base_url}")
    if glossary_csv_path:
        glossary_csv_path = glossary_csv_path.strip()
    if glossary_csv_path:
        if not os.path.isfile(glossary_csv_path):
            raise FileNotFoundError(f"Glossary CSV file not found: {glossary_csv_path}")
        glossary_entries_csv, entry_count = _load_glossary_csv(glossary_csv_path)
        glossary_name = f"TakoWorks_{Path(glossary_csv_path).stem}_{uuid.uuid4().hex[:8]}"
        log(f"[i] Glossary entries: {entry_count}")

        glossary_id = client.create_glossary(
            name=glossary_name,
            entries_csv=glossary_entries_csv,
            source_lang="en",
            target_lang="es",
        )
        log(f"[i] Created DeepL glossary: {glossary_id}")
    else:
        log("[i] No glossary selected; translating without a DeepL glossary.")

    try:
        if glossary_id:
            client.wait_glossary_ready(glossary_id)

        run_id = uuid.uuid4().hex[:10]
        payloads: List[str] = []
        mappings: List[Dict[str, str]] = []
        refs: List[dict] = []

        for ev in dialogue_events:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Cancelado")
            text_idx = ev.get("text_idx")
            if text_idx is None or text_idx < 0 or text_idx >= len(ev.get("parts", [])):
                continue
            original = ev["parts"][text_idx]
            encoded, mapping = _tokenize_ass_text(original, run_id)
            payloads.append(encoded)
            mappings.append(mapping)
            refs.append(ev)

        batches = _chunk_texts(
            payloads,
            max_items=DEEPL_TRANSLATE_MAX_ITEMS,
            max_bytes=DEEPL_TRANSLATE_MAX_BYTES,
        )

        translated_payloads: List[str] = []
        for batch in batches:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Cancelado")
            batch_translated = client.translate_batch(
                batch,
                glossary_id=glossary_id,
                source_lang="EN",
                target_lang="ES",
            )
            translated_payloads.extend(batch_translated)

        if len(translated_payloads) != len(refs):
            raise RuntimeError("DeepL translation count does not match the input lines.")

        for ev, mapping, translated in zip(refs, mappings, translated_payloads):
            text_idx = ev.get("text_idx")
            if text_idx is None:
                continue
            restored = _restore_ass_text(translated, mapping)
            ev["parts"][text_idx] = _normalize_translated_text(restored)

        out_path = out_path or _make_output_path(ass_path)
        out_lines = list(src_lines)
        for ev in dialogue_events:
            i = ev["line_index"]
            out_lines[i] = f"{ev['leading']}{ev['prefix']}: " + ",".join(ev["parts"]) + ev["line_ending"]

        with open(out_path, "w", encoding="utf-8-sig", errors="replace") as fh:
            fh.writelines(out_lines)
        log(f"[ok] Wrote translated ASS: {out_path}")
        return out_path
    finally:
        try:
            if glossary_id:
                client.delete_glossary(glossary_id)
                log(f"[i] Deleted temporary DeepL glossary: {glossary_id}")
        except Exception as exc:
            log(f"[warn] Could not delete temporary glossary {glossary_id}: {exc}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="TakoWorks Translator (ASS -> DeepL -> ASS)")
    ap.add_argument("--ass", required=True, help="Input ASS file")
    ap.add_argument("--glossary", default="", help="Optional CSV glossary (English-Spanish)")
    ap.add_argument("--out", default=None, help="Output ASS file")
    ap.add_argument("--api-key", default="", help="DeepL auth key (optional; can come from config/env)")
    args = ap.parse_args(argv)

    out_path = translate_ass_file(
        args.ass,
        args.glossary or None,
        args.out,
        auth_key=args.api_key.strip() or None,
    )
    print(f"[ok] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
