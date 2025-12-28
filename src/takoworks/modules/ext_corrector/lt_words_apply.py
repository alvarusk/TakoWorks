#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply a word list to LanguageTool Plus personal dictionary.

Usage:
  py lt_words_apply.py words.txt
  py lt_words_apply.py words.txt --settings corrector_settings.json --dict MiDiccionario

Reads credentials from corrector_settings.json:
  lt_base_url, lt_username, lt_api_key

Calls:
  POST {lt_base_url}/words/add   (form-urlencoded)
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Set, Tuple

def load_settings(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_settings(explicit: Optional[str]) -> Optional[str]:
    if explicit and os.path.isfile(explicit):
        return explicit
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "corrector_settings.json")
    if os.path.isfile(cand):
        return cand
    return None

def post_add_word(base_url: str, username: str, api_key: str, word: str, dict_name: str = "") -> Tuple[bool, str]:
    url = base_url.rstrip("/") + "/words/add"
    data = {"word": word, "username": username, "apiKey": api_key}
    if dict_name:
        data["dict"] = dict_name
    encoded = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(url, data=encoded, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded; charset=utf-8")
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    try:
        js = json.loads(raw)
        if bool(js.get("added")):
            return True, "added"
        return False, raw
    except Exception:
        if "true" in raw.lower() or "added" in raw.lower() or "ok" in raw.lower():
            return True, raw
        return False, raw

def main() -> int:
    ap = argparse.ArgumentParser(description="Add words to LanguageTool Plus personal dictionary (/words/add).")
    ap.add_argument("words_file", help="TXT con una palabra por línea")
    ap.add_argument("--settings", default=None, help="Ruta a corrector_settings.json")
    ap.add_argument("--dict", default="", help="Nombre del diccionario (opcional). Si se omite, usa el default.")
    args = ap.parse_args()

    if not os.path.isfile(args.words_file):
        print(f"[ERROR] No existe: {args.words_file}")
        return 2

    settings_path = find_settings(args.settings)
    if not settings_path:
        print("[ERROR] No se encontró corrector_settings.json. Pásalo con --settings.")
        return 3

    s = load_settings(settings_path)
    base = str(s.get("lt_base_url", "https://api.languagetoolplus.com/v2")).rstrip("/")
    username = str(s.get("lt_username", "")).strip()
    api_key = str(s.get("lt_api_key", "")).strip()

    if not username or not api_key:
        print("[ERROR] Falta lt_username o lt_api_key en el settings.")
        return 4

    words: Set[str] = set()
    with open(args.words_file, "r", encoding="utf-8-sig", errors="replace") as f:
        for ln in f:
            w = ln.strip()
            if not w:
                continue
            if any(c.isspace() for c in w):
                continue
            words.add(w)

    if not words:
        print("[i] No hay palabras válidas que añadir.")
        return 0

    print(f"[+] Añadiendo {len(words)} palabra(s)…")
    ok = 0
    for w in sorted(words):
        try:
            added, _ = post_add_word(base, username, api_key, w, args.dict)
            if added:
                ok += 1
                print(f"  ✅ {w}")
            else:
                print(f"  ⚠️  {w} (no confirmado)")
        except Exception as e:
            print(f"  ❌ {w}: {e}")

    print(f"[+] Hecho. OK: {ok}/{len(words)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
