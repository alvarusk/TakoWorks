#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local bridge server for TakoWorks Corrector HTML -> LanguageTool Plus dictionary.

It exposes:
  - GET  /health
  - POST /api/add-word   JSON: {"word":"...", "dict":""}

Run:
  py lt_bridge_server.py --dir "C:\Path\To\FolderWithHTML" --port 8765
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Tuple

def load_settings(settings_path: Path) -> Dict[str, Any]:
    with open(settings_path, "r", encoding="utf-8") as f:
        return json.load(f)

def post_add_word(base_url: str, username: str, api_key: str, word: str, dict_name: str = "") -> Tuple[bool, str]:
    url = base_url.rstrip("/") + "/words/add"
    data = {"word": word, "username": username, "apiKey": api_key}
    if dict_name:
        data["dict"] = dict_name
    encoded = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(url, data=encoded, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded; charset=utf-8")
    req.add_header("User-Agent", "TakoWorks-Corrector-Bridge/0.471")
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    try:
        js = json.loads(raw)
        if "added" in js:
            if bool(js.get("added")):
                return True, "added"
            # added=false means "it was added before" (not an error)
            return True, "already_present"
        return True, raw

    except Exception:
        if "true" in raw.lower() or "added" in raw.lower() or "ok" in raw.lower():
            return True, raw
        return False, raw

class Handler(BaseHTTPRequestHandler):
    server_version = "TakoWorksCorrectorBridge/0.471"

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

        root: Path = self.server.root_dir  # type: ignore[attr-defined]
        if self.path == "/" or self.path == "":
            self._set_headers(200, "text/plain; charset=utf-8")
            items = "\n".join(sorted([p.name for p in root.glob("*") if p.is_file()]))
            self.wfile.write(items.encode("utf-8"))
            return

        rel = self.path.lstrip("/").split("?", 1)[0]
        file_path = (root / rel).resolve()
        if not str(file_path).startswith(str(root.resolve())):
            self._set_headers(403, "text/plain; charset=utf-8")
            self.wfile.write(b"Forbidden")
            return
        if file_path.is_file():
            ct = "text/plain; charset=utf-8"
            if file_path.suffix.lower() == ".html":
                ct = "text/html; charset=utf-8"
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

        cfg: Dict[str, Any] = self.server.settings  # type: ignore[attr-defined]
        base = str(cfg.get("lt_base_url", "https://api.languagetoolplus.com/v2")).rstrip("/")
        username = str(cfg.get("lt_username", "")).strip()
        api_key = str(cfg.get("lt_api_key", "")).strip()

        if not username or not api_key:
            self._set_headers(500)
            self.wfile.write(json.dumps({"ok": False, "error": "missing_credentials"}).encode("utf-8"))
            return

        try:
            ok, detail = post_add_word(base, username, api_key, word, dict_name)
            self._set_headers(200 if ok else 502)
            self.wfile.write(json.dumps({"ok": ok, "word": word, "detail": detail}).encode("utf-8"))
        except Exception as e:
            self._set_headers(502)
            self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))

def main() -> int:
    ap = argparse.ArgumentParser(description="Local bridge server for HTML -> LanguageTool Plus dictionary.")
    ap.add_argument("--settings", default=None, help="Ruta a corrector_settings.json (default: junto al script)")
    ap.add_argument("--dir", default=".", help="Directorio a servir (donde está el HTML)")
    ap.add_argument("--port", type=int, default=8765, help="Puerto local (default 8765)")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    settings_path = Path(args.settings).resolve() if args.settings else (here / "corrector_settings.json")
    if not settings_path.is_file():
        print(f"[ERROR] No encuentro settings: {settings_path}")
        return 2

    root_dir = Path(args.dir).resolve()
    if not root_dir.is_dir():
        print(f"[ERROR] No existe el directorio: {root_dir}")
        return 3

    settings = load_settings(settings_path)
    addr = ("127.0.0.1", int(args.port))
    httpd = ThreadingHTTPServer(addr, Handler)
    httpd.settings = settings  # type: ignore[attr-defined]
    httpd.root_dir = root_dir  # type: ignore[attr-defined]

    print(f"[+] Bridge listo en http://{addr[0]}:{addr[1]}")
    print(f"[+] Sirviendo directorio: {root_dir}")
    print("[i] Ctrl+C para salir.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[i] Cerrando…")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
