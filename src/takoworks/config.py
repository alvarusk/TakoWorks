from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from . import paths

_REL_KEYS = {"jpdict_dir", "cndict_dir", "ffmpeg_dir", "yomitoku_dir"}

def _to_portable_path(p: str) -> str:
    if not p:
        return ""
    root = paths.app_root().resolve()
    pp = Path(p).resolve()
    try:
        rel = pp.relative_to(root)   # si está dentro de app_root
        return str(rel)
    except Exception:
        return str(pp)               # si está fuera, se queda absoluto

def _from_portable_path(p: str) -> str:
    if not p:
        return ""
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str((paths.app_root() / pp).resolve())

DEFAULT_CONFIG: Dict[str, Any] = {
    "jpdict_dir": "",
    "cndict_dir": "",
    "ffmpeg_dir": "",      # si lo embebes: bin/ffmpeg
    "yomitoku_dir": "",    # si lo embebes: bin/yomitoku

    "stylizer_options": {
        "clean_carteles": False,
        "clean_comments": False,
        "add_styles": True,
        "transform_styles": True,
        "clean_text": False,
    },

    "last": {
        "ass_in": "",
        "video_in": "",
        "word_in": "",
        "pdf_in": "",
        "out_dir": "",
        "splitter_pos":0,
    }
}


def _portable_config_path() -> Path:
    # config.json junto al exe (portable) o junto al repo (dev)
    return paths.app_root() / "config.json"


def _user_config_path() -> Path:
    # fallback por si no se puede escribir en app_root (instalado típico)
    appdata = os.environ.get("APPDATA")
    if appdata:
        base = Path(appdata) / "TakoWorks"
    else:
        base = Path.home() / ".config" / "TakoWorks"
    paths.ensure_dir(base)
    return base / "config.json"


def config_path() -> Path:
    p = _portable_config_path()
    try:
        # probamos escritura
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            return p
        # intentamos crear vacío
        p.write_text("", encoding="utf-8")
        p.unlink(missing_ok=True)
        return p
    except Exception:
        return _user_config_path()


def load_config() -> Dict[str, Any]:
    p = config_path()
    if not p.exists():
        return json.loads(json.dumps(DEFAULT_CONFIG))

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        data = {}

    # merge suave
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg.update({k: v for k, v in data.items() if k in cfg})

    # sub-merge
    if isinstance(data.get("stylizer_options"), dict):
        cfg["stylizer_options"].update(data["stylizer_options"])
    if isinstance(data.get("last"), dict):
        cfg["last"].update(data["last"])

    for k in _REL_KEYS:
        cfg[k] = _from_portable_path(cfg.get(k, ""))

    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)

    # Copia (profunda) para no tocar cfg en memoria
    portable = json.loads(json.dumps(cfg))

    # Guardar estas claves como rutas relativas si están dentro de app_root()
    for k in _REL_KEYS:
        portable[k] = _to_portable_path(str(portable.get(k, "") or ""))

    p.write_text(json.dumps(portable, ensure_ascii=False, indent=2), encoding="utf-8")
