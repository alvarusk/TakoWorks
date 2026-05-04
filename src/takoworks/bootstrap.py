from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from . import paths
from .config import load_config, save_config


def _first_existing(*candidates: Path) -> str:
    for c in candidates:
        if c and c.exists():
            return str(c)
    return ""


def bootstrap() -> Dict[str, Any]:
    cfg = load_config()

    # bin/ en PATH (ffmpeg/yomitoku embebidos)
    if not cfg.get("ffmpeg_dir"):
        cfg["ffmpeg_dir"] = _first_existing(paths.bin_dir() / "ffmpeg")
    if not cfg.get("yomitoku_dir"):
        cfg["yomitoku_dir"] = _first_existing(paths.bin_dir() / "yomitoku")

    extra = []
    if cfg.get("ffmpeg_dir"):
        extra.append(cfg["ffmpeg_dir"])
    if cfg.get("yomitoku_dir"):
        extra.append(cfg["yomitoku_dir"])

    if extra:
        os.environ["PATH"] = os.pathsep.join(extra + [os.environ.get("PATH", "")])

    save_config(cfg)
    return cfg
