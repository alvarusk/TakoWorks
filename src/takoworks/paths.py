from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def repo_root() -> Path:
    # .../src/takoworks/paths.py -> parents[2] = raíz del repo
    return Path(__file__).resolve().parents[2]


def exe_dir() -> Path:
    return Path(sys.executable).resolve().parent


def app_root() -> Path:
    # En modo portable/instalado, asumimos que data/ y bin/ están junto al .exe
    return exe_dir() if is_frozen() else repo_root()


def data_dir() -> Path:
    return app_root() / "data"


def bin_dir() -> Path:
    return app_root() / "bin"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
