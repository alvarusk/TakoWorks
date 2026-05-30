#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TakoWorks - Translator (Context Menu)

Abre TakoWorks con la pestaña Translator y el archivo ASS preseleccionado.

Uso:
  py translator_context.py file.ass
"""

from __future__ import annotations

import os
import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("[ERROR] No input file.")
        return 2

    ass_path = os.path.abspath(argv[0])
    if not os.path.isfile(ass_path):
        print(f"[ERROR] Not found: {ass_path}")
        return 3

    try:
        from takoworks.__main__ import main as takoworks_main  # type: ignore
    except Exception as e:
        print("[ERROR] No se pudo importar TakoWorks:", e)
        return 4

    return int(takoworks_main(["--tab", "Translator", "--ass", ass_path]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
