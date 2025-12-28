#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TakoWorks - Stylizer (Context Menu)

Acciones (sin Clean text):
- Add/replace styles
- Transform styles
- Clean typesetting
- Clean comments
- Remove linebreaks

Uso:
  py stylizer_context.py file1.ass file2.ass ...
"""
import os
import sys

def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("[ERROR] No input files.")
        return 2

    # Intentamos importar el core desde el paquete TakoWorks.
    core = None
    try:
        from takoworks.modules.stylizer import core as _core  # type: ignore
        core = _core
    except Exception:
        # Fallback: si este script está junto a core.py copiado por el instalador
        try:
            import core as _core  # type: ignore
            core = _core
        except Exception as e:
            print("[ERROR] No se pudo importar Stylizer core:", e)
            return 3

    ok = 0
    for inp in argv:
        if not os.path.isfile(inp):
            print(f"[WARN] Not found: {inp}")
            continue
        base = os.path.splitext(os.path.basename(inp))[0]
        out_path = os.path.join(os.path.dirname(inp), f"{base}_OUT.ass")
        try:
            lines = core.read_file(inp)
            lines = core.replace_styles_section(lines, console=None)
            lines = core.process_events(
                lines,
                clean_carteles=True,
                clean_comments=True,
                transform_styles=True,
                clean_text=False,
                remove_linebreaks=True,
                console=None,
            )
            core.write_file(out_path, lines)
            print(f"[OK] {out_path}")
            ok += 1
        except Exception as e:
            print(f"[ERROR] {inp}: {e}")
    return 0 if ok > 0 else 4

if __name__ == "__main__":
    raise SystemExit(main())
