#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TakoWorks - Unworder (Context Menu)

Convierte .docx/.txt (Actor: Texto) a .ass.
Soporta multi-selección: recibe varios archivos.

Uso:
  py unworder_context.py file1.docx file2.docx ...
"""
import os
import sys

def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("[ERROR] No input files.")
        return 2

    core = None
    try:
        from takoworks.modules.unworder import core as _core  # type: ignore
        core = _core
    except Exception:
        try:
            import core as _core  # type: ignore
            core = _core
        except Exception as e:
            print("[ERROR] No se pudo importar Unworder core:", e)
            return 3

    ok = 0
    for inp in argv:
        if not os.path.isfile(inp):
            print(f"[WARN] Not found: {inp}")
            continue
        base = os.path.splitext(os.path.basename(inp))[0]
        out_ass = os.path.join(os.path.dirname(inp), f"{base}.ass")
        try:
            lines = core.read_lines_from_doc(inp, console=_NullConsole())
            if not lines:
                print(f"[WARN] No valid lines in: {inp}")
                continue
            ass_content = core.build_ass_content(lines)
            with open(out_ass, "w", encoding="utf-8-sig", errors="replace") as f:
                f.write(ass_content)
            print(f"[OK] {out_ass}")
            ok += 1
        except Exception as e:
            print(f"[ERROR] {inp}: {e}")
    return 0 if ok > 0 else 4

class _NullConsole:
    def insert(self, *_args): pass
    def see(self, *_args): pass
    def update_idletasks(self): pass

if __name__ == "__main__":
    raise SystemExit(main())
