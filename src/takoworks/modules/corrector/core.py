from __future__ import annotations

import os
import sys
import subprocess
from typing import Optional

def run_corrector(
    ass_path: str,
    *,
    serve: bool = False,
    no_open: bool = False,
    settings_path: Optional[str] = None,
    max_chars: int = 55000,
    keep_intermediate: bool = False,
    log=None,
) -> int:
    """
    Wrapper pensado para TakoWorks:
    - Ejecuta Corrector como script externo (py -3) para mantenerlo aislado.
    - Si empaquetas TakoWorks sin Python, cambia esto por un import directo del corrector.py.

    Retorna el exit code.
    """
    ass_path = os.path.abspath(ass_path)
    if not os.path.isfile(ass_path):
        raise FileNotFoundError(ass_path)

    # Localiza corrector.py junto al instalador (C:\TakoWorks\src\takoworks\modules\corrector\corrector.py)
    here = os.path.dirname(__file__)
    # Si lo copias dentro del módulo: takoworks/modules/corrector_module/corrector.py
    local_candidate = os.path.join(here, "corrector.py")
    # Instalación "tool" separada: C:\TakoWorks\src\takoworks\modules\corrector\corrector.py
    global_candidate = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(here))), "Corrector", "corrector.py")

    script = local_candidate if os.path.isfile(local_candidate) else global_candidate
    if not os.path.isfile(script):
        raise RuntimeError(
            "No se encontró corrector.py. Copia la carpeta Corrector del instalador "
            "o incluye corrector.py dentro del módulo takoworks/modules/corrector_module/."
        )

    cmd = ["py", "-3", script, ass_path, "--max-chars", str(max_chars)]
    if serve:
        cmd.append("--serve")
    if no_open:
        cmd.append("--no-open")
    if keep_intermediate:
        cmd.append("--keep-intermediate")
    if settings_path:
        cmd += ["--settings", settings_path]

    if log:
        log(f"[Corrector] CMD: {' '.join(cmd)}")

    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if log:
        if p.stdout.strip():
            log(p.stdout.strip())
        if p.stderr.strip():
            log(p.stderr.strip())
    return p.returncode
