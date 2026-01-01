def _ass_sanitize_braces(s: str) -> str:
    # Evita romper tags ASS si el texto contiene llaves
    return (s or "").replace("{", "｛").replace("}", "｝")


def _ass_unsanitize_braces(s: str) -> str:
    # Revierte la sanitización de llaves aplicada para ASS
    return (s or "").replace("｛", "{").replace("｝", "}")


def _ass_hide(s: str) -> str:
    # Texto oculto en render, visible como "comentario" dentro del evento en Aegisub
    return "{" + _ass_sanitize_braces(s) + "}"


def _ass_hide_prefix(existing: str) -> str:
    # Oculta todas las líneas existentes (y sus \N) para que no dejen líneas en blanco.
    raw = existing or ""
    if not raw.strip():
        return ""
    lines = raw.split(r"\N")
    out = []
    for ln in lines:
        if ln == "":
            continue
        # Si ya viene como { ... }, evitamos doble llave
        if ln.startswith("{") and ln.endswith("}"):
            ln = ln[1:-1]
        out.append("{" + _ass_sanitize_braces(ln) + r"\N" + "}")
    return "".join(out)
