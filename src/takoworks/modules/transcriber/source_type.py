from __future__ import annotations

SOURCE_TYPE_ALIASES = {
    "manga": "Manga",
    "manhwa": "Manhwa",
    "light novel": "Light novel",
    "novela ligera": "Light novel",
    "none": "None",
    "nada": "None",
    "": "None",
}


def normalize_source_type(source_type: str) -> str:
    key = str(source_type or "").strip().casefold()
    return SOURCE_TYPE_ALIASES.get(key, "None")


def describe_source_type(source_type: str) -> str:
    source_type = normalize_source_type(source_type)
    if source_type == "Manga":
        return "The series is based on a manga. Prefer official manga terminology and translations when possible."
    if source_type == "Manhwa":
        return "The series is based on a manhwa. Prefer official manhwa terminology and translations when possible."
    if source_type == "Light novel":
        return "The series is based on a light novel. Prefer official light novel terminology and translations when possible."
    return "No hay material original claramente definido o no es relevante; prioriza la coherencia interna de la serie."
