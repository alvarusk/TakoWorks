import json
import re
from typing import List, Optional


def parse_json_translations(raw_content: str, fallback_lines: List[str]) -> List[str]:
    """
    Extrae un array de traducciones desde un JSON con forma:
      {"translations": ["...", "...", ...]}
    Aplica heurísticas básicas para limpiar fences ``` ``` y saltos de línea
    insertados. Si no se puede parsear, devuelve las líneas originales
    (fallback_lines).
    """
    raw = (raw_content or "").strip()
    if not raw:
        return fallback_lines

    def _normalize(translations: List[str]) -> List[str]:
        translations = [("" if t is None else str(t)) for t in translations]
        if len(translations) == len(fallback_lines):
            return translations
        print("[AVISO] Nº de traducciones != nº de líneas. Se ajusta al mínimo en común.")
        if len(translations) > len(fallback_lines):
            return translations[:len(fallback_lines)]
        return translations + fallback_lines[len(translations):]

    def _fix_invalid_backslashes(s: str) -> str:
        # En JSON, solo son válidos: \" \\ \/ \b \f \n \r \t \uXXXX
        # Esto rescata cosas típicas como \N (ASS) o \an8, etc.
        return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)

    def _extract_translations_loose(s: str) -> Optional[List[str]]:
        """
        Extrae strings del array translations con un parser ligero, tolerante a:
        - saltos de línea literales dentro de strings
        - comas / whitespace extra
        """
        m = re.search(r'"translations"\s*:\s*\[', s)
        quote_mode = '"'
        if not m:
            m = re.search(r"'translations'\s*:\s*\[", s)
            quote_mode = "'"
        if not m:
            return None

        i = m.end()  # justo después del '['
        depth = 1
        in_str = False
        esc = False
        q = ""
        buf: List[str] = []
        items: List[str] = []

        while i < len(s):
            ch = s[i]
            if in_str:
                if esc:
                    buf.append(ch)
                    esc = False
                elif ch == "\\":
                    buf.append(ch)
                    esc = True
                elif ch == q:
                    # fin de string
                    raw_item = "".join(buf)
                    items.append(raw_item)
                    buf = []
                    in_str = False
                    q = ""
                else:
                    # Permitimos '\n' literal dentro de strings (JSON estricto no lo permite)
                    buf.append(ch)
            else:
                if ch in ('"', "'"):
                    in_str = True
                    q = ch
                    buf = []
                elif ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        break
            i += 1

        # Decodificar escapes “tipo JSON” de manera segura
        out: List[str] = []
        for raw_item in items:
            fixed = raw_item.replace("\r\n", "\n").replace("\r", "\n")
            fixed = fixed.replace("\n", "\\n")  # convertir newline literal a escape JSON
            fixed = _fix_invalid_backslashes(fixed)

            # Intento 1: json.loads (si la cadena estaba en comillas dobles originalmente)
            try:
                if quote_mode == '"':
                    out.append(json.loads('"' + fixed + '"'))
                    continue
            except Exception:
                pass

            # Intento 2: decodificación “manual” conservadora
            # (mantiene secuencias raras tal cual, pero rescata \n, \t, \\, \")
            fixed2 = fixed.replace(r"\n", "\n").replace(r"\t", "\t").replace(r"\r", "\r")
            fixed2 = fixed2.replace(r"\\", "\\").replace(r"\/", "/").replace(r"\"", '"')
            out.append(fixed2)

        return out

    # 1) Quitar fences ```...``` (incluye ```json)
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    # 2) Quedarnos con lo que parece el JSON “real”
    first = raw.find("{")
    last = raw.rfind("}")
    candidate = raw[first:last + 1] if first != -1 and last != -1 and last > first else raw

    # 3) Intento directo
    try:
        data = json.loads(candidate)
        if isinstance(data, dict) and isinstance(data.get("translations"), list):
            return _normalize(list(data["translations"]))
        if isinstance(data, list):
            return _normalize(list(data))
    except Exception:
        pass

    # 4) Heurística: buscar la clave "translations" y decodificar con JSONDecoder
    decoder = json.JSONDecoder()
    for m in re.finditer(r'"translations"\s*:', candidate):
        start = m.end()
        try:
            arr, _ = decoder.raw_decode(candidate[start:].lstrip())
        except Exception:
            continue
        if isinstance(arr, list):
            return _normalize(list(arr))

    # 5) Parse “tolerante” (para casos tipo: Unterminated string / newlines literales)
    try:
        rescued = _extract_translations_loose(candidate)
        if rescued is not None and len(rescued) > 0:
            return _normalize(rescued)
    except Exception as e:
        print(f"[AVISO] Error parseando JSON, se usa fallback. Detalle: {e}")

    # 6) Fallback: devolver las líneas originales
    return fallback_lines
