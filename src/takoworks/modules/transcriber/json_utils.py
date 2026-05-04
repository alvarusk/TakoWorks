import json
import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TranslationParseResult:
    translations: List[str]
    expected_count: int
    raw_count: int
    parser: str
    exact_match: bool
    normalized: bool
    used_fallback: bool
    error: Optional[str] = None

    @property
    def missing_indices(self) -> List[int]:
        if self.raw_count >= self.expected_count:
            return []
        return list(range(self.raw_count, self.expected_count))

    @property
    def extra_count(self) -> int:
        return max(0, self.raw_count - self.expected_count)


def parse_json_translations_result(raw_content: str, fallback_lines: List[str]) -> TranslationParseResult:
    """
    Extrae un array de traducciones desde un JSON con forma:
      {"translations": ["...", "...", ...]}
    Aplica heuristicas basicas para limpiar fences ``` ``` y saltos de linea
    insertados. Si no se puede parsear, devuelve las lineas originales
    (fallback_lines), junto con metadatos de diagnostico.
    """
    expected_count = len(fallback_lines)
    raw = (raw_content or "").strip()

    def _result(
        translations: List[str],
        *,
        parser: str,
        error: Optional[str] = None,
    ) -> TranslationParseResult:
        raw_count = len(translations)
        cooked = [("" if t is None else str(t)) for t in translations]
        exact_match = raw_count == expected_count
        normalized = not exact_match
        used_fallback = False

        if normalized:
            print("[AVISO] Nº de traducciones != nº de lineas. Se ajusta al minimo en comun.")
            if raw_count > expected_count:
                cooked = cooked[:expected_count]
            else:
                cooked = cooked + fallback_lines[raw_count:]
                used_fallback = True

        return TranslationParseResult(
            translations=cooked,
            expected_count=expected_count,
            raw_count=raw_count,
            parser=parser,
            exact_match=exact_match,
            normalized=normalized,
            used_fallback=used_fallback or parser == "fallback",
            error=error,
        )

    def _fix_invalid_backslashes(s: str) -> str:
        # En JSON, solo son validos: \" \\ \/ \b \f \n \r \t \uXXXX
        # Esto rescata cosas tipicas como \N (ASS) o \an8, etc.
        return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)

    def _extract_translations_loose(s: str) -> Optional[List[str]]:
        """
        Extrae strings del array translations con un parser ligero, tolerante a:
        - saltos de linea literales dentro de strings
        - comas / whitespace extra
        """
        m = re.search(r'"translations"\s*:\s*\[', s)
        quote_mode = '"'
        if not m:
            m = re.search(r"'translations'\s*:\s*\[", s)
            quote_mode = "'"
        if not m:
            return None

        i = m.end()
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
                    items.append("".join(buf))
                    buf = []
                    in_str = False
                    q = ""
                else:
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

        out: List[str] = []
        for raw_item in items:
            fixed = raw_item.replace("\r\n", "\n").replace("\r", "\n")
            fixed = fixed.replace("\n", "\\n")
            fixed = _fix_invalid_backslashes(fixed)

            try:
                if quote_mode == '"':
                    out.append(json.loads('"' + fixed + '"'))
                    continue
            except Exception:
                pass

            fixed2 = fixed.replace(r"\n", "\n").replace(r"\t", "\t").replace(r"\r", "\r")
            fixed2 = fixed2.replace(r"\\", "\\").replace(r"\/", "/").replace(r"\"", '"')
            out.append(fixed2)

        return out

    if not raw:
        return _result(list(fallback_lines), parser="fallback", error="empty_response")

    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    first = raw.find("{")
    last = raw.rfind("}")
    candidate = raw[first:last + 1] if first != -1 and last != -1 and last > first else raw

    try:
        data = json.loads(candidate)
        if isinstance(data, dict) and isinstance(data.get("translations"), list):
            return _result(list(data["translations"]), parser="json_object")
        if isinstance(data, list):
            return _result(list(data), parser="json_array")
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for m in re.finditer(r'"translations"\s*:', candidate):
        start = m.end()
        try:
            arr, _ = decoder.raw_decode(candidate[start:].lstrip())
        except Exception:
            continue
        if isinstance(arr, list):
            return _result(list(arr), parser="decoder_array")

    try:
        rescued = _extract_translations_loose(candidate)
        if rescued is not None and len(rescued) > 0:
            return _result(rescued, parser="loose_array")
    except Exception as e:
        print(f"[AVISO] Error parseando JSON, se usa fallback. Detalle: {e}")
        return _result(list(fallback_lines), parser="fallback", error=str(e))

    return _result(list(fallback_lines), parser="fallback", error="unparseable_response")


def parse_json_translations(raw_content: str, fallback_lines: List[str]) -> List[str]:
    return parse_json_translations_result(raw_content, fallback_lines).translations


@dataclass
class RomanizationParseResult:
    romanizations: List[str]
    expected_count: int
    raw_count: int
    parser: str
    exact_match: bool
    normalized: bool
    used_fallback: bool
    error: Optional[str] = None

    @property
    def missing_indices(self) -> List[int]:
        if self.raw_count >= self.expected_count:
            return []
        return list(range(self.raw_count, self.expected_count))

    @property
    def extra_count(self) -> int:
        return max(0, self.raw_count - self.expected_count)


def parse_json_romanizations_result(raw_content: str, fallback_lines: List[str]) -> RomanizationParseResult:
    """
    Extrae un array de romanizaciones desde un JSON con forma:
      {"romanizations": ["...", "...", ...]}
    o un JSON array simple.
    """
    expected_count = len(fallback_lines)
    raw = (raw_content or "").strip()

    def _result(
        romanizations: List[str],
        *,
        parser: str,
        error: Optional[str] = None,
    ) -> RomanizationParseResult:
        raw_count = len(romanizations)
        cooked = [("" if t is None else str(t)) for t in romanizations]
        exact_match = raw_count == expected_count
        normalized = not exact_match
        used_fallback = False

        if normalized:
            print("[AVISO] N\u00ba de romanizaciones != n\u00ba de lineas. Se ajusta al minimo en comun.")
            if raw_count > expected_count:
                cooked = cooked[:expected_count]
            else:
                cooked = cooked + fallback_lines[raw_count:]
                used_fallback = True

        return RomanizationParseResult(
            romanizations=cooked,
            expected_count=expected_count,
            raw_count=raw_count,
            parser=parser,
            exact_match=exact_match,
            normalized=normalized,
            used_fallback=used_fallback or parser == "fallback",
            error=error,
        )

    if not raw:
        return _result(list(fallback_lines), parser="fallback", error="empty_response")

    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    first = raw.find("{")
    last = raw.rfind("}")
    candidate = raw[first:last + 1] if first != -1 and last != -1 and last > first else raw

    try:
        data = json.loads(candidate)
        if isinstance(data, dict) and isinstance(data.get("romanizations"), list):
            return _result(list(data["romanizations"]), parser="json_object")
        if isinstance(data, list):
            return _result(list(data), parser="json_array")
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for m in re.finditer(r'"romanizations"\s*:', candidate):
        start = m.end()
        try:
            arr, _ = decoder.raw_decode(candidate[start:].lstrip())
        except Exception:
            continue
        if isinstance(arr, list):
            return _result(list(arr), parser="decoder_array")

    return _result(list(fallback_lines), parser="fallback", error="unparseable_response")
