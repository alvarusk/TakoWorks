import json
import re
from functools import lru_cache
from typing import Callable, List, Optional, Tuple


KANJI_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\u3005\u3006\u30FC]")
JAPANESE_SPAN_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\u3005\u3006\u30FC\u3040-\u30FF]+")
JAPANESE_SCRIPT_RE = re.compile(r"[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\u3005\u3006\u30FC]")


def _normalize_context_line(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def get_context_window(lines: List[str], index: int) -> Tuple[str, str, str, str, str]:
    def pick(offset: int) -> str:
        pos = index + offset
        if 0 <= pos < len(lines):
            return _normalize_context_line(lines[pos])
        return ""

    return (
        pick(-2),
        pick(-1),
        pick(0),
        pick(1),
        pick(2),
    )


def _contains_kanji(text: str) -> bool:
    return bool(KANJI_RE.search(text or ""))


def contains_japanese_script(text: str) -> bool:
    return bool(JAPANESE_SCRIPT_RE.search(text or ""))


@lru_cache(maxsize=1)
def _get_default_reading_provider() -> Optional[Callable[[str], str]]:
    try:
        from pykakasi import kakasi
    except Exception:
        return None

    kks = kakasi()

    def _provider(span: str) -> str:
        try:
            parts = kks.convert(span)
        except Exception:
            return ""
        reading = "".join(
            (
                item.get("hira")
                or item.get("kana")
                or item.get("orig")
                or ""
            )
            if isinstance(item, dict)
            else str(item)
            for item in parts
        )
        return (reading or "").strip()

    return _provider


def ensure_japanese_furigana(
    text: str,
    reading_provider: Optional[Callable[[str], str]] = None,
) -> str:
    raw = text or ""
    if not raw or not _contains_kanji(raw):
        return raw

    provider = reading_provider or _get_default_reading_provider()
    if provider is None:
        return raw

    out: List[str] = []
    last = 0

    for match in JAPANESE_SPAN_RE.finditer(raw):
        start, end = match.span()
        span = match.group(0)

        if not _contains_kanji(span):
            continue
        if end < len(raw) and raw[end] in ("(", "ï¼ˆ"):
            continue

        reading = (provider(span) or "").strip()
        if not reading:
            continue

        out.append(raw[last:start])
        out.append(f"{span}({reading})")
        last = end

    if last == 0:
        return raw

    out.append(raw[last:])
    return "".join(out)


def build_contextual_explanation_prompt(lang: str, lines: List[str], index: int) -> str:
    line_minus_2, line_minus_1, target_line, line_plus_1, line_plus_2 = get_context_window(lines, index)
    language_name = "japones" if lang == "ja" else "chino"
    language_adj = "japonesa" if lang == "ja" else "china"
    target_label = "Linea japonesa objetivo" if lang == "ja" else "Linea china objetivo"

    return f"""Eres un profesor experto de {language_name} para hispanohablantes y un analista de guion audiovisual.

Tu tarea es analizar SOLO la linea {language_adj} objetivo y explicarla en espanol de forma util para subtitulacion. Debes tener en cuenta las dos lineas anteriores y las dos posteriores unicamente como contexto para desambiguar tono, referente, elipsis, intencion, registro y posibles implicaciones culturales.

IMPORTANTE:
- No traduzcas todas las lineas del contexto: analizalas solo para entender mejor la linea objetivo.
- Tu explicacion debe centrarse en la linea objetivo.
- El resultado debe ser breve: entre una linea y un pequeno parrafo.
- Debes combinar, cuando sea relevante, estos tres planos:
  1. semantico: que quiere decir realmente la frase en contexto;
  2. sintactico: como esta construida y que funcion cumplen las partes importantes;
  3. cultural/pragmatico: matices de registro, implicaturas, relaciones entre personajes, referencias culturales o usos tipicos del {language_name}.
- No hagas una lista larga ni un analisis academico excesivo.
- No inventes informacion cultural si no esta razonablemente sugerida por la frase o el contexto.
- Si la frase es muy simple, se breve.
- Si hay ambiguedad, indicala de forma natural y di cual es la interpretacion mas probable en este contexto.
- Si aparece una contraccion, una particula final, una forma elidida o una expresion coloquial, explicalo de forma breve y clara.
- Si el orden natural en espanol difiere mucho del {language_name}, puedes mencionarlo brevemente.
- Escribe siempre en espanol de Espana, natural y claro.
- No escribas ninguna parte de la explicacion en japones, chino, hiragana, katakana, kanji ni romaji.
- Si necesitas mencionar un elemento del original, parfrasealo o traducelo al espanol.

FORMATO DE SALIDA:
Devuelve solo la nota final, sin encabezados, sin viñetas, sin bloques de codigo y sin JSON.

REGLAS DE ESTILO PARA "explicacion":
- 1 a 4 frases como maximo.
- Tono claro, docente y natural.
- Debe poder leerse como una nota breve de subtitulacion.
- No empieces con "Esta frase significa...".
- No uses vinetas, numeracion ni encabezados.
- No incluyas nada fuera del contenido pedido.

CONTEXTO:
Linea -2: {line_minus_2}
Linea -1: {line_minus_1}
{target_label}: {target_line}
Linea +1: {line_plus_1}
Linea +2: {line_plus_2}"""


def build_contextual_explanation_repair_prompt(lang: str, lines: List[str], index: int, note: str) -> str:
    line_minus_2, line_minus_1, target_line, line_plus_1, line_plus_2 = get_context_window(lines, index)
    language_name = "japones" if lang == "ja" else "chino"
    language_adj = "japonesa" if lang == "ja" else "china"
    target_label = "Linea japonesa objetivo" if lang == "ja" else "Linea china objetivo"
    note_text = _normalize_context_line(note)

    return f"""Reescribe la siguiente nota contextual al espanol de Espana.

La version final debe:
- conservar el sentido, el tono y la brevedad de la nota original;
- sonar natural para subtitulacion;
- no incluir japones, chino, hiragana, katakana, kanji ni romaji;
- devolver solo la nota final, sin explicaciones sobre el cambio y sin JSON.

Eres un profesor experto de {language_name} para hispanohablantes y un analista de guion audiovisual.
Tu tarea es reexpresar la nota, no analizar de nuevo el contexto.

NOTA A CORREGIR:
{note_text}

CONTEXTO:
Linea -2: {line_minus_2}
Linea -1: {line_minus_1}
{target_label}: {target_line}
Linea +1: {line_plus_1}
Linea +2: {line_plus_2}"""


def parse_contextual_explanation_response(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""

    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    try:
        payload = json.loads(text)
    except Exception:
        payload = None
        json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if json_match:
            try:
                payload = json.loads(json_match.group(0))
            except Exception:
                payload = None

    if isinstance(payload, dict):
        for key in ("explicacion", "explicación", "analysis", "note", "nota"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                text = value.strip()
                break

    if text.startswith("{") and text.endswith("}"):
        inner = text[1:-1].strip()
        if inner and not re.search(r'"[^"]+"\s*:', inner):
            text = inner

    return text.strip()
