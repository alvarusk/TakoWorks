import json
import re
from functools import lru_cache
from typing import Callable, List, Optional, Tuple


KANJI_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\u3005\u3006\u30FC]")
JAPANESE_SPAN_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\u3005\u3006\u30FC\u3040-\u30FF]+")
JAPANESE_SCRIPT_RE = re.compile(r"[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\u3005\u3006\u30FC]")
CHINESE_SPAN_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF]+")
CHINESE_FORBIDDEN_SCRIPT_RE = re.compile(r"[\u3040-\u30FF]")


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


def contains_forbidden_chinese_script(text: str) -> bool:
    """Return whether text contains kana, which is not expected in a Chinese note."""
    return bool(CHINESE_FORBIDDEN_SCRIPT_RE.search(text or ""))


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


@lru_cache(maxsize=1)
def _get_default_pinyin_provider() -> Optional[Callable[[str], str]]:
    try:
        from pypinyin import lazy_pinyin, Style
    except Exception:
        return None

    def _provider(span: str) -> str:
        try:
            return " ".join(lazy_pinyin(span, style=Style.TONE, errors="ignore")).strip()
        except Exception:
            return ""

    return _provider


def ensure_chinese_pinyin(
    text: str,
    pinyin_provider: Optional[Callable[[str], str]] = None,
) -> str:
    """Add tone-marked pinyin after every hanzi span mentioned in a note."""
    raw = text or ""
    if not raw:
        return raw

    provider = pinyin_provider or _get_default_pinyin_provider()
    if provider is None:
        return raw

    out: List[str] = []
    last = 0
    for match in CHINESE_SPAN_RE.finditer(raw):
        start, end = match.span()
        if end < len(raw) and raw[end] in ("(", "\uff08"):
            continue

        pinyin = (provider(match.group(0)) or "").strip()
        if not pinyin:
            continue

        out.append(raw[last:start])
        out.append(f"{match.group(0)}({pinyin})")
        last = end

    if last == 0:
        return raw
    out.append(raw[last:])
    return "".join(out)


def build_contextual_explanation_prompt(lang: str, lines: List[str], index: int) -> str:
    line_minus_2, line_minus_1, target_line, line_plus_1, line_plus_2 = get_context_window(lines, index)
    language_name = "japonés" if lang == "ja" else "chino"
    language_adj = "japonesa" if lang == "ja" else "china"
    target_label = "Línea japonesa objetivo" if lang == "ja" else "Línea china objetivo"
    original_term_rules = (
        "- En Vocabulario, escribe el furigana en hiragana en la línea inmediatamente superior al término japonés y después `: definición en español`.\n"
        if lang == "ja"
        else "- En Vocabulario, escribe el hanzi seguido inmediatamente de su pinyin con tonos entre paréntesis y después su definición en español; formato: término (pinyin): definición.\n"
    )
    script_rule = (
        "No escribas ninguna parte de la explicación en japonés, chino, hiragana, katakana, kanji ni romaji, salvo los términos originales que debas citar siguiendo la regla anterior."
        if lang == "ja"
        else "Escribe la explicación en español de España. Puedes citar hanzi solo para definir palabras o expresiones relevantes, y cada cita debe llevar su pinyin entre paréntesis. No uses caracteres japoneses kana."
    )

    return f"""Eres un profesor experto de {language_name} para hispanohablantes y un analista de guion audiovisual.

Tu tarea es analizar SOLO la línea {language_adj} objetivo y explicarla en español de forma útil para subtitulación. Debes tener en cuenta las dos líneas anteriores y las dos posteriores únicamente como contexto para desambiguar tono, referente, elipsis, intención, registro y posibles implicaciones culturales.

IMPORTANTE:
- No traduzcas todas las líneas del contexto: analízalas solo para entender mejor la línea objetivo.
- Tu explicación debe centrarse en la línea objetivo.
- El resultado debe ser breve: entre una línea y un pequeño párrafo.
- Debes combinar, cuando sea relevante, estos tres planos:
  1. semántico: qué quiere decir realmente la frase en contexto;
  2. sintáctico: cómo está construida y qué función cumplen las partes importantes;
  3. cultural/pragmático: matices de registro, implicaturas, relaciones entre personajes, referencias culturales o usos típicos del {language_name}.
- Incluye entre 2 y 5 términos de vocabulario relevantes.
- No inventes información cultural si no está razonablemente sugerida por la frase o el contexto.
- Si la frase es muy simple, sé breve.
- Si hay ambigüedad, indícala de forma natural y di cuál es la interpretación más probable en este contexto.
- Si aparece una contracción, una partícula final, una forma elidida o una expresión coloquial, explícalo de forma breve y clara.
- Si el orden natural en español difiere mucho del {language_name}, puedes mencionarlo brevemente.
- Escribe siempre en español de España, natural y claro.
- {script_rule}
- Si necesitas mencionar un elemento del original, parafrásalo o tradúcelo al español.
{original_term_rules}

FORMATO DE SALIDA (obligatorio):
Explicación: 1 a 4 frases que combinen gramática y significado semántico con el contexto.
Vocabulario:
- [furigana en hiragana]
  [término japonés]: [traducción o definición en español]
- [furigana en hiragana]
  [término japonés]: [traducción o definición en español]

REGLAS DE ESTILO PARA "explicación":
- 1 a 4 frases como maximo.
- Tono claro, docente y natural.
- Debe poder leerse como una nota breve de subtitulacion.
- No empieces con "Esta frase significa...".
- Usa exactamente los encabezados "Explicación:" y "Vocabulario:".
- Usa viñetas solo en la lista de vocabulario.
- No incluyas nada fuera de esta estructura.

CONTEXTO:
Línea -2: {line_minus_2}
Línea -1: {line_minus_1}
{target_label}: {target_line}
Línea +1: {line_plus_1}
Línea +2: {line_plus_2}"""


def build_contextual_explanation_repair_prompt(lang: str, lines: List[str], index: int, note: str) -> str:
    line_minus_2, line_minus_1, target_line, line_plus_1, line_plus_2 = get_context_window(lines, index)
    language_name = "japonés" if lang == "ja" else "chino"
    language_adj = "japonesa" if lang == "ja" else "china"
    target_label = "Línea japonesa objetivo" if lang == "ja" else "Línea china objetivo"
    note_text = _normalize_context_line(note)
    script_rule = (
        "no incluir japonés, chino, hiragana, katakana, kanji ni romaji, salvo términos japoneses citados con su lectura en hiragana;"
        if lang == "ja"
        else "no incluir kana ni caracteres japoneses; si conserva una palabra en hanzi, añadir inmediatamente su pinyin con tonos entre paréntesis;"
    )

    return f"""Reescribe la siguiente nota contextual al español de España.

La version final debe:
- conservar el sentido, el tono y la brevedad de la nota original;
- conservar exactamente los encabezados "Explicación:" y "Vocabulario:";
- conservar la lista de vocabulario con el furigana en hiragana encima de cada término japonés;
- sonar natural para subtitulacion;
- {script_rule}
- devolver solo la nota final, sin explicaciones sobre el cambio y sin JSON.

Eres un profesor experto de {language_name} para hispanohablantes y un analista de guion audiovisual.
Tu tarea es reexpresar la nota, no analizar de nuevo el contexto.

NOTA A CORREGIR:
{note_text}

CONTEXTO:
Línea -2: {line_minus_2}
Línea -1: {line_minus_1}
{target_label}: {target_line}
Línea +1: {line_plus_1}
Línea +2: {line_plus_2}"""


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
        for key in ("explicación", "explicacion", "explicaciÃ³n", "analysis", "note", "nota"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                text = value.strip()
                break

    if text.startswith("{") and text.endswith("}"):
        inner = text[1:-1].strip()
        if inner and not re.search(r'"[^"]+"\s*:', inner):
            text = inner

    return text.strip()
