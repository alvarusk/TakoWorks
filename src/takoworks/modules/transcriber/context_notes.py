import json
import re
from typing import List, Tuple


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


def build_contextual_explanation_prompt(lang: str, lines: List[str], index: int) -> str:
    line_minus_2, line_minus_1, target_line, line_plus_1, line_plus_2 = get_context_window(lines, index)
    language_name = "japonés" if lang == "ja" else "chino"
    language_adj = "japonesa" if lang == "ja" else "china"
    target_label = "Línea japonesa objetivo" if lang == "ja" else "Línea china objetivo"

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
- No hagas una lista larga ni un análisis académico excesivo.
- No inventes información cultural si no está razonablemente sugerida por la frase o el contexto.
- Si la frase es muy simple, sé breve.
- Si hay ambigüedad, indícala de forma natural y di cuál es la interpretación más probable en este contexto.
- Si aparece una contracción, una partícula final, una forma elidida o una expresión coloquial, explícalo de forma breve y clara.
- Si el orden natural en español difiere mucho del {language_name}, puedes mencionarlo brevemente.
- No uses romaji salvo que sea realmente útil para aclarar un punto.
- Escribe siempre en español de España, natural y claro.

FORMATO DE SALIDA:
Entre una frase y un párrafo entre {{ }}.

REGLAS DE ESTILO PARA "explicacion":
- 1 a 4 frases como máximo.
- Tono claro, docente y natural.
- Debe poder leerse como una nota breve de subtitulación.
- No empieces con “Esta frase significa...”.
- No uses viñetas, numeración ni encabezados.
- No incluyas nada fuera del JSON.

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
