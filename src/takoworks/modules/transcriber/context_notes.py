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
    language_name = "japones" if lang == "ja" else "chino"
    language_adj = "japonesa" if lang == "ja" else "china"
    target_label = "Linea japonesa objetivo" if lang == "ja" else "Linea china objetivo"

    if lang == "ja":
        reading_rules = (
            "- Si citas o analizas una palabra o expresion japonesa que contenga kanji, anade siempre su lectura en hiragana justo despues, entre parentesis y sin espacios, con este patron exacto: 言葉(ことば), 気を付けて(きをつけて).\n"
            "- No hace falta anadir lectura a palabras escritas solo en kana.\n"
            "- No uses romaji para indicar lecturas japonesas."
        )
    else:
        reading_rules = "- No uses romaji salvo que sea realmente util para aclarar un punto."

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
- {reading_rules}
- Escribe siempre en espanol de Espana, natural y claro.

FORMATO DE SALIDA:
Entre una frase y un parrafo entre {{ }}.

REGLAS DE ESTILO PARA "explicacion":
- 1 a 4 frases como maximo.
- Tono claro, docente y natural.
- Debe poder leerse como una nota breve de subtitulacion.
- No empieces con "Esta frase significa...".
- No uses vinetas, numeracion ni encabezados.
- No incluyas nada fuera del JSON.

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
