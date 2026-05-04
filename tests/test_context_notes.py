import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from takoworks.modules.transcriber.context_notes import (  # type: ignore
    build_contextual_explanation_prompt,
    build_contextual_explanation_repair_prompt,
    contains_japanese_script,
    ensure_japanese_furigana,
    get_context_window,
    parse_contextual_explanation_response,
)
from takoworks.modules.transcriber.core import analyze_contextual_note_with_claude  # type: ignore


class _DummyBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _DummyUsage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _DummyMessage:
    def __init__(self, text: str, input_tokens: int = 10, output_tokens: int = 6):
        self.content = [_DummyBlock(text)]
        self.usage = _DummyUsage(input_tokens, output_tokens)


class _DummyMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses[len(self.calls) - 1]


class _DummyClient:
    def __init__(self, responses):
        self.messages = _DummyMessages(responses)


def test_get_context_window_pads_missing_neighbors():
    lines = ["uno", "dos", "tres"]
    assert get_context_window(lines, 0) == ("", "", "uno", "dos", "tres")


def test_prompt_includes_target_and_neighbors():
    lines = ["A", "B", "C", "D", "E"]
    prompt = build_contextual_explanation_prompt("ja", lines, 2)
    assert "Linea -2: A" in prompt
    assert "Linea -1: B" in prompt
    assert "Linea japonesa objetivo: C" in prompt
    assert "Linea +1: D" in prompt
    assert "Linea +2: E" in prompt


def test_prompt_forces_spanish_only_output():
    prompt = build_contextual_explanation_prompt("ja", ["A"], 0)
    assert "Escribe siempre en espanol de Espana" in prompt
    assert "No escribas ninguna parte de la explicacion en japones" in prompt
    assert "anade SIEMPRE su lectura completa en hiragana" not in prompt
    assert "No uses romaji" not in prompt


def test_repair_prompt_explicitly_rewrites_to_spanish():
    prompt = build_contextual_explanation_repair_prompt("ja", ["A"], 0, "日本語の説明")
    assert "Reescribe la siguiente nota contextual al espanol de Espana." in prompt
    assert "no incluir japones" in prompt
    assert "NOTA A CORREGIR" in prompt


def test_contains_japanese_script_detects_kana_and_kanji():
    assert contains_japanese_script("これは日本語です")
    assert contains_japanese_script("かな")
    assert not contains_japanese_script("Esto es solo espanol.")


def test_context_note_analysis_repairs_japanese_output():
    client = _DummyClient(
        [
            _DummyMessage("これは日本語の説明です。"),
            _DummyMessage("La linea se entiende por el contexto y suena coloquial."),
        ]
    )

    note, usage = analyze_contextual_note_with_claude(client, ["a", "b", "c"], 1, "ja")

    assert note == "La linea se entiende por el contexto y suena coloquial."
    assert usage.prompt_tokens == 20
    assert usage.completion_tokens == 12
    assert len(client.messages.calls) == 2
    assert "Reescribe la siguiente nota contextual" in client.messages.calls[1]["messages"][0]["content"]


def test_ensure_japanese_furigana_adds_readings_to_every_kanji_span():
    mapping = {
        "言葉": "ことば",
        "気を付けて": "きをつけて",
    }
    text = "Analiza 言葉 y el matiz de 気を付けて."
    expected = "Analiza 言葉(ことば) y el matiz de 気を付けて(きをつけて)."
    assert ensure_japanese_furigana(text, mapping.get) == expected


def test_ensure_japanese_furigana_preserves_existing_readings():
    mapping = {"言葉": "ことば"}
    text = "Ya viene como 言葉(ことば) en la nota."
    assert ensure_japanese_furigana(text, mapping.get) == text


def test_parse_contextual_response_accepts_braced_text():
    raw = "{Matiza una elipsis coloquial y suena algo brusco en contexto.}"
    assert parse_contextual_explanation_response(raw) == "Matiza una elipsis coloquial y suena algo brusco en contexto."


def test_parse_contextual_response_accepts_json_payload():
    raw = '{"explicacion":"Se sobreentiende el sujeto y el tono es cercano, casi de reproche."}'
    assert parse_contextual_explanation_response(raw) == "Se sobreentiende el sujeto y el tono es cercano, casi de reproche."
