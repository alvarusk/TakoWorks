import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from takoworks.modules.transcriber.context_notes import (  # type: ignore
    build_contextual_explanation_prompt,
    get_context_window,
    parse_contextual_explanation_response,
)


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


def test_prompt_requests_hiragana_readings_for_japanese_terms():
    prompt = build_contextual_explanation_prompt("ja", ["A"], 0)
    assert "sin espacios" in prompt
    assert "言葉(ことば)" in prompt
    assert "No uses romaji para indicar lecturas japonesas." in prompt


def test_parse_contextual_response_accepts_braced_text():
    raw = "{Matiza una elipsis coloquial y suena algo brusco en contexto.}"
    assert parse_contextual_explanation_response(raw) == "Matiza una elipsis coloquial y suena algo brusco en contexto."


def test_parse_contextual_response_accepts_json_payload():
    raw = '{"explicacion":"Se sobreentiende el sujeto y el tono es cercano, casi de reproche."}'
    assert parse_contextual_explanation_response(raw) == "Se sobreentiende el sujeto y el tono es cercano, casi de reproche."
