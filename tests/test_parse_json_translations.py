import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from takoworks.modules.transcriber.json_utils import (  # type: ignore
    parse_json_translations,
    parse_json_translations_result,
)


def test_parse_valid_json_object():
    raw = '{"translations": ["uno", "dos"]}'
    out = parse_json_translations(raw, fallback_lines=["a", "b"])
    assert out == ["uno", "dos"]


def test_parse_list_root():
    raw = '["x", "y", "z"]'
    out = parse_json_translations(raw, fallback_lines=["a", "b", "c"])
    assert out == ["x", "y", "z"]


def test_parse_malformed_uses_fallback_length():
    raw = '{"translations": ["ok", "incomplete"]'  # missing closing brace
    out = parse_json_translations(raw, fallback_lines=["a", "b", "c"])
    assert out == ["ok", "incomplete", "c"]


def test_parse_result_reports_short_block_metadata():
    raw = '{"translations": ["uno", "dos"]}'
    result = parse_json_translations_result(raw, fallback_lines=["a", "b", "c"])
    assert result.translations == ["uno", "dos", "c"]
    assert result.raw_count == 2
    assert result.expected_count == 3
    assert result.parser == "json_object"
    assert result.exact_match is False
    assert result.used_fallback is True
    assert result.missing_indices == [2]


def test_parse_result_reports_extra_items_trimmed():
    raw = '{"translations": ["uno", "dos", "tres"]}'
    result = parse_json_translations_result(raw, fallback_lines=["a", "b"])
    assert result.translations == ["uno", "dos"]
    assert result.raw_count == 3
    assert result.expected_count == 2
    assert result.extra_count == 1
    assert result.exact_match is False
    assert result.used_fallback is False
