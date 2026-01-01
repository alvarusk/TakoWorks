import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from takoworks.modules.transcriber.core import parse_json_translations  # type: ignore


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
