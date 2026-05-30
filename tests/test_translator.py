import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from takoworks.modules.translator.core import (  # type: ignore
    _load_glossary_csv,
    _make_output_path,
    _normalize_translated_text,
    _restore_ass_text,
    _tokenize_ass_text,
)


def test_make_output_path_rewrites_en_us_and_full():
    out = _make_output_path(r"C:\subs\show.en-us.full.ass")
    assert out.endswith(r"show.es-es.edited.ass")


def test_make_output_path_adds_suffix_when_name_is_unchanged():
    out = _make_output_path(r"C:\subs\show.ass")
    assert out.endswith(r"show_es-es.ass")


def test_load_glossary_csv_skips_header_and_preserves_commas(tmp_path):
    path = tmp_path / "glossary.csv"
    path.write_text(
        "English,Spanish\n\"hello, world\",hola\ncat,gato\n",
        encoding="utf-8",
    )
    csv_text, count = _load_glossary_csv(str(path))

    assert count == 2
    assert '"hello, world",hola' in csv_text
    assert "cat,gato" in csv_text


def test_tokenize_restore_roundtrip_preserves_tags_and_breaks():
    original = r"{\i1}Hello{\i0} ...\NShe said: «bye» and \h space"
    encoded, mapping = _tokenize_ass_text(original, "abc123")
    restored = _restore_ass_text(encoded, mapping)
    assert restored == original
    assert _normalize_translated_text(restored) == "{\\i1}Hello{\\i0} …\\NShe said: \"bye\" and \\h space"
