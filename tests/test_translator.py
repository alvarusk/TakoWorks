import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from takoworks.modules.translator import core as translator_core  # type: ignore
from takoworks.modules.translator.core import (  # type: ignore
    _load_glossary_csv,
    _make_output_path,
    _normalize_translated_text,
    _restore_ass_text,
    _tokenize_ass_text,
    translate_ass_file,
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


def test_load_glossary_csv_accepts_semicolon_delimited_files(tmp_path):
    path = tmp_path / "glossary.csv"
    path.write_text(
        "en;es\nhello;hola\ncat;gato\n",
        encoding="utf-8",
    )
    csv_text, count = _load_glossary_csv(str(path))

    assert count == 2
    assert "hello,hola" in csv_text
    assert "cat,gato" in csv_text


def test_translate_ass_file_runs_without_glossary(tmp_path, monkeypatch):
    out_path = tmp_path / "out.ass"
    calls = {}

    def fake_parse_ass(_path):
        return (
            ["Dialogue: 0,0,0,0,0,0,0,0,Hello\n"],
            [
                {
                    "prefix": "Dialogue",
                    "parts": ["0", "0", "0", "0", "0", "0", "0", "0", "Hello"],
                    "text_idx": 8,
                    "line_index": 0,
                    "leading": "",
                    "line_ending": "\n",
                }
            ],
        )

    class FakeClient:
        def __init__(self, auth_key):
            calls["auth_key"] = auth_key
            self.base_url = "https://example.test"

        def translate_batch(self, texts, *, glossary_id=None, source_lang="EN", target_lang="ES"):
            calls["translate_batch"] = {
                "texts": list(texts),
                "glossary_id": glossary_id,
                "source_lang": source_lang,
                "target_lang": target_lang,
            }
            return ["Hola" for _ in texts]

        def create_glossary(self, **kwargs):
            calls["create_glossary"] = kwargs
            raise AssertionError("create_glossary should not be called without a glossary")

        def wait_glossary_ready(self, glossary_id):
            calls["wait_glossary_ready"] = glossary_id
            raise AssertionError("wait_glossary_ready should not be called without a glossary")

        def delete_glossary(self, glossary_id):
            calls["delete_glossary"] = glossary_id
            raise AssertionError("delete_glossary should not be called without a glossary")

    monkeypatch.setattr(translator_core, "parse_ass", fake_parse_ass)
    monkeypatch.setattr(translator_core, "_read_api_key", lambda: "test-key")
    monkeypatch.setattr(translator_core, "DeepLClient", FakeClient)

    result = translate_ass_file("input.ass", None, str(out_path), log=lambda *args, **kwargs: None)

    assert result == str(out_path)
    assert out_path.read_text(encoding="utf-8-sig") == "Dialogue: 0,0,0,0,0,0,0,0,Hola\n"
    assert calls["auth_key"] == "test-key"
    assert calls["translate_batch"]["glossary_id"] is None
    assert "create_glossary" not in calls
    assert "wait_glossary_ready" not in calls
    assert "delete_glossary" not in calls


def test_tokenize_restore_roundtrip_preserves_tags_and_breaks():
    original = r"{\i1}Hello{\i0} ...\NShe said: «bye» and \h space"
    encoded, mapping = _tokenize_ass_text(original, "abc123")
    restored = _restore_ass_text(encoded, mapping)
    assert restored == original
    assert _normalize_translated_text(restored) == "{\\i1}Hello{\\i0} …\\NShe said: \"bye\" and \\h space"
