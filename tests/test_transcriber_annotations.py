import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from takoworks.modules.transcriber import core


@pytest.mark.parametrize("roman,notes", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("with_asr", [False, True])
def test_annotations_run_independently(monkeypatch, roman, notes, with_asr):
    subs = core.pysubs2.SSAFile()
    subs.append(core.pysubs2.SSAEvent(start=0, end=1000, text="source"))
    calls = []

    def romanize(lines, lang):
        calls.append("roman")
        return ["romaji"], core.ApiUsage(engine="deepseek", model_name="test"), None

    def contextualize(lines, lang, usage_accumulator=None):
        calls.append("notes")
        return ["context note"]

    monkeypatch.setattr(core, "romanize_with_deepseek", romanize)
    monkeypatch.setattr(core, "build_contextual_notes", contextualize)
    if with_asr:
        monkeypatch.setattr(core.pysubs2, "load", lambda *a, **kw: subs)
        monkeypatch.setattr(core, "extract_segment", lambda *a: None)
        monkeypatch.setattr(core, "build_asr_pipeline", lambda lang: lambda path: {"text": "source"})
        monkeypatch.setattr(core, "refine_punctuation_free", lambda lines, lang: lines)
        result = core.transcribe_ass(
            "input.ass", "video.mp4", 0, "ja", False,
            do_romanization=roman, do_context_note=notes,
        )
    else:
        subs[0].text += r"\Nexisting translation"
        result = core.add_roman_morph_to_subs(
            subs, "ja", do_romanization=roman, do_context_note=notes,
        )
        assert "existing translation" in result[0].text

    assert calls == (["roman"] if roman else []) + (["notes"] if notes else [])
    assert ("{romaji}" in result[0].text) == roman
    assert ("{context note}" in result[0].text) == notes


def test_context_note_without_romaji_has_correct_html_column(tmp_path):
    subs = core.pysubs2.SSAFile()
    subs.append(core.pysubs2.SSAEvent(text=r"source\N{context note}"))
    output = tmp_path / "summary.html"
    core.generate_html(subs, {}, str(output), has_romanization=False)
    html = output.read_text(encoding="utf-8")
    assert "<td>source</td><td></td><td>context note</td>" in html
