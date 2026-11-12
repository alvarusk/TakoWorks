import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from takoworks.shared import final_pass as fp


def write_ass(path, text="Hello", duration="0:00:02.00"):
    path.write_text(
        "[Script Info]\nTitle: Keep this title\nScriptType: v4.00+\n\n"
        "[V4+ Styles]\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Comment: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,Keep this comment\n"
        f"Dialogue: 0,0:00:00.00,{duration},Default,Alice,0,0,0,,{text}\n",
        encoding="utf-8-sig",
    )
    return path


def fake_client(responses):
    calls = []
    results = iter(responses)

    def create(**kwargs):
        # Snapshot mutable messages before retries append to them.
        calls.append(json.loads(json.dumps(kwargs)))
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return NS(choices=[NS(message=NS(content=json.dumps(result)), finish_reason="stop")],
                  usage=NS(prompt_tokens=10, completion_tokens=5))

    return NS(chat=NS(completions=NS(create=create))), calls


def answer(text, notes=()):
    return {"subtitles": [{"id": 1, "text": text, "notes": list(notes)}], "notes": []}


def test_limits_measure_rendered_text_and_exact_boundaries():
    text = r"{\i1}" + "x" * 40 + r"\N" + "x" * 40 + r"{\i0}"
    assert fp.validate_subtitle(text, 4000, [r"{\i1}", r"{\i0}"]) == []
    assert any("CPS" in p for p in fp.validate_subtitle(text, 3990, [r"{\i1}", r"{\i0}"]))
    assert any("CPL" in p for p in fp.validate_subtitle("x" * 41, 10000))
    assert any("lineas" in p for p in fp.validate_subtitle(r"Uno\NDos\NTres", 10000))
    assert any("funcional" in p for p in fp.validate_subtitle(r"Vamos a\Ncomer", 2000))
    assert any("Duracion" in p for p in fp.validate_subtitle("Hola", 0))
    assert any("Llaves" in p for p in fp.validate_subtitle(r"Hola{\i1", 1000))


def test_final_pass_repairs_limits_preserves_metadata_and_opens_notes(tmp_path, monkeypatch):
    source = write_ass(tmp_path / "show.en-us.ass", r"{\i1}Hello{\i0}")
    original = source.read_bytes()
    output = tmp_path / "show_final.ass"
    client, calls = fake_client([answer("x" * 90), answer(r"{\i1}Hola.{\i0}", ["Revisar el tratamiento."])])
    opened = []
    monkeypatch.setattr(fp.subprocess, "Popen", lambda args: opened.append(args))
    usage = fp.run_final_pass(source, {"deepl": [r"{\i1}Hola{\i0}"]}, output,
                              source_lang="EN", client=client)
    assert len(calls) == 2
    assert calls[0]["model"] == fp.OPENAI_MODEL
    assert calls[0]["response_format"]["json_schema"]["strict"] is True
    assert usage == {"prompt_tokens": 20, "completion_tokens": 10}
    assert source.read_bytes() == original
    result = output.read_text(encoding="utf-8-sig")
    assert "Title: Keep this title" in result
    assert "Keep this comment" in result
    assert r"Default,Alice,0,0,0,,{\i1}Hola.{\i0}" in result
    report = output.with_suffix(".txt").read_text(encoding="utf-8-sig")
    assert "Revisar el tratamiento" in report
    assert "Limites tecnicos verificados" in report
    if fp.os.name == "nt":
        assert opened == [["notepad.exe", str(output.with_suffix(".txt").resolve())]]


def test_unresolved_limits_are_reported_without_truncating(tmp_path):
    source = write_ass(tmp_path / "source.ass", duration="0:00:00.10")
    output = tmp_path / "source_final.ass"
    client, calls = fake_client([answer("Hola.", ["No se puede abreviar mas."])] * 3)
    fp.run_final_pass(source, {"gpt": ["Hola."]}, output, source_lang="ja", client=client, open_notes=False)
    assert len(calls) == 3
    assert "Hola." in output.read_text(encoding="utf-8-sig")
    report = output.with_suffix(".txt").read_text(encoding="utf-8-sig")
    assert "50.00 CPS" in report
    assert "REQUIERE REVISION" in report
    assert "No se puede abreviar" in report


@pytest.mark.parametrize("result", [RuntimeError("API unavailable"),
                                   {"subtitles": [], "notes": []}, answer(r"Hola{\p1}")])
def test_api_or_structural_failure_keeps_existing_final_and_writes_report(tmp_path, result):
    source = write_ass(tmp_path / "source.ass")
    output = tmp_path / "source_final.ass"
    output.write_text("previous final", encoding="utf-8")
    client, calls = fake_client([result] * 3)
    with pytest.raises(RuntimeError):
        fp.run_final_pass(source, {"gpt": ["Hola"]}, output, source_lang="zh", client=client, open_notes=False)
    assert len(calls) == 3
    assert output.read_text(encoding="utf-8") == "previous final"
    assert "NO COMPLETADO" in output.with_suffix(".txt").read_text(encoding="utf-8-sig")
    assert not list(tmp_path.glob("*.tmp"))


def test_cancellation_writes_notes_without_calling_api(tmp_path):
    source = write_ass(tmp_path / "source.ass")
    output = tmp_path / "source_final.ass"
    client, calls = fake_client([])
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(RuntimeError, match="Cancelado"):
        fp.run_final_pass(source, {"gpt": ["Hola"]}, output, source_lang="EN", client=client,
                          cancel_event=cancelled, open_notes=False)
    assert calls == []
    assert not output.exists()
    assert "Cancelado" in output.with_suffix(".txt").read_text(encoding="utf-8-sig")


def test_batches_include_neighbor_context_and_asr_notes(tmp_path):
    source = write_ass(tmp_path / "source.ass", r"Original{Explicacion semantica\Ngramatica}")
    content = source.read_text(encoding="utf-8-sig")
    dialogue = content.splitlines()[-1]
    source.write_text(content + (dialogue + "\n") * 20, encoding="utf-8-sig")
    answers = [{"subtitles": [{"id": i, "text": "Hola.", "notes": []} for i in range(1, 21)], "notes": []},
               {"subtitles": [{"id": 21, "text": "Hola.", "notes": []}], "notes": []}]
    client, calls = fake_client(answers)
    fp.run_final_pass(source, {"gpt": ["Hola"] * 21, "claude": ["Buenas"] * 21},
                      tmp_path / "final.ass", source_lang="ja", client=client, open_notes=False)
    first = json.loads(calls[0]["messages"][1]["content"])
    second = json.loads(calls[1]["messages"][1]["content"])
    assert "Explicacion semantica" in first["subtitles"][0]["reference_ass"]
    assert first["subtitles"][0]["translations"] == {"gpt": "Hola", "claude": "Buenas"}
    assert first["context_after"][0]["id"] == 21
    assert [r["id"] for r in second["context_before"]] == [19, 20]


@pytest.mark.parametrize("lang", ["EN", "FR"])
def test_translator_runs_review_after_deepl(tmp_path, monkeypatch, lang):
    from takoworks.modules.translator import core

    source = write_ass(tmp_path / f"show.{lang.lower()}.ass")
    client, calls = fake_client([answer("Hola.")])
    monkeypatch.setattr(fp, "create_openai_client", lambda: client)
    monkeypatch.setattr(fp.subprocess, "Popen", lambda args: None)
    monkeypatch.setattr(core, "DeepLClient", lambda **kwargs: NS(
        base_url="test", translate_batch=lambda texts, **kwargs: ["Buenas."]))
    translated = Path(core.translate_ass_file(str(source), auth_key="test", source_lang=lang, final_pass=True))
    assert "Buenas." in translated.read_text(encoding="utf-8-sig")
    assert "Hola." in translated.with_name(translated.stem + "_final.ass").read_text(encoding="utf-8-sig")
    payload = json.loads(calls[0]["messages"][1]["content"])
    assert payload["source_lang"] == lang
    assert payload["subtitles"][0]["translations"] == {"deepl": "Buenas."}


@pytest.mark.parametrize("lang", ["ja", "zh"])
def test_transcriber_reviews_all_translations_after_last_engine(tmp_path, monkeypatch, lang):
    from takoworks.modules.transcriber import core

    source = write_ass(tmp_path / "source.ass", "Original{Explicacion semantica}")
    order = []
    for engine in ("openai", "claude", "gemini", "deepseek"):
        def translate(*args, _engine=engine, **kwargs):
            order.append(_engine)
            return ["Hola."], core.ApiUsage(engine=_engine, model_name=_engine), None
        monkeypatch.setattr(core, "translate_with_" + engine, translate)
    client, calls = fake_client([answer("Buenas.")])
    def create_client():
        assert order == ["openai", "claude", "gemini", "deepseek"]
        return client
    monkeypatch.setattr(fp, "create_openai_client", create_client)
    monkeypatch.setattr(fp.subprocess, "Popen", lambda args: None)
    monkeypatch.setattr(core, "persist_costs_to_supabase", lambda *args: None)
    monkeypatch.setattr(core, "_load_price_table", lambda: core.DEFAULT_PRICE_PER_1K)
    core.main([str(source), "--skip-asr", "--lang", lang, "--series", "Test", "--source-type", "None", "--final-pass"])
    assert (tmp_path / "source_final.ass").exists()
    payload = json.loads(calls[0]["messages"][1]["content"])
    assert set(payload["subtitles"][0]["translations"]) == {"gpt", "claude", "gemini", "deepseek"}
    assert "Explicacion semantica" in payload["subtitles"][0]["reference_ass"]


def test_panels_expose_independent_persistent_final_pass_options():
    import tkinter as tk
    from takoworks.modules.transcriber.panel import TranscriberPanel
    from takoworks.modules.translator.panel import TranslatorPanel

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")
    root.withdraw()
    try:
        runner = NS(cancel=lambda: None)
        cfg = {"last": {"transcriber_final_pass": True, "translator_final_pass": False}}
        panels = [TranscriberPanel(root, runner, cfg), TranslatorPanel(root, runner, cfg)]
        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)
        for panel in panels:
            checkboxes = [w for w in descendants(panel) if w.winfo_class() == "TCheckbutton" and w.cget("text") == "Final Pass"]
            assert len(checkboxes) == 1
        assert panels[0].v_final_pass.get() is True
        assert panels[1].final_pass_var.get() is False
        root.update_idletasks()
    finally:
        root.destroy()
