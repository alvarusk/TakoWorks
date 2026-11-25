import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from takoworks.shared import final_pass as fp


@pytest.fixture(autouse=True)
def fixed_final_pass_prices(monkeypatch):
    monkeypatch.setattr(fp, "load_price_table", lambda: {"gpt": {"input": 0.002, "output": 0.01}})


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
    assert any("lines" in p for p in fp.validate_subtitle(r"Uno\NDos\NTres", 10000))
    assert any("function word" in p for p in fp.validate_subtitle(r"Vamos a\Ncomer", 2000))
    assert any("duration" in p for p in fp.validate_subtitle("Hola", 0))
    assert any("braces" in p for p in fp.validate_subtitle(r"Hola{\i1", 1000))


def test_final_pass_repairs_limits_preserves_metadata_and_opens_notes(tmp_path, monkeypatch):
    source = write_ass(tmp_path / "show.en-us.ass", r"{\i1}Hello{\i0}")
    original = source.read_bytes()
    output = tmp_path / "show_final.ass"
    client, calls = fake_client([answer("x" * 90), answer(r"{\i1}Hola.{\i0}", ["Revisar el tratamiento."])])
    opened = []
    clock = iter([100.0, 112.5])
    monkeypatch.setattr(fp, "time", NS(perf_counter=lambda: next(clock)))
    monkeypatch.setattr(fp.subprocess, "Popen", lambda args: opened.append(args))
    usage = fp.run_final_pass(source, {"deepl": [r"{\i1}Hola{\i0}"]}, output,
                              source_lang="EN", client=client)
    assert len(calls) == 2
    assert calls[0]["model"] == fp.OPENAI_MODEL
    assert calls[0]["response_format"]["json_schema"]["strict"] is True
    assert usage["prompt_tokens"] == 20
    assert usage["completion_tokens"] == 10
    assert usage["cost_usd"] == pytest.approx(0.00014)
    assert usage["elapsed_seconds"] == 12.5
    assert usage["requests"] == 2
    assert source.read_bytes() == original
    result = output.read_text(encoding="utf-8-sig")
    assert "Title: Keep this title" in result
    assert "Keep this comment" in result
    assert r"Default,Alice,0,0,0,,{\i1}Hola.{\i0}" in result
    report = output.with_suffix(".txt").read_text(encoding="utf-8-sig")
    assert "Revisar el tratamiento" in report
    assert "Technical limits verified" in report
    assert "Final Pass time: 12.50 s" in report
    assert "Estimated Final Pass cost: $0.000140 USD" in report
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
    assert "REVIEW REQUIRED" in report
    assert "No se puede abreviar" in report


@pytest.mark.parametrize("result", [RuntimeError("API unavailable"),
                                   {"subtitles": [], "notes": []}, answer(r"Hola{\p1}")])
def test_api_or_structural_failure_uses_marked_gpt_fallback(tmp_path, result):
    source = write_ass(tmp_path / "source.ass")
    output = tmp_path / "source_final.ass"
    output.write_text("previous final", encoding="utf-8")
    client, calls = fake_client([result] * 3)
    fp.run_final_pass(source, {"gpt": ["Hola"]}, output, source_lang="zh", client=client, open_notes=False)
    assert len(calls) == 3
    assert "Hola{FINAL PASS FALLBACK GPT}" in output.read_text(encoding="utf-8-sig")
    report = output.with_suffix(".txt").read_text(encoding="utf-8-sig")
    assert "FALLBACK GPT" in report
    assert "REVIEW REQUIRED" in report
    assert "NOT COMPLETED" not in report
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_review_still_reports_consumed_tokens_and_cost(tmp_path, monkeypatch):
    source = write_ass(tmp_path / "source.ass")
    output = tmp_path / "source_final.ass"
    client, _ = fake_client([{"subtitles": [], "notes": []}] * 3)
    clock = iter([50.0, 65.0])
    monkeypatch.setattr(fp, "time", NS(perf_counter=lambda: next(clock)))
    fp.run_final_pass(source, {"gpt": ["Hola"]}, output, source_lang="ja", client=client, open_notes=False)
    report = output.with_suffix(".txt").read_text(encoding="utf-8-sig")
    assert "Final Pass time: 15.00 s" in report
    assert "input=30, output=15" in report
    assert "$0.000210 USD" in report


def test_client_setup_failure_also_uses_gpt_fallback(tmp_path, monkeypatch):
    source = write_ass(tmp_path / "source.ass", r"{\i1}Hello{\i0}")
    original = source.read_bytes()
    output = tmp_path / "final.ass"
    def unavailable():
        raise RuntimeError("client unavailable")
    monkeypatch.setattr(fp, "create_openai_client", unavailable)
    usage = fp.run_final_pass(source, {"gpt": [r"{\i1}Hola{\i0}"]}, output,
                              source_lang="ja", open_notes=False)
    assert r"Default,Alice,0,0,0,,{\i1}Hola{\i0}{FINAL PASS FALLBACK GPT}" in output.read_text(encoding="utf-8-sig")
    assert source.read_bytes() == original
    assert usage["requests"] == 0
    assert "client unavailable" in output.with_suffix(".txt").read_text(encoding="utf-8-sig")


def test_cancellation_after_response_preserves_its_consumption(tmp_path, monkeypatch):
    source = write_ass(tmp_path / "source.ass")
    output = tmp_path / "source_final.ass"
    output.write_text("previous final", encoding="utf-8")
    client, _ = fake_client([answer("Hola")])
    create = client.chat.completions.create
    cancelled = threading.Event()
    def cancel_after_response(**kwargs):
        response = create(**kwargs)
        cancelled.set()
        return response
    client.chat.completions.create = cancel_after_response
    with pytest.raises(RuntimeError, match="Cancelled"):
        fp.run_final_pass(source, {"gpt": ["Hola"]}, output, source_lang="EN", client=client,
                          cancel_event=cancelled, open_notes=False)
    report = output.with_suffix(".txt").read_text(encoding="utf-8-sig")
    assert "input=10, output=5" in report
    assert "$0.000070 USD" in report
    assert output.read_text(encoding="utf-8") == "previous final"


@pytest.mark.parametrize("no_response", [True, False])
def test_unknown_consumption_is_reported_as_incomplete(tmp_path, no_response):
    source = write_ass(tmp_path / "source.ass")
    output = tmp_path / "source_final.ass"
    client, _ = fake_client([RuntimeError("timeout")] * 3 if no_response else [answer("Hola")])
    if not no_response:
        create = client.chat.completions.create
        def without_usage(**kwargs):
            response = create(**kwargs)
            response.usage = None
            return response
        client.chat.completions.create = without_usage
    fp.run_final_pass(source, {"gpt": ["Hola"]}, output, source_lang="EN", client=client, open_notes=False)
    report = output.with_suffix(".txt").read_text(encoding="utf-8-sig")
    assert "INCOMPLETE USAGE" in report


def test_cancellation_writes_notes_without_calling_api(tmp_path):
    source = write_ass(tmp_path / "source.ass")
    output = tmp_path / "source_final.ass"
    client, calls = fake_client([])
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(RuntimeError, match="Cancelled"):
        fp.run_final_pass(source, {"gpt": ["Hola"]}, output, source_lang="EN", client=client,
                          cancel_event=cancelled, open_notes=False)
    assert calls == []
    assert not output.exists()
    assert "Cancelled" in output.with_suffix(".txt").read_text(encoding="utf-8-sig")


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


def test_failed_block_keeps_previous_review_and_continues_with_next_block(tmp_path):
    source = write_ass(tmp_path / "source.ass")
    content = source.read_text(encoding="utf-8-sig")
    source.write_text(content + (content.splitlines()[-1] + "\n") * 40, encoding="utf-8-sig")
    def block(start, end):
        return {"subtitles": [{"id": i, "text": f"Revisado {i}", "notes": []}
                              for i in range(start, end + 1)], "notes": []}
    client, calls = fake_client([block(1, 20)] + [RuntimeError("timeout")] * 3 + [block(41, 41)])
    output = tmp_path / "final.ass"
    usage = fp.run_final_pass(source, {"gpt": [f"GPT {i}" for i in range(1, 42)]}, output,
                             source_lang="ja", client=client, open_notes=False)
    _, events = fp._dialogues(output)
    texts = [ev["parts"][ev["text_idx"]] for ev in events]
    assert len(texts) == 41
    assert texts[:20] == [f"Revisado {i}" for i in range(1, 21)]
    assert texts[20:40] == [f"GPT {i}{{FINAL PASS FALLBACK GPT}}" for i in range(21, 41)]
    assert texts[40] == "Revisado 41"
    assert usage["requests"] == len(calls) == 5


def test_invalid_subtitle_does_not_discard_valid_neighbors_or_earlier_attempts(tmp_path):
    source = write_ass(tmp_path / "source.ass")
    content = source.read_text(encoding="utf-8-sig")
    source.write_text(content + (content.splitlines()[-1] + "\n") * 2, encoding="utf-8-sig")
    def mixed(texts):
        return {"subtitles": [{"id": i, "text": text, "notes": []}
                              for i, text in enumerate(texts, 1)], "notes": []}
    client, _ = fake_client([mixed(["Primero", "", ""]),
                            mixed(["", "Segundo", r"Invalido{\p1}"]),
                            RuntimeError("timeout")])
    output = tmp_path / "final.ass"
    fp.run_final_pass(source, {"gpt": ["Uno", "Dos", "Tres"], "claude": ["Otro"] * 3},
                      output, source_lang="ja", client=client, open_notes=False)
    _, events = fp._dialogues(output)
    assert [ev["parts"][ev["text_idx"]] for ev in events] == [
        "Primero", "Segundo", "Tres{FINAL PASS FALLBACK GPT}"]


def test_failure_without_gpt_does_not_mislabel_another_translation(tmp_path):
    source = write_ass(tmp_path / "source.ass")
    output = tmp_path / "final.ass"
    output.write_text("previous final", encoding="utf-8")
    client, _ = fake_client([RuntimeError("timeout")] * 3)
    with pytest.raises(RuntimeError, match="no GPT translation available"):
        fp.run_final_pass(source, {"deepl": ["Hola"]}, output, source_lang="EN",
                          client=client, open_notes=False)
    assert output.read_text(encoding="utf-8") == "previous final"


@pytest.mark.parametrize("broken_items", [[], [{"id": 2, "text": "Duplicado", "notes": []}] * 2])
def test_missing_or_duplicate_id_only_falls_back_affected_subtitle(tmp_path, broken_items):
    source = write_ass(tmp_path / "source.ass")
    content = source.read_text(encoding="utf-8-sig")
    source.write_text(content + content.splitlines()[-1] + "\n", encoding="utf-8-sig")
    response = {"subtitles": [{"id": 1, "text": "Revisado", "notes": []}] + broken_items,
                "notes": []}
    client, _ = fake_client([response] * 3)
    output = tmp_path / "final.ass"
    fp.run_final_pass(source, {"gpt": ["Uno", "Dos"]}, output,
                      source_lang="ja", client=client, open_notes=False)
    _, events = fp._dialogues(output)
    assert [ev["parts"][ev["text_idx"]] for ev in events] == [
        "Revisado", "Dos{FINAL PASS FALLBACK GPT}"]


@pytest.mark.parametrize("lang", ["EN", "FR"])
def test_translator_runs_review_after_deepl(tmp_path, monkeypatch, lang):
    from takoworks.modules.translator import core

    source = write_ass(tmp_path / f"show.{lang.lower()}.ass")
    client, calls = fake_client([answer("Hola.")])
    monkeypatch.setattr(fp, "create_openai_client", lambda: client)
    monkeypatch.setattr(fp.subprocess, "Popen", lambda args: None)
    monkeypatch.setattr(core, "DeepLClient", lambda **kwargs: NS(
        base_url="test", translate_batch=lambda texts, **kwargs: ["Buenas."]))
    logs = []
    translated = Path(core.translate_ass_file(str(source), auth_key="test", source_lang=lang, final_pass=True, log=logs.append))
    assert "Buenas." in translated.read_text(encoding="utf-8-sig")
    assert "Hola." in translated.with_name(translated.stem + "_final.ass").read_text(encoding="utf-8-sig")
    payload = json.loads(calls[0]["messages"][1]["content"])
    assert payload["source_lang"] == lang
    assert payload["subtitles"][0]["translations"] == {"deepl": "Buenas."}
    assert any("Final Pass time:" in line and "$0.000070 USD" in line for line in logs)


@pytest.mark.parametrize("lang", ["ja", "zh"])
@pytest.mark.parametrize("models", ["gpt,claude,gemini,deepseek", "claude"])
def test_transcriber_reviews_all_translations_after_last_engine(tmp_path, monkeypatch, capsys, lang, models):
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
        assert order == (["openai", "claude", "gemini", "deepseek"] if "gemini" in models
                         else ["openai", "claude"])
        return client
    monkeypatch.setattr(fp, "create_openai_client", create_client)
    monkeypatch.setattr(fp.subprocess, "Popen", lambda args: None)
    saved = []
    monkeypatch.setattr(core, "persist_costs_to_supabase", lambda *args: saved.append(args[-1]))
    monkeypatch.setattr(core, "_load_price_table", lambda: core.DEFAULT_PRICE_PER_1K)
    core.main([str(source), "--skip-asr", "--lang", lang, "--series", "Test", "--source-type", "None",
               "--final-pass", "--models", models])
    assert (tmp_path / "source_final.ass").exists()
    payload = json.loads(calls[0]["messages"][1]["content"])
    assert set(payload["subtitles"][0]["translations"]) == set(models.split(",")) | {"gpt"}
    assert "Explicacion semantica" in payload["subtitles"][0]["reference_ass"]
    assert saved[0]["gpt"].total_tokens == 0
    assert saved[0]["gpt"].cost_usd == 0
    assert saved[0]["final_pass"].engine == "final_pass"
    assert saved[0]["final_pass"].total_tokens == 15
    assert saved[0]["final_pass"].cost_usd == pytest.approx(0.00007)
    log = capsys.readouterr().out
    assert "Final Pass: prompt=10 completion=5 total=15" in log
    assert "[Costs] TOTAL: tokens=15" in log
    assert any(line.startswith("Final Pass:") and ";" in line for line in log.splitlines())


def test_shared_prices_respect_config_and_environment(monkeypatch):
    from takoworks.shared import api_costs

    monkeypatch.setattr(api_costs.app_config, "load_config", lambda: {
        "cost_per_1k": {"gpt": {"input": 0.02, "output": 0.04}},
    })
    monkeypatch.delenv("COST_GPT_OUTPUT_PER_1K", raising=False)
    monkeypatch.setenv("COST_GPT_INPUT_PER_1K", "0.03")
    prices = api_costs.load_price_table()["gpt"]
    assert prices == {"input": 0.03, "output": 0.04}
    assert api_costs.calculate_cost(prices, 2000, 1000) == pytest.approx(0.1)


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
