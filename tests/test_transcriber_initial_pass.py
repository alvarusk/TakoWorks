import copy
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from takoworks.modules.transcriber import core, initial_pass as initial


def episode():
    subs = core.pysubs2.SSAFile()
    for name in ("Gen_Main", "Alt_Gen_Second", "Sign"):
        subs.styles[name] = core.pysubs2.SSAStyle()
    subs.events = [
        core.pysubs2.SSAEvent(start=100, end=1100, text="raw", style="Gen_Main", name="A", layer=2),
        core.pysubs2.SSAEvent(start=100, end=1100, text="comment", style="Gen_Main", type="Comment"),
        core.pysubs2.SSAEvent(start=1200, end=2200, text="sign", style="Sign"),
        core.pysubs2.SSAEvent(start=2300, end=3300, text="raw2", style="Alt_Gen_Second", name="B"),
    ]
    return subs


def corrected(rows, suffix):
    return json.dumps({"lines": [dict(id=row["id"], text=row["current"] + suffix,
                                     uncertain=False, reason="PDF evidence", page=2) for row in rows]})


def fake_clients(monkeypatch):
    uploaded, deleted, closed = [], [], []
    gpt = NS(files=NS(
        create=lambda **kw: uploaded.append(("gpt", kw["purpose"])) or NS(id="gpt-file"),
        delete=lambda id: deleted.append(("gpt", id))), close=lambda: closed.append("gpt"))
    gemini = NS(files=NS(
        upload=lambda **kw: uploaded.append(("gemini", kw["config"])) or NS(name="gemini-file", uri="uri", state="ACTIVE"),
        delete=lambda **kw: deleted.append(("gemini", kw["name"]))), close=lambda: closed.append("gemini"))
    monkeypatch.setattr(core, "get_openai_client", lambda: gpt)
    monkeypatch.setattr(core, "get_gemini_model", lambda *args: ("google-genai", gemini, "unused"))
    return uploaded, deleted, closed


def pdf_file(tmp_path):
    path = tmp_path / "episode.pdf"
    path.write_bytes(b"%PDF-1.4\nmock document for mocked providers")
    return path


@pytest.mark.parametrize("lang", ["ja", "zh"])
@pytest.mark.parametrize("exit_code", [0, 3221226505])
def test_purfview_batches_only_target_events_and_preserves_metadata(monkeypatch, tmp_path, lang, exit_code, capsys):
    subs = episode()
    source = tmp_path / "input.ass"
    subs.save(str(source))
    original_bytes = source.read_bytes()
    executable, video = tmp_path / "faster-whisper-xxl.exe", tmp_path / "episode.mkv"
    executable.touch()
    video.touch()
    commands = []

    def run(command, cancel_event=None, cwd=None, **kwargs):
        commands.append(command)
        if command[0] == "ffmpeg":
            Path(command[-1]).touch()
        else:
            clips = list(Path(command[1]).glob("*.wav"))
            assert [p.stem for p in sorted(clips)] == ["line_000000", "line_000003"]
            output = Path(command[command.index("--output_dir") + 1])
            for clip in clips:
                (output / f"{clip.stem}.txt").write_text(
                    "[00:00.000 --> 00:01.100] spoken\n[00:01.100 --> 00:02.200] words",
                    encoding="utf-8-sig")
            if exit_code:
                raise initial.ProcessError(exit_code, "Operation finished in: 0:05:46")

    monkeypatch.setattr(initial, "run_process", run)
    result = initial.transcribe_purfview(source, video, lang, str(executable), pad_ms=200)
    assert len(commands) == 3
    assert commands[0][commands[0].index("-ss") + 1] == "0.000"
    assert commands[0][commands[0].index("-t") + 1] == "1.300"
    assert commands[-1][commands[-1].index("--model") + 1] == "large-v3-turbo"
    assert commands[-1][commands[-1].index("--language") + 1] == lang
    for index, (before, after) in enumerate(zip(subs, result)):
        expected = before.as_dict()
        if index in (0, 3):
            expected["text"] = "spoken words"
        assert after.as_dict() == expected
    assert source.read_bytes() == original_bytes
    log = capsys.readouterr().out
    assert ("Warning" in log) == bool(exit_code)
    if exit_code:
        assert str(exit_code) in log
    assert "Output ready 2/2" in log
    assert "Transcription complete: 2/2 lines validated" in log
    assert "excluding audio preparation" in log


@pytest.mark.parametrize("output_kind", ["missing", "invalid", "unreadable", "cancelled"])
def test_failed_purfview_requires_complete_valid_outputs(monkeypatch, tmp_path, output_kind):
    source = tmp_path / "input.ass"
    episode().save(str(source))
    original_bytes = source.read_bytes()
    executable, video = tmp_path / "xxl.exe", tmp_path / "video.mkv"
    executable.touch()
    video.touch()
    cancel = threading.Event()
    failure = initial.ProcessError(3221226505, "native failure")

    def run(command, cancel_event=None, cwd=None, **kwargs):
        if command[0] == "ffmpeg":
            return
        output = Path(command[command.index("--output_dir") + 1])
        (output / "line_000000.txt").write_text("spoken", encoding="utf-8")
        if output_kind != "missing":
            (output / "line_000003.txt").write_bytes(
                b"\xff" if output_kind == "unreadable" else
                b"{unsafe}" if output_kind == "invalid" else b"spoken")
        if output_kind == "cancelled":
            cancel.set()
        raise failure

    monkeypatch.setattr(initial, "run_process", run)
    expected = {"missing": initial.ProcessError, "invalid": ValueError,
                "unreadable": UnicodeDecodeError, "cancelled": InterruptedError}[output_kind]
    with pytest.raises(expected) as exc:
        initial.transcribe_purfview(source, video, "ja", str(executable), cancel_event=cancel)
    if output_kind == "missing":
        assert exc.value is failure
    assert source.read_bytes() == original_bytes


def test_run_process_preserves_exit_code_and_diagnostics():
    with pytest.raises(initial.ProcessError, match="diagnostic") as exc:
        initial.run_process([sys.executable, "-c", "import sys; print('diagnostic'); sys.exit(3)"])
    assert exc.value.returncode == 3


def test_process_output_is_available_before_exit_and_can_be_cancelled():
    cancel = threading.Event()
    lines = []

    def output(line):
        lines.append(line)
        cancel.set()

    with pytest.raises(InterruptedError):
        initial.run_process(
            [sys.executable, "-c", "import time; print('live progress', flush=True); time.sleep(30)"],
            cancel_event=cancel, on_output=output)
    assert lines == ["live progress"]


def test_process_output_decodes_split_utf8_and_final_line():
    lines = []
    initial.run_process([sys.executable, "-c",
        "import sys,time; data=bytes([227,129,130]); "
        "sys.stdout.buffer.write(data[:1]); sys.stdout.buffer.flush(); time.sleep(.3); "
        "sys.stdout.buffer.write(data[1:]+b'\\r\\nlast'); sys.stdout.buffer.flush()"], on_output=lines.append)
    assert lines == ["\u3042", "last"]


def test_purfview_reports_current_audio_completion_and_elapsed_time(monkeypatch, tmp_path, capsys):
    now = [0.0]
    monkeypatch.setattr(initial.time, "monotonic", lambda: now[0])
    progress = initial._PurfviewProgress(initial.eligible_events(episode()), tmp_path)
    progress.output(r"Starting to process: C:\audio\line_000000.wav")
    now[0] = 2.0
    progress.output("Subtitles are written to text directory.")
    (tmp_path / "line_000000.txt").touch()
    progress.poll(final=True)
    progress.output(r"Starting to process: C:\audio\line_000003.wav")
    now[0] = 8.0
    progress.poll()
    log = capsys.readouterr().out
    assert "Transcribing audio 1/2 (ASS line 1)" in log
    assert "Transcribing audio 2/2 (ASS line 4)" in log
    assert log.count("Output ready 1/2") == 1
    assert "audio time 2.0 s" in log
    assert "current ASS line 4; elapsed 8.0 s" in log


@pytest.mark.parametrize("raw,expected", [
    ("[00:00.000 --> 00:01.400] spoken", "spoken"),
    ("[00:00.000 --> 00:00.700] one [00:01.100 --> 00:01.300] two", "one two"),
    ("[00:00:00,000 --> 00:00:01,400] spoken\nnext", "spoken next"),
    ("[applause] 12:00 [123]", "[applause] 12:00 [123]"),
    ("[00:00.000 --> 00:01.000]", ""),
])
def test_asr_timestamp_cleanup_is_specific(raw, expected):
    assert initial.clean_asr_text(raw) == expected


def test_missing_asr_output_stops_instead_of_misaligning_lines(monkeypatch, tmp_path):
    source = tmp_path / "input.ass"
    episode().save(str(source))
    executable, video = tmp_path / "xxl.exe", tmp_path / "video.mkv"
    executable.touch()
    video.touch()
    monkeypatch.setattr(initial, "run_process", lambda *a, **kw: None)
    with pytest.raises(RuntimeError, match="ASS line 1"):
        initial.transcribe_purfview(source, video, "ja", str(executable))


def test_cancelled_asr_does_not_start_process(monkeypatch, tmp_path):
    source = tmp_path / "input.ass"
    episode().save(str(source))
    executable, video = tmp_path / "xxl.exe", tmp_path / "video.mkv"
    executable.touch()
    video.touch()
    cancel = threading.Event()
    cancel.set()
    monkeypatch.setattr(initial, "run_process", lambda *a, **kw: pytest.fail("started after cancellation"))
    with pytest.raises(InterruptedError):
        initial.transcribe_purfview(source, video, "ja", str(executable), cancel_event=cancel)


@pytest.mark.parametrize("change", [
    {"id": 4}, {"text": ""}, {"text": "{hidden}"}, {"text": "a\\Nb"},
    {"text": "a\nb"}, {"uncertain": True}, {"page": 0}, {"page": True}, {"reason": ""},
    {"text": "[00:00.000 --> 00:01.000] speech"},
])
def test_review_rejects_unsafe_or_unsupported_changes(change):
    line = dict(id=1, text="corrected", uncertain=False, reason="PDF", page=1)
    line.update(change)
    with pytest.raises(ValueError):
        initial.parse_review(json.dumps({"lines": [line]}), [{"id": 1, "current": "source"}])


def test_uncertain_review_can_preserve_an_empty_line():
    raw = json.dumps({"lines": [dict(id=7, text="", uncertain=True, reason="No match", page=0)]})
    assert initial.parse_review(raw, [{"id": 7, "current": ""}])[0]["uncertain"]


def test_two_passes_use_previous_corrections_and_keep_non_targets(monkeypatch, tmp_path):
    uploaded, deleted, closed = fake_clients(monkeypatch)
    subs = episode()
    original = copy.deepcopy(subs)
    calls, usage, timings = [], {}, {}

    def request(core, engine, client, document, system, prompt, count, accumulator):
        payload = json.loads(prompt)
        calls.append((engine, payload))
        accumulator.prompt_tokens += 100
        accumulator.cost_usd += .1
        return corrected(payload["lines"], "_" + engine)

    monkeypatch.setattr(initial, "_review_request", request)
    initial.verify_with_pdf(subs, pdf_file(tmp_path), "ja", tmp_path / "stages",
                            usage_by_model=usage, phase_timings=timings)
    assert [call[0] for call in calls] == ["gpt", "gemini"]
    assert calls[1][1]["lines"][0]["current"] == "raw_gpt"
    assert calls[1][1]["lines"][0]["asr"] == "raw"
    assert [row["id"] for row in calls[1][1]["lines"]] == [1, 4]
    assert subs[0].text == "raw_gpt_gemini"
    assert subs[1:3] == original[1:3]
    assert subs[0].start == original[0].start and subs[0].name == "A"
    assert len(uploaded) == len(deleted) == len(closed) == 2
    assert set(timings) == {"verification_gpt", "verification_gemini"}
    assert sum(value.cost_usd for value in usage.values()) == pytest.approx(.2)
    saved = json.loads((tmp_path / "stages/verification.json").read_text(encoding="utf-8"))
    assert len(saved["attempts"]) == 2
    assert saved["usage"]["verification_gemini"]["prompt_tokens"] == 100
    assert (tmp_path / "stages/verification_gpt.ass").is_file()
    assert (tmp_path / "stages/verification_gemini.ass").is_file()


def test_bad_review_retries_then_stops_and_keeps_billed_usage(monkeypatch, tmp_path):
    uploaded, deleted, closed = fake_clients(monkeypatch)
    subs = episode()
    usage = {}

    def request(core, engine, client, document, system, prompt, count, accumulator):
        accumulator.prompt_tokens += 15
        return '{"lines": []}'

    monkeypatch.setattr(initial, "_review_request", request)
    with pytest.raises(RuntimeError, match="Verification gpt failed"):
        initial.verify_with_pdf(subs, pdf_file(tmp_path), "ja", tmp_path, usage_by_model=usage)
    assert subs[0].text == "raw"
    assert usage["verification_gpt"].prompt_tokens == 45
    assert len(uploaded) == len(deleted) == len(closed) == 1
    saved = json.loads((tmp_path / "verification.json").read_text(encoding="utf-8"))
    assert len(saved["attempts"]) == 3
    assert all("error" in attempt for attempt in saved["attempts"])


def test_cancellation_after_response_keeps_usage_and_releases_upload(monkeypatch, tmp_path):
    uploaded, deleted, closed = fake_clients(monkeypatch)
    subs = episode()
    usage, cancel = {}, threading.Event()

    def request(core, engine, client, document, system, prompt, count, accumulator):
        accumulator.prompt_tokens += 30
        cancel.set()
        return corrected(json.loads(prompt)["lines"], "_changed")

    monkeypatch.setattr(initial, "_review_request", request)
    with pytest.raises(InterruptedError):
        initial.verify_with_pdf(subs, pdf_file(tmp_path), "ja", tmp_path,
                                cancel_event=cancel, usage_by_model=usage)
    assert subs[0].text == "raw"
    assert usage["verification_gpt"].prompt_tokens == 30
    assert len(uploaded) == len(deleted) == len(closed) == 1
    assert json.loads((tmp_path / "verification.json").read_text(encoding="utf-8"))["usage"]


def test_multiple_blocks_keep_exact_ids_and_pass_context(monkeypatch, tmp_path):
    fake_clients(monkeypatch)
    subs = episode()
    for index in range(39):
        subs.append(core.pysubs2.SSAEvent(start=4000 + 1000 * index, end=5000 + 1000 * index,
                                        text=f"source {index}", style="Gen_Main"))
    calls = []

    def request(core, engine, client, document, system, prompt, count, accumulator):
        payload = json.loads(prompt)
        calls.append((engine, payload))
        assert not set(row["id"] for row in payload["lines"]) & set(row["id"] for row in payload["context_only"])
        return corrected(payload["lines"], "_" + engine)

    monkeypatch.setattr(initial, "_review_request", request)
    initial.verify_with_pdf(subs, pdf_file(tmp_path), "ja", tmp_path)
    assert [(engine, len(payload["lines"])) for engine, payload in calls] == [
        ("gpt", 40), ("gpt", 1), ("gemini", 10), ("gemini", 10),
        ("gemini", 10), ("gemini", 10), ("gemini", 1)]
    assert calls[1][1]["lines"][0]["id"] == len(subs)
    assert all(event.text.endswith("_gpt_gemini") for _, event in initial.eligible_events(subs))


def test_gemini_truncation_splits_blocks_without_repeating_accepted_lines(monkeypatch, tmp_path):
    fake_clients(monkeypatch)
    subs = episode()
    for index in range(8):
        subs.append(core.pysubs2.SSAEvent(start=4000 + index * 1000, end=5000 + index * 1000,
                                        text=f"line {index}", style="Gen_Main"))
    calls = []

    def request(core, engine, client, document, system, prompt, count, usage):
        payload = json.loads(prompt)
        rows = payload["lines"]
        calls.append((engine, len(rows)))
        assert not {r["id"] for r in rows} & {r["id"] for r in payload["context_only"]}
        usage.prompt_tokens += 1
        if engine == "gemini" and len(rows) > 3:
            raise initial.ReviewResponseError({"finish_reason": "MAX_TOKENS", "block_reason": "NONE"})
        return corrected(rows, "_" + engine)

    monkeypatch.setattr(initial, "_review_request", request)
    initial.verify_with_pdf(subs, pdf_file(tmp_path), "ja", tmp_path)
    assert calls == [("gpt", 10), ("gemini", 10), ("gemini", 5), ("gemini", 2),
                     ("gemini", 3), ("gemini", 5), ("gemini", 2), ("gemini", 3)]
    report = json.loads((tmp_path / "verification.json").read_text(encoding="utf-8"))
    assert sum(a.get("action") == "split" for a in report["attempts"]) == 3
    accepted = [r["id"] for a in report["attempts"] if a["engine"] == "gemini" and "lines" in a for r in a["lines"]]
    assert accepted == [i + 1 for i, _ in initial.eligible_events(subs)]
    assert report["usage"]["verification_gemini"]["prompt_tokens"] == 7
    assert all(e.text.endswith("_gpt_gemini") for _, e in initial.eligible_events(subs))
    assert (tmp_path / "verification_gemini.ass").is_file()


def test_gemini_content_block_stops_without_repeated_or_split_requests(monkeypatch, tmp_path):
    fake_clients(monkeypatch)
    calls = []

    def request(core, engine, client, document, system, prompt, count, usage):
        calls.append(engine)
        if engine == "gemini":
            raise initial.ReviewResponseError({"finish_reason": "SAFETY", "block_reason": "NONE"})
        return corrected(json.loads(prompt)["lines"], "_gpt")

    monkeypatch.setattr(initial, "_review_request", request)
    with pytest.raises(RuntimeError, match="Verification gemini failed"):
        initial.verify_with_pdf(episode(), pdf_file(tmp_path), "ja", tmp_path)
    assert calls == ["gpt", "gemini"]
    assert (tmp_path / "verification_gpt.ass").is_file()
    assert not (tmp_path / "verification_gemini.ass").exists()


def test_verification_removes_old_asr_timestamps_before_both_requests(monkeypatch, tmp_path):
    fake_clients(monkeypatch)
    subs = episode()
    subs[0].text = "[00:00.000 --> 00:00.700] one [00:00.700 --> 00:01.000] two"
    original = copy.deepcopy(subs)

    def request(core, engine, client, document, system, prompt, count, usage):
        rows = json.loads(prompt)["lines"]
        assert rows[0]["asr"] == rows[0]["current"] == "one two"
        return json.dumps({"lines": [dict(id=r["id"], text=r["current"], uncertain=True,
                                         reason="No change", page=0) for r in rows]})

    monkeypatch.setattr(initial, "_review_request", request)
    initial.verify_with_pdf(subs, pdf_file(tmp_path), "ja", tmp_path)
    for name in ("verification_gpt.ass", "verification_gemini.ass"):
        result = core.pysubs2.load(str(tmp_path / name))
        assert result[0].text == "one two"
        assert result[0].start == original[0].start and result[0].end == original[0].end
        assert result[1:3] == original[1:3]


def test_provider_requests_reuse_translation_models_and_count_thoughts(monkeypatch):
    calls = []
    monkeypatch.setattr(core, "estimate_cost", lambda engine, pt, ct: (pt + ct) / 100)
    response = NS(usage=NS(prompt_tokens=100, completion_tokens=25),
                  choices=[NS(finish_reason="stop", message=NS(content="{}", refusal=None))])
    client = NS(chat=NS(completions=NS(create=lambda **kw: calls.append(kw) or response)))
    usage = core.ApiUsage(engine="verification_gpt", model_name=core.OPENAI_MODEL)
    initial._review_request(core, "gpt", client, NS(id="file"), "system", "prompt", 2, usage)
    assert calls[0]["model"] == core.OPENAI_MODEL
    assert calls[0]["messages"][1]["content"][0]["file"]["file_id"] == "file"
    assert "temperature" not in calls[0]
    assert usage.total_tokens == 125 and usage.cost_usd == 1.25

    response = NS(usage_metadata=NS(prompt_token_count=100, candidates_token_count=20, thoughts_token_count=7),
                  candidates=[NS(finish_reason="STOP")], text="{}")
    client = NS(models=NS(generate_content=lambda **kw: calls.append(kw) or response))
    usage = core.ApiUsage(engine="verification_gemini", model_name=core.GEMINI_MODEL)
    initial._review_request(core, "gemini", client, NS(uri="https://example.com/pdf"), "system", "prompt", 2, usage)
    assert calls[1]["model"] == core.GEMINI_MODEL
    assert calls[1]["config"].response_mime_type == "application/json"
    assert calls[1]["config"].thinking_config.thinking_level.lower() == "low"
    assert usage.completion_tokens == 27 and usage.cost_usd == 1.27


@pytest.mark.parametrize("finish_reason,block_reason", [
    ("MAX_TOKENS", None), ("SAFETY", None), (None, "PROHIBITED_CONTENT"),
])
def test_gemini_rejection_reports_actual_reason_and_partial_text(monkeypatch, finish_reason, block_reason):
    monkeypatch.setattr(core, "estimate_cost", lambda *args: .3)
    response = NS(
        usage_metadata=NS(prompt_token_count=50, candidates_token_count=20, thoughts_token_count=5),
        prompt_feedback=NS(block_reason=block_reason, block_reason_message="Prompt rejected" if block_reason else None),
        candidates=[NS(finish_reason=finish_reason, finish_message="Generation stopped",
                       content=NS(parts=[NS(text="private thought", thought=True), NS(text='{"lines":')]))]
                   if finish_reason else [],
    )
    client = NS(models=NS(generate_content=lambda **kw: response))
    usage = core.ApiUsage(engine="verification_gemini", model_name="model")
    with pytest.raises(initial.ReviewResponseError) as exc:
        initial._review_request(core, "gemini", client, NS(uri="https://example.com/pdf"), "sys", "p", 1, usage)
    error = exc.value
    assert error.diagnostics["finish_reason"] == (finish_reason or "UNKNOWN")
    assert error.diagnostics["block_reason"] == (block_reason or "NONE")
    assert error.diagnostics["tokens"]["thoughts_token_count"] == 5
    assert (finish_reason or block_reason) in str(error)
    assert error.raw == ('{"lines":' if finish_reason else "")
    assert usage.completion_tokens == 25 and usage.cost_usd == .3


def test_gemini_diagnostics_survive_retries_in_report(monkeypatch, tmp_path, capsys):
    fake_clients(monkeypatch)
    diagnostics = {"finish_reason": "MAX_TOKENS", "block_reason": "NONE"}

    def request(core, engine, client, document, system, prompt, count, usage):
        if engine == "gpt":
            return corrected(json.loads(prompt)["lines"], "_gpt")
        raise initial.ReviewResponseError(diagnostics, '{"lines":')

    monkeypatch.setattr(initial, "_review_request", request)
    with pytest.raises(RuntimeError, match="Verification gemini failed"):
        initial.verify_with_pdf(episode(), pdf_file(tmp_path), "ja", tmp_path)
    report = json.loads((tmp_path / "verification.json").read_text(encoding="utf-8"))
    failures = [attempt for attempt in report["attempts"] if attempt["engine"] == "gemini"]
    assert len(failures) == 4
    assert failures[0]["action"] == "split"
    assert all(attempt["diagnostics"] == diagnostics and attempt["raw"] == '{"lines":' for attempt in failures)
    assert (tmp_path / "verification_gpt.ass").is_file()
    assert not (tmp_path / "verification_gemini.ass").exists()
    log = capsys.readouterr().out
    assert "[Verification gemini] Invalid block; retry 2/3" in log
    assert "MAX_TOKENS" in log


@pytest.mark.parametrize("engine", ["gpt", "gemini"])
def test_truncated_requests_are_billed_before_rejection(monkeypatch, engine):
    monkeypatch.setattr(core, "estimate_cost", lambda *args: .3)
    response = NS(usage=NS(prompt_tokens=50, completion_tokens=25),
                  choices=[NS(finish_reason="length", message=NS(content="partial"))],
                  usage_metadata=NS(prompt_token_count=50, candidates_token_count=20, thoughts_token_count=5),
                  candidates=[NS(finish_reason="MAX_TOKENS")])
    client = NS(chat=NS(completions=NS(create=lambda **kw: response)),
                models=NS(generate_content=lambda **kw: response))
    usage = core.ApiUsage(engine=engine, model_name="model")
    with pytest.raises(ValueError):
        initial._review_request(core, engine, client, NS(id="file", uri="https://example.com/pdf"), "sys", "p", 1, usage)
    assert usage.total_tokens == 75 and usage.cost_usd == .3


def test_cli_chains_initial_workflow_before_annotations_and_counts_costs_once(monkeypatch, tmp_path):
    source = tmp_path / "input.ass"
    episode().save(str(source))
    original_bytes = source.read_bytes()
    pdf = pdf_file(tmp_path)
    calls, persisted = [], []
    monkeypatch.setattr(core, "OPENAI_API_KEY", "test")
    monkeypatch.setattr(core, "GEMINI_API_KEY", "test")

    def asr(*args, **kw):
        calls.append("asr")
        return episode()

    def verify(subs, *args, **kw):
        calls.append("verify")
        subs[0].text = "corrected twice"
        kw["usage_by_model"]["verification_gpt"] = core.ApiUsage("verification_gpt", core.OPENAI_MODEL, 10, 5, .2)
        return subs

    def annotate(subs, *args, **kw):
        assert subs[0].text == "corrected twice"
        calls.append("annotations")
        return subs

    def translate(subs, *args, **kw):
        calls.append("translations")
        return {}, {}, {}

    monkeypatch.setattr(initial, "transcribe_purfview", asr)
    monkeypatch.setattr(initial, "verify_with_pdf", verify)
    monkeypatch.setattr(core, "add_roman_morph_to_subs", annotate)
    monkeypatch.setattr(core, "process_all_models_with_subs", translate)
    monkeypatch.setattr(core, "generate_html", lambda *a, **kw: calls.append("html"))
    monkeypatch.setattr(core, "persist_costs_to_supabase", lambda *a: persisted.append(dict(a[-1])))
    from takoworks.shared import final_pass

    def final_review(source_path, translations, output_path, **kwargs):
        calls.append("final")
        assert Path(source_path).name == "annotated.ass"
        assert core.pysubs2.load(source_path)[0].text == "corrected twice"
        return dict(prompt_tokens=5, completion_tokens=2, cost_usd=.1, elapsed_seconds=.01)

    monkeypatch.setattr(final_pass, "run_final_pass", final_review)
    core.main([str(source), "video.mkv", "--asr-engine", "purfview", "--reference-pdf", str(pdf),
               "--lang", "ja", "--series", "Series", "--source-type", "None", "--models", "",
               "--do-roman-morph", "--html", "--final-pass"])
    assert calls == ["asr", "verify", "annotations", "translations", "final", "html"]
    assert len(persisted) == 2
    assert persisted[0]["verification_gpt"].cost_usd == .2
    assert set(persisted[1]) == {"final_pass"}
    assert source.read_bytes() == original_bytes


def test_cli_persists_verification_costs_when_verification_fails(monkeypatch, tmp_path):
    source = tmp_path / "input.ass"
    episode().save(str(source))
    monkeypatch.setattr(core, "OPENAI_API_KEY", "test")
    monkeypatch.setattr(core, "GEMINI_API_KEY", "test")
    persisted = []

    def fail(subs, *args, **kw):
        kw["usage_by_model"]["verification_gpt"] = core.ApiUsage("verification_gpt", core.OPENAI_MODEL, 15, 5, .2)
        raise RuntimeError("failed")

    monkeypatch.setattr(initial, "verify_with_pdf", fail)
    monkeypatch.setattr(core, "persist_costs_to_supabase", lambda *a: persisted.append(dict(a[-1])))
    monkeypatch.setattr(core, "process_all_models_with_subs", lambda *a, **kw: pytest.fail("translated after failure"))
    with pytest.raises(RuntimeError, match="failed"):
        core.main([str(source), "--skip-asr", "--reference-pdf", str(pdf_file(tmp_path)),
                   "--lang", "ja", "--series", "Series", "--source-type", "None"])
    assert len(persisted) == 1 and persisted[0]["verification_gpt"].total_tokens == 20


def test_panel_initial_controls_are_independent_and_scrollable():
    tk = pytest.importorskip("tkinter")
    from takoworks.config import DEFAULT_CONFIG
    from takoworks.modules.transcriber.panel import TranscriberPanel

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display is not available")
    root.withdraw()
    try:
        panel = TranscriberPanel(root, NS(cancel=lambda: None), copy.deepcopy(DEFAULT_CONFIG))
        panel.pack(fill="both", expand=True)
        root.update_idletasks()
        assert str(panel.video_entry["state"]) == "disabled"
        assert str(panel.pdf_entry["state"]) == "disabled"
        panel.v_verify_pdf.set(True)
        panel._toggle_video()
        assert str(panel.pdf_entry["state"]) == "normal"
        assert str(panel.video_entry["state"]) == "disabled"
        panel.v_skip_asr.set(False)
        panel._toggle_video()
        assert str(panel.purfview_entry["state"]) == "normal"
        assert any(isinstance(widget, tk.Canvas) for widget in panel.winfo_children())
        panel.asr_engine_var.set("legacy")
        panel._toggle_video()
        assert str(panel.purfview_entry["state"]) == "disabled"
    finally:
        root.destroy()
