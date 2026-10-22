import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from takoworks.modules.transcriber import core


@pytest.fixture(autouse=True)
def no_wait_or_live_prices(monkeypatch):
    monkeypatch.setattr(core.time, "sleep", lambda _: None)
    monkeypatch.setattr(core, "_load_price_table", lambda: core.DEFAULT_PRICE_PER_1K)


def gemini_response(text=None, finish="STOP", feedback=None):
    candidates = [] if text is None else [NS(
        content=NS(parts=[NS(text=text, thought=False)]),
        finish_reason=finish, finish_message="provider details",
        safety_ratings=[], citation_metadata=None,
    )]
    return NS(candidates=candidates, prompt_feedback=feedback, usage_metadata=NS(
        prompt_token_count=10, candidates_token_count=5, thoughts_token_count=3,
    ))


def fake_gemini(monkeypatch, responses):
    calls = []
    results = iter(responses)

    def generate(**kwargs):
        calls.append(kwargs)
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    client = NS(models=NS(generate_content=generate))
    monkeypatch.setattr(core, "get_gemini_model", lambda *args: ("google-genai", client, "system"))
    monkeypatch.setattr(core, "google_genai_types", NS(GenerateContentConfig=NS, ThinkingConfig=NS))
    return calls


def test_gemini_ten_line_blocks_and_thinking_usage(monkeypatch):
    calls = fake_gemini(monkeypatch, [
        gemini_response(json.dumps({"translations": ["translated"] * n}))
        for n in (10, 10, 1)
    ])
    output, usage, error = core.translate_with_gemini(["source"] * 21, "ja", "Series", "None")
    assert output == ["translated"] * 21
    assert error is None
    assert len(calls) == 3
    assert all(call["model"] == "gemini-3.8-flash" for call in calls)
    assert calls[0]["config"].thinking_config.thinking_level == "low"
    assert usage.prompt_tokens == 30
    assert usage.completion_tokens == 24


@pytest.mark.parametrize("kind", ["prompt_block", "recitation", "truncated", "api_error", "mismatch", "refusal"])
def test_gemini_failures_are_logged_and_saved(monkeypatch, tmp_path, capsys, kind):
    responses = {
        "prompt_block": gemini_response(feedback=NS(block_reason="PROHIBITED_CONTENT")),
        "recitation": gemini_response('{"translations":["partial"]}', "RECITATION"),
        "truncated": gemini_response('{"translations":["partial"]}', "MAX_TOKENS"),
        "api_error": RuntimeError("503 unavailable"),
        "mismatch": gemini_response('{"translations":[]}'),
        "refusal": gemini_response("I cannot translate this copyrighted material."),
    }
    calls = fake_gemini(monkeypatch, [responses[kind]] * 3)
    output, usage, error = core.translate_with_gemini(
        ["source"], "ja", "Series", "None", debug_dir=str(tmp_path),
    )
    assert len(calls) == 3
    assert output == ["source"]
    assert error == "partial_error"
    log = capsys.readouterr().out
    assert "[Gemini ERROR]" in log
    assert "attempt=3/3" in log
    saved = [json.loads(path.read_text(encoding="utf-8")) for path in tmp_path.glob("*.json")]
    assert len(saved) >= 3
    assert all(item["source_lines"] == ["source"] for item in saved)
    if kind == "recitation":
        assert "RECITATION" in log
        assert any(item["raw_response"] for item in saved)
    if kind == "prompt_block":
        assert "PROHIBITED_CONTENT" in log
    assert usage.prompt_tokens == (0 if kind == "api_error" else 30)


def test_gemini_does_not_reuse_partial_response_after_api_errors(monkeypatch):
    fake_gemini(monkeypatch, [
        gemini_response('{"translations":["partial"]}', "MAX_TOKENS"),
        RuntimeError("503"), RuntimeError("503"),
    ])
    output, usage, error = core.translate_with_gemini(["source"], "ja", "Series", "None")
    assert output == ["source"]
    assert usage.prompt_tokens == 10
    assert error == "partial_error"


def test_claude_cache_request_and_retry_costs(monkeypatch, capsys):
    calls = []
    results = iter([
        NS(content=[NS(type="thinking"), NS(type="text", text='{"translations":[]}')], usage=NS(
            input_tokens=10, output_tokens=5, cache_creation_input_tokens=1000, cache_read_input_tokens=0)),
        NS(content=[NS(type="thinking"), NS(type="text", text='{"translations":["translated"]}')], usage=NS(
            input_tokens=10, output_tokens=5, cache_creation_input_tokens=0, cache_read_input_tokens=1000)),
    ])

    def create(**kwargs):
        calls.append(kwargs)
        return next(results)

    monkeypatch.setattr(core, "get_claude_client", lambda: NS(messages=NS(create=create)))
    output, usage, error = core.translate_with_claude(["source"], "ja", "Series", "None")
    assert output == ["translated"]
    assert error is None
    assert len(calls) == 2
    assert calls[0]["model"] == "claude-opus-5-5"
    assert calls[0]["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert calls[0]["system"] == calls[1]["system"]
    assert "temperature" not in calls[0]
    assert usage.prompt_tokens == 2020
    assert usage.completion_tokens == 10
    assert usage.cost_usd == pytest.approx((20 + 1000 * 1.25 + 1000 * .05) * .004 / 1000 + 10 * .02 / 1000)
    assert "read=1000" in capsys.readouterr().out


def test_gpt_request_omits_unsupported_temperature(monkeypatch):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return NS(choices=[NS(message=NS(content='{"translations":["translated"]}'))],
                  usage=NS(prompt_tokens=10, completion_tokens=5))

    monkeypatch.setattr(core, "OPENAI_API_KEY", "test")
    monkeypatch.setattr(core, "get_openai_client", lambda: NS(chat=NS(completions=NS(create=create))))
    output, _, error = core.translate_with_openai(["source"], "ja", "Series", "None")
    assert output == ["translated"]
    assert error is None
    assert calls[0]["model"] == "gpt-6.1-sol"
    assert "temperature" not in calls[0]


def test_new_and_legacy_model_names_remain_accepted():
    assert core.normalize_models_arg("GPT-6.1 Sol,Claude Opus 5.5,Gemini 3.8 Flash,DeepSeek V4.1 Flash") == {
        "gpt", "claude", "gemini", "deepseek",
    }
    assert core.normalize_models_arg("gpt-5.6-terra,claude-opus-5,gemini-3.7-flash,deepseek-v4-flash") == {
        "gpt", "claude", "gemini", "deepseek",
    }
    assert core.DEEPSEEK_MODEL == "deepseek-flash"
