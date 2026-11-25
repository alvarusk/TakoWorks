"""Final translation review with measured ASS limits and a human-readable report."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

from .openai_client import OPENAI_MODEL, create_openai_client
from .api_costs import load_price_table, calculate_cost
from ..modules.transferer.transferer import parse_ass

TAG_RE = re.compile(r"\{[^}]*\}")
FUNCTION_WORDS = set(
    "a ante bajo con contra de desde durante en entre hacia hasta mediante para por "
    "segun según sin sobre tras al del y e ni o u pero aunque porque si que el la lo los las un una "
    "unos unas este esta estos estas ese esa esos esas aquel aquella aquellos aquellas "
    "mi mis tu tus su sus nuestro nuestra nuestros nuestras vuestro vuestra".split()
)


def visible_lines(text):
    text = TAG_RE.sub("", text).replace(r"\h", " ")
    return re.split(r"\\[Nn]|\r\n|\r|\n", text)


def formatting_tags(text):
    return [tag for tag in TAG_RE.findall(text) if tag.startswith("{\\")]


def unsafe_ass(text, required_tags):
    return (
        not isinstance(text, str)
        or formatting_tags(text) != list(required_tags)
        or "\n" in text or "\r" in text
        or any(tag not in required_tags for tag in TAG_RE.findall(text))
        or bool(re.search(r"[{}]", TAG_RE.sub("", text)))
        or not any(line.strip() for line in visible_lines(text))
    )


def validate_subtitle(text, duration_ms, required_tags=()):
    lines = visible_lines(text)
    problems = []
    if not text.strip() or not any(line.strip() for line in lines):
        problems.append("Empty text")
    if len(lines) > 2:
        problems.append(f"{len(lines)} lines (maximum 2)")
    for i, line in enumerate(lines, 1):
        if len(line) > 40:
            problems.append(f"Line {i}: {len(line)} CPL (maximum 40)")
        if i < len(lines):
            words = re.findall(r"\w+", line.lower())
            if words and words[-1] in FUNCTION_WORDS:
                problems.append(f"Line break after function word: {words[-1]}")
    count = sum(map(len, lines))
    if duration_ms <= 0:
        problems.append("Zero, negative, or unknown duration; cannot verify CPS")
    elif count * 1000 > 20 * duration_ms:
        problems.append(f"{count * 1000 / duration_ms:.2f} CPS (maximum 20.00)")
    if formatting_tags(text) != list(required_tags):
        problems.append("ASS formatting tags were changed")
    if any(tag not in required_tags for tag in TAG_RE.findall(text)):
        problems.append("Unauthorized ASS tags or comments")
    if re.search(r"[{}]", TAG_RE.sub("", text)):
        problems.append("Unclosed or unauthorized ASS braces")
    if "\n" in text or "\r" in text:
        problems.append("Physical line break not allowed within an ASS event")
    return problems


def _response_format():
    note_list = {"type": "array", "items": {"type": "string"}}
    return {"type": "json_schema", "json_schema": {
        "name": "final_subtitles", "strict": True,
        "schema": {"type": "object", "additionalProperties": False,
                   "properties": {
                       "subtitles": {"type": "array", "items": {
                           "type": "object", "additionalProperties": False,
                           "properties": {"id": {"type": "integer"},
                                          "text": {"type": "string"}, "notes": note_list},
                           "required": ["id", "text", "notes"]}},
                       "notes": note_list}, "required": ["subtitles", "notes"]}}}


def _dialogues(path):
    lines, events = parse_ass(str(path))
    return lines, [ev for ev in events if ev["prefix"].lower() == "dialogue"]


def run_final_pass(reference_path, translations, output_path, *, source_lang,
                   model=OPENAI_MODEL, client=None, instructions="", cancel_event=None,
                   log=print, open_notes=True, source_notes=()):
    """Return tokens, estimated cost and elapsed time; report them even on failure."""
    started = time.perf_counter()
    output_path = Path(output_path)
    notes_path = output_path.with_suffix(".txt")
    notes = [f"Final Pass | model: {model}", f"Reference: {reference_path}"]
    notes.extend(source_notes)
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    prices = {}
    requests = 0
    responses = 0
    missing_usage = 0

    def check_cancel():
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Cancelled")

    try:
        check_cancel()
        prices = load_price_table()["gpt"]
        if output_path.resolve() == Path(reference_path).resolve():
            raise ValueError("The final file cannot overwrite the reference.")
        lines, events = _dialogues(reference_path)
        if not events:
            raise ValueError("No Dialogue events to review.")
        if not translations:
            raise ValueError("No translations available for Final Pass.")
        for engine, texts in translations.items():
            if len(texts) != len(events):
                raise ValueError(f"{engine}: translations do not match the reference events.")
        rules = Path(__file__).with_suffix(".md").read_text(encoding="utf-8")
        if instructions:
            rules += "\n\nTerminologia e instrucciones de la serie:\n" + instructions
        records = []
        if source_lang.lower() in {"ja", "zh"}:
            rules += "\nIdioma de origen: " + source_lang + ". Evalua las explicaciones del ASR; indica en notes si faltan."
        for i, ev in enumerate(events):
            reference = ev["parts"][ev["text_idx"]]
            start, end = ev["start_ms"], ev["end_ms"]
            duration = end - start if start is not None and end is not None else 0
            records.append({"id": i + 1, "start_ms": start, "end_ms": end,
                            "duration_ms": duration,
                            "max_chars": min(80, max(0, math.floor(duration * 20 / 1000))),
                            "reference_ass": reference,
                            "translations": {key: texts[i] for key, texts in translations.items()},
                            "required_tags": formatting_tags(reference)})

        outputs = {}
        unresolved = 0
        for offset in range(0, len(records), 20):
            check_cancel()
            batch = records[offset:offset + 20]
            # Drawing commands have no linguistic text to review.
            targets = [r for r in batch if not re.search(r"\\p[1-9]\d*", r["reference_ass"])]
            for record in batch:
                if record not in targets:
                    outputs[record["id"]] = record["reference_ass"]
                    notes.append(f"Subtitle {record['id']}: ASS drawing preserved; text limits do not apply.")
            if not targets:
                continue
            payload = {"source_lang": source_lang, "subtitles": targets,
                       "context_before": records[max(0, offset - 2):offset],
                       "context_after": records[offset + 20:offset + 22]}
            messages = [{"role": "system", "content": rules},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
            accepted = {}
            accepted_notes = []
            last_error = ""
            for attempt in range(3):
                check_cancel()
                log(f"[Final Pass] {offset + 1}-{offset + len(batch)}, attempt {attempt + 1}/3")
                try:
                    client = client or create_openai_client()
                    requests += 1
                    response = client.chat.completions.create(
                        model=model, messages=messages, response_format=_response_format(),
                        max_completion_tokens=6000,
                    )
                    responses += 1
                    api_usage = getattr(response, "usage", None)
                    if api_usage is None:
                        missing_usage += 1
                    for key in ("prompt_tokens", "completion_tokens"):
                        usage[key] += int(getattr(api_usage, key, 0) or 0)
                    check_cancel()
                    choice = response.choices[0]
                    message = choice.message
                    if getattr(message, "refusal", None):
                        raise ValueError(f"Model refusal: {message.refusal}")
                    if getattr(choice, "finish_reason", "stop") != "stop":
                        raise ValueError(f"Incomplete response: {choice.finish_reason}")
                    raw = message.content or ""
                    data = json.loads(raw)
                    items = data["subtitles"]
                    ids = [item["id"] for item in items]
                    errors = []
                    if ids != [r["id"] for r in targets] or len(set(ids)) != len(ids):
                        errors.append("Missing, duplicate, or unordered IDs")
                    by_id = {item["id"]: item for item in items if ids.count(item["id"]) == 1}
                    if not isinstance(data["notes"], list) or not all(isinstance(n, str) for n in data["notes"]):
                        raise ValueError("Invalid global notes")
                    for record in targets:
                        item = by_id.get(record["id"])
                        if item is None:
                            errors.append(f"Subtitle {record['id']}: Missing or duplicate ID")
                            continue
                        if not isinstance(item.get("text"), str) or not isinstance(item.get("notes"), list) or not all(isinstance(n, str) for n in item["notes"]):
                            errors.append(f"Subtitle {record['id']}: Invalid text or notes")
                            continue
                        problems = validate_subtitle(item["text"], record["duration_ms"], record["required_tags"])
                        if unsafe_ass(item["text"], record["required_tags"]):
                            errors.append(f"Subtitle {item['id']}: Empty text or altered ASS formatting")
                            continue
                        previous = accepted.get(record["id"])
                        if previous is None or len(problems) < len(previous[1]):
                            accepted[record["id"]] = (item, problems)
                        errors.extend(f"Subtitle {item['id']}: {p}" for p in problems)
                    accepted_notes.extend(n for n in data["notes"] if n not in accepted_notes)
                    if not errors:
                        break
                    last_error = "\n".join(errors)
                    messages.extend([{"role": "assistant", "content": raw},
                                     {"role": "user", "content": "Corrige estos problemas sin perder sentido. Devuelve todos los IDs del bloque:\n" + last_error}])
                except Exception as exc:
                    check_cancel()
                    last_error = str(exc)
                    messages.append({"role": "user", "content": "Respuesta invalida. Repite todo el bloque: " + last_error})
            notes.extend(accepted_notes)
            errors = []
            for record in targets:
                candidate = accepted.get(record["id"])
                if candidate is None:
                    fallback = translations.get("gpt")
                    if fallback is None:
                        raise RuntimeError(
                            f"Subtitle {record['id']}: no GPT translation available for fallback: {last_error}"
                        )
                    text = fallback[record["id"] - 1]
                    # Keep GPT's wording while preventing physical ASS event breaks.
                    text = text.replace("\r\n", r"\N").replace("\r", r"\N").replace("\n", r"\N")
                    problems = validate_subtitle(text, record["duration_ms"], record["required_tags"])
                    outputs[record["id"]] = text + "{FINAL PASS FALLBACK GPT}"
                    unresolved += 1
                    reason = f"Subtitle {record['id']}: FALLBACK GPT; reviewer failed: {last_error}"
                    notes.append(reason)
                    notes.extend(f"Subtitle {record['id']} (fallback GPT): {p}" for p in problems)
                    log(f"[Final Pass] {reason}")
                    continue
                item, problems = candidate
                errors.extend(f"Subtitle {record['id']}: {p}" for p in problems)
                text = item["text"]
                outputs[item["id"]] = text
                record["final_text"] = text
                notes.extend(f"Subtitle {item['id']} ({record['start_ms']}-{record['end_ms']} ms): {n}" for n in item["notes"])
            if errors:
                unresolved += len(errors)
                notes.append("UNRESOLVED WARNINGS after three attempts (best proposal preserved without truncation):")
                notes.extend(errors)

        check_cancel()
        for i, ev in enumerate(events, 1):
            ev["parts"][ev["text_idx"]] = outputs[i]
            lines[ev["line_index"]] = f"{ev['leading']}{ev['prefix']}: " + ",".join(ev["parts"]) + ev["line_ending"]
        # Publish only after every block has been processed.
        temp_path = output_path.with_suffix(".ass.tmp")
        try:
            temp_path.write_text("".join(lines), encoding="utf-8-sig")
            os.replace(temp_path, output_path)
        finally:
            temp_path.unlink(missing_ok=True)
        status = "REVIEW REQUIRED: GPT fallbacks, text limits, or line breaks need review." if unresolved else "Technical limits verified: 20.00 CPS / 40 CPL / 2 lines."
        notes.append(status)
        if len(notes) == 3:
            notes.append("No additional questions or warnings from the model.")
        log(f"[Final Pass] {output_path}: {status}")
        return usage
    except Exception as exc:
        notes.append(f"PROCESS NOT COMPLETED: {exc}. No new final file was published.")
        raise
    finally:
        usage["elapsed_seconds"] = max(0.0, time.perf_counter() - started)
        usage["cost_usd"] = calculate_cost(prices, usage["prompt_tokens"], usage["completion_tokens"])
        usage["requests"] = requests
        summary = (
            f"Final Pass time: {usage['elapsed_seconds']:.2f} s | "
            f"Estimated Final Pass cost: ${usage['cost_usd']:.6f} USD | "
            f"Requests: {requests} (including reviewer retries)"
        )
        notes.append(summary)
        notes.append(f"API tokens: input={usage['prompt_tokens']}, output={usage['completion_tokens']}")
        notes.append(
            f"Configured USD/1000 token prices: input={prices.get('input', 0)}, "
            f"output={prices.get('output', 0)}. Estimate; not the provider invoice."
        )
        if missing_usage or requests > responses:
            notes.append(
                f"INCOMPLETE USAGE: {missing_usage} responses without token data and "
                f"{requests - responses} requests without responses. The reported cost only "
                "includes tokens reported by the API; additional usage may exist."
            )
        if requests and (float(prices.get("input", 0) or 0) <= 0 or float(prices.get("output", 0) or 0) <= 0):
            notes.append("WARNING: some prices are zero; configure cost_per_1k.gpt to estimate costs.")
        log(f"[Final Pass] {summary}")
        notes_path.write_text("\n".join(notes) + "\n", encoding="utf-8-sig")
        log(f"[Final Pass] Notes: {notes_path}")
        if open_notes and os.name == "nt":
            try:
                subprocess.Popen(["notepad.exe", str(notes_path.resolve())])
            except OSError as exc:
                log(f"[Final Pass] Could not open Notepad: {exc}")
