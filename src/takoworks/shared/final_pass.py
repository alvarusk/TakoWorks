"""Final translation review with measured ASS limits and a human-readable report."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path

from .openai_client import OPENAI_MODEL, create_openai_client
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


def validate_subtitle(text, duration_ms, required_tags=()):
    lines = visible_lines(text)
    problems = []
    if not text.strip() or not any(line.strip() for line in lines):
        problems.append("Texto vacio")
    if len(lines) > 2:
        problems.append(f"{len(lines)} lineas (maximo 2)")
    for i, line in enumerate(lines, 1):
        if len(line) > 40:
            problems.append(f"Linea {i}: {len(line)} CPL (maximo 40)")
        if i < len(lines):
            words = re.findall(r"\w+", line.lower())
            if words and words[-1] in FUNCTION_WORDS:
                problems.append(f"Corte tras palabra funcional: {words[-1]}")
    count = sum(map(len, lines))
    if duration_ms <= 0:
        problems.append("Duracion nula, negativa o desconocida; no se puede verificar CPS")
    elif count * 1000 > 20 * duration_ms:
        problems.append(f"{count * 1000 / duration_ms:.2f} CPS (maximo 20,00)")
    if formatting_tags(text) != list(required_tags):
        problems.append("Se han alterado los tags ASS de formato")
    if any(tag not in required_tags for tag in TAG_RE.findall(text)):
        problems.append("Tags o comentarios ASS no autorizados")
    if re.search(r"[{}]", TAG_RE.sub("", text)):
        problems.append("Llaves ASS sin cerrar o no autorizadas")
    if "\n" in text or "\r" in text:
        problems.append("Salto fisico no permitido dentro de un evento ASS")
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
    """Preserve event metadata; return API token usage. Never truncate to fit limits."""
    output_path = Path(output_path)
    notes_path = output_path.with_suffix(".txt")
    notes = [f"Final Pass | modelo: {model}", f"Referencia: {reference_path}"]
    notes.extend(source_notes)
    usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def check_cancel():
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Cancelado")

    try:
        check_cancel()
        if output_path.resolve() == Path(reference_path).resolve():
            raise ValueError("El archivo final no puede sobrescribir la referencia.")
        lines, events = _dialogues(reference_path)
        if not events:
            raise ValueError("No hay eventos Dialogue para revisar.")
        if not translations:
            raise ValueError("No hay traducciones disponibles para Final Pass.")
        for engine, texts in translations.items():
            if len(texts) != len(events):
                raise ValueError(f"{engine}: las traducciones no coinciden con los eventos de referencia.")
        client = client or create_openai_client()
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
                    notes.append(f"Subtitulo {record['id']}: dibujo ASS conservado; limites de texto no aplicables.")
            if not targets:
                continue
            payload = {"source_lang": source_lang, "subtitles": targets,
                       "context_before": records[max(0, offset - 2):offset],
                       "context_after": records[offset + 20:offset + 22]}
            messages = [{"role": "system", "content": rules},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
            accepted = None
            last_error = ""
            for attempt in range(3):
                check_cancel()
                log(f"[Final Pass] {offset + 1}-{offset + len(batch)}, intento {attempt + 1}/3")
                try:
                    response = client.chat.completions.create(
                        model=model, messages=messages, response_format=_response_format(),
                        max_completion_tokens=6000,
                    )
                    check_cancel()
                    api_usage = getattr(response, "usage", None)
                    for key in usage:
                        usage[key] += int(getattr(api_usage, key, 0) or 0)
                    choice = response.choices[0]
                    message = choice.message
                    if getattr(message, "refusal", None):
                        raise ValueError(f"Rechazo del modelo: {message.refusal}")
                    if getattr(choice, "finish_reason", "stop") != "stop":
                        raise ValueError(f"Respuesta incompleta: {choice.finish_reason}")
                    raw = message.content or ""
                    data = json.loads(raw)
                    items = data["subtitles"]
                    ids = [item["id"] for item in items]
                    if ids != [r["id"] for r in targets] or len(set(ids)) != len(ids):
                        raise ValueError("IDs ausentes, repetidos o fuera de orden")
                    if not isinstance(data["notes"], list) or not all(isinstance(n, str) for n in data["notes"]):
                        raise ValueError("Notas globales invalidas")
                    errors = []
                    for record, item in zip(targets, items):
                        if not isinstance(item["text"], str) or not isinstance(item["notes"], list) or not all(isinstance(n, str) for n in item["notes"]):
                            raise ValueError("Texto o notas invalidos")
                        problems = validate_subtitle(item["text"], record["duration_ms"], record["required_tags"])
                        if (formatting_tags(item["text"]) != record["required_tags"]
                                or "\n" in item["text"] or "\r" in item["text"]
                                or any(tag not in record["required_tags"] for tag in TAG_RE.findall(item["text"]))
                                or re.search(r"[{}]", TAG_RE.sub("", item["text"]))
                                or not any(line.strip() for line in visible_lines(item["text"]))):
                            raise ValueError(f"Subtitulo {item['id']}: texto vacio o formato ASS alterado")
                        errors.extend(f"Subtitulo {item['id']}: {p}" for p in problems)
                    if accepted is None or len(errors) < len(accepted[2]):
                        accepted = (items, data["notes"], errors)
                    if not errors:
                        break
                    last_error = "\n".join(errors)
                    messages.extend([{"role": "assistant", "content": raw},
                                     {"role": "user", "content": "Corrige estos problemas sin perder sentido. Devuelve todos los IDs del bloque:\n" + last_error}])
                except Exception as exc:
                    check_cancel()
                    last_error = str(exc)
                    messages.append({"role": "user", "content": "Respuesta invalida. Repite todo el bloque: " + last_error})
            if accepted is None:
                raise RuntimeError(f"Bloque {offset + 1}: no se obtuvo una respuesta valida: {last_error}")
            items, global_notes, errors = accepted
            notes.extend(global_notes)
            for item, record in zip(items, targets):
                text = item["text"]
                outputs[item["id"]] = text
                record["final_text"] = text
                notes.extend(f"Subtitulo {item['id']} ({record['start_ms']}-{record['end_ms']} ms): {n}" for n in item["notes"])
            if errors:
                unresolved += len(errors)
                notes.append("AVISOS PENDIENTES tras tres intentos (se conserva la mejor propuesta sin recortarla):")
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
        status = "REQUIERE REVISION: hay limites o cortes pendientes." if unresolved else "Limites tecnicos verificados: 20,00 CPS / 40 CPL / 2 lineas."
        notes.append(status)
        if len(notes) == 3:
            notes.append("Sin dudas ni avisos adicionales del modelo.")
        log(f"[Final Pass] {output_path}: {status}")
        return usage
    except Exception as exc:
        notes.append(f"PROCESO NO COMPLETADO: {exc}. No se ha publicado un nuevo archivo final.")
        raise
    finally:
        notes.append(f"Tokens API: entrada={usage['prompt_tokens']}, salida={usage['completion_tokens']}")
        notes_path.write_text("\n".join(notes) + "\n", encoding="utf-8-sig")
        log(f"[Final Pass] Notas: {notes_path}")
        if open_notes and os.name == "nt":
            try:
                subprocess.Popen(["notepad.exe", str(notes_path.resolve())])
            except OSError as exc:
                log(f"[Final Pass] No se pudo abrir Bloc de notas: {exc}")
