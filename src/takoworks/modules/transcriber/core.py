import os
import argparse
import json
import time
import copy
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import List, Set, Dict, Optional, Tuple
import html
import re
from functools import lru_cache
from ... import config as app_config
from .ass_utils import (
    _ass_hide,
    _ass_hide_prefix,
    _ass_sanitize_braces,
    _ass_unsanitize_braces,
)
from .context_notes import (
    build_contextual_explanation_prompt,
    build_contextual_explanation_repair_prompt,
    contains_japanese_script,
    parse_contextual_explanation_response,
)
from .json_utils import (
    RomanizationParseResult,
    TranslationParseResult,
    parse_json_romanizations_result,
    parse_json_translations_result,
)
from .source_type import describe_source_type, normalize_source_type

try:
    import requests  # type: ignore
except Exception:
    requests = None

import pysubs2
import torch
from pykakasi import kakasi
from pypinyin import lazy_pinyin, Style

from fugashi import Tagger

from openai import OpenAI          # OpenAI + DeepSeek (API compatible)
import anthropic                   # Claude
try:
    from google import genai as google_genai  # Gemini SDK nuevo
    from google.genai import types as google_genai_types
    legacy_genai = None
except Exception:  # pragma: no cover - fallback para entornos antiguos
    google_genai = None
    google_genai_types = None
    try:
        import google.generativeai as legacy_genai  # Gemini SDK antiguo
    except Exception:  # pragma: no cover - fallback opcional
        legacy_genai = None

from typing import Callable, Optional, List

# Tipo de callback de progreso:
# stage: "transcribir" / "romanizar" / "pulir"
# current: índice (1-based)
# total: total de subtítulos
# text: texto del subtítulo actual
ProgressCallback = Optional[Callable[[str, int, int, str], None]]

# ============================================================
#  CONFIGURACIÓN: MODELOS, CLAVES API Y DICCIONARIOS
# ============================================================

OPENAI_MODEL   = "gpt-5.5"                  # OpenAI (ajusta si hace falta)
CLAUDE_MODEL   = "claude-opus-4-7"          # Anthropic (ajusta si hace falta)
CONTEXT_NOTE_MODEL = "claude-sonnet-4-6"    # Prompt Explicativo / nota contextual
GEMINI_MODEL   = "gemini-3-flash-preview"   # Gemini 3 Flash Preview
DEEPSEEK_MODEL = "deepseek-v4-flash"        # DeepSeek (OpenAI-like)
OPENAI_TIMEOUT_S = 180
OPENAI_MAX_RETRIES = 2
OPENAI_BLOCK_ATTEMPTS = 3
MODEL_BLOCK_ATTEMPTS = 3
CONTEXT_NOTE_MAX_TOKENS = 800
ROMANIZATION_MAX_TOKENS = 1200


def _add_openai_temperature(request_kwargs: Dict[str, object], temperature: float) -> None:
    if OPENAI_MODEL != "gpt-5.5":
        request_kwargs["temperature"] = temperature


def _read_api_key(key: str, env_name: str) -> str:
    env_val = os.getenv(env_name)
    if env_val:
        return env_val
    try:
        cfg = app_config.load_config()
        api_keys = cfg.get("api_keys", {})
        if isinstance(api_keys, dict):
            return str(api_keys.get(key, "") or "")
    except Exception:
        pass
    return ""


OPENAI_API_KEY   = _read_api_key("openai", "OPENAI_API_KEY")
ANTHROPIC_API_KEY = _read_api_key("anthropic", "ANTHROPIC_API_KEY")
GEMINI_API_KEY    = _read_api_key("gemini", "GEMINI_API_KEY")
DEEPSEEK_API_KEY  = _read_api_key("deepseek", "DEEPSEEK_API_KEY")

CHUNK_SIZE   = 20  # tamaño de lote para GPT/Claude/DeepSeek
GEMINI_CHUNK = 3  # bloques más pequeños para Gemini, más estable

# ============================================================
#  NOMBRES DE MODELOS (UI) Y NORMALIZACIÓN
# ============================================================

# Nombres que quieres ver en la GUI / logs / HTML
DISPLAY_NAMES = {
    "gpt": "GPT-5.5",
    "claude": "Claude Opus 4.7",
    "context_note": "Prompt Explicativo (Claude Sonnet 4.6)",
    "gemini": "Gemini 3 Flash",
    "deepseek": "DeepSeek V4 Flash",
    "romanization": "Romanization (DeepSeek V4 Flash)",
}

# Alias aceptados en --models (CLI/GUI). Se normalizan a claves internas:
MODEL_ALIASES = {
    "gpt-5.5": "gpt",
    "gpt-5": "gpt",
    "gpt5": "gpt",
    "gpt": "gpt",
    "openai": "gpt",
    "chatgpt": "gpt",

    "claude opus 4.7": "claude",
    "claude-opus-4-7": "claude",
    "claude": "claude",
    "anthropic": "claude",

    "gemini 3 flash": "gemini",
    "gemini-3-flash-preview": "gemini",
    "gemini 2.5 flash": "gemini",
    "gemini-2.5-flash": "gemini",
    "gemini": "gemini",

    "deepseek v4 flash": "deepseek",
    "deepseek-v4-flash": "deepseek",
    "deepseek": "deepseek",
    "deepseek-chat": "deepseek",
}

def normalize_models_arg(models_str: str) -> Set[str]:
    """
    Acepta tanto: 'gpt,claude' como 'GPT-5.5,Claude Opus 4.7,Gemini 3 Flash,DeepSeek V4 Flash'
    y devuelve el set interno: {'gpt','claude','gemini','deepseek'}.
    """
    if models_str is None:
        return set()

    items = [x.strip() for x in str(models_str).split(",") if x.strip()]
    out: Set[str] = set()

    for it in items:
        key = MODEL_ALIASES.get(it.lower(), it.lower())
        if key in DISPLAY_NAMES:
            out.add(key)

    return out

# ============================================================
#  COSTES API + SUPABASE
# ============================================================


@dataclass
class ApiUsage:
    engine: str
    model_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def merge_api_usage(base: ApiUsage, extra: Optional[ApiUsage]) -> ApiUsage:
    if extra is None:
        return base
    base.prompt_tokens += extra.prompt_tokens
    base.completion_tokens += extra.completion_tokens
    base.cost_usd += extra.cost_usd
    if not base.model_name and extra.model_name:
        base.model_name = extra.model_name
    return base


def _record_phase_time(phase_timings: Optional[Dict[str, float]], key: str, elapsed: float) -> None:
    if phase_timings is None:
        return
    phase_timings[key] = phase_timings.get(key, 0.0) + max(0.0, elapsed)


def _format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0.0))
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes = int(seconds // 60)
    rem = seconds - (minutes * 60)
    return f"{minutes} min {rem:.1f} s"


def _format_cost(cost_usd: float) -> str:
    return f"${float(cost_usd or 0.0):.4f}"


def _split_ass_text_lines(text: str) -> List[str]:
    raw = text or ""
    lines: List[str] = []
    buf: List[str] = []
    brace_depth = 0
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch == "{":
            brace_depth += 1
            buf.append(ch)
            i += 1
            continue
        if ch == "}":
            if brace_depth > 0:
                brace_depth -= 1
            buf.append(ch)
            i += 1
            continue
        if ch == "\\" and i + 1 < len(raw) and raw[i + 1] == "N" and brace_depth == 0:
            lines.append("".join(buf))
            buf = []
            i += 2
            continue
        buf.append(ch)
        i += 1
    lines.append("".join(buf))
    return lines


def _extract_braced_text(raw: str) -> str:
    txt = (raw or "").strip()
    if txt.startswith("{") and txt.endswith("}"):
        txt = txt[1:-1]
    return _ass_unsanitize_braces(txt).replace("\\N", "\n")


def _is_hidden_ass_line(line: str) -> bool:
    txt = (line or "").strip()
    return txt.startswith("{") and txt.endswith("}")


def _normalize_dialogue_marker_line(line: str) -> str:
    txt = (line or "").strip()
    if txt.startswith("-"):
        return txt
    # Some source lines arrive as "！-texto"; the dialogue marker should own
    # the start of the line so romaji/notes can keep both speakers grouped.
    return re.sub(r"^[\s!！?？。．…]+(?=-)", "", txt)


def _has_dialogue_marker(line: str) -> bool:
    return _normalize_dialogue_marker_line(line).lstrip().startswith("-")


def _strip_dialogue_marker(line: str) -> str:
    txt = _normalize_dialogue_marker_line(line).lstrip()
    if txt.startswith("-"):
        return txt[1:].strip()
    return txt.strip()


def _dialogue_original_line_count(ev: pysubs2.SSAEvent, lines: List[str]) -> int:
    actor = str(getattr(ev, "name", "") or "")
    if ";" not in actor:
        return 1

    count = 0
    for line in lines:
        if _is_hidden_ass_line(line):
            break
        if not _has_dialogue_marker(line):
            break
        count += 1
    return count if count >= 2 else 1


def _split_event_original_and_extra(ev: pysubs2.SSAEvent) -> Tuple[List[str], List[str]]:
    lines = _split_ass_text_lines(getattr(ev, "text", "") or "")
    if not lines:
        return [], []

    original_count = _dialogue_original_line_count(ev, lines)
    original_lines = lines[:original_count]
    if original_count > 1:
        original_lines = [_normalize_dialogue_marker_line(line) for line in original_lines]
    return original_lines, lines[original_count:]


def _event_source_text(ev: pysubs2.SSAEvent) -> str:
    original_lines, _extra_lines = _split_event_original_and_extra(ev)
    return "\\N".join(line.strip() for line in original_lines if line.strip()).strip()


def _romanize_source_text(
    text: str,
    lang: str,
    romaji_converter,
    ja_tagger: Optional[Tagger],
) -> str:
    out_lines: List[str] = []
    for raw_line in (text or "").split("\\N"):
        line = _normalize_dialogue_marker_line(raw_line)
        has_marker = _has_dialogue_marker(line)
        source = _strip_dialogue_marker(line) if has_marker else line.strip()
        if not source:
            out_lines.append("-" if has_marker else "")
            continue

        if lang == "ja":
            converted = ""
            if romaji_converter is not None:
                if ja_tagger is not None:
                    converted = japanese_to_romaji_pretty(source, romaji_converter, ja_tagger)
                else:
                    converted = japanese_to_romaji_line(source, romaji_converter)
        elif lang == "zh":
            converted = text_to_pinyin(source)
        else:
            converted = ""

        converted = (converted or "").strip()
        if has_marker and converted:
            converted = "- " + converted
        out_lines.append(converted)
    return "\\N".join(line for line in out_lines if line).strip()


def _safe_int(val) -> int:
    try:
        return int(val or 0)
    except Exception:
        return 0


def _read_price(model_key: str, kind: str, default: float) -> float:
    env_name = f"COST_{model_key.upper()}_{kind.upper()}_PER_1K"
    raw = os.getenv(env_name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except Exception:
        print(f"[Costes] Valor no valido en {env_name}: {raw}. Se usa {default}.")
        return default


DEFAULT_PRICE_PER_1K: Dict[str, Dict[str, float]] = {
    "gpt": {"input": 0.005, "output": 0.03},
    "claude": {"input": 0.005, "output": 0.025},
    "context_note": {"input": 0.0015, "output": 0.0075},
    "gemini": {"input": 0.0005, "output": 0.003},
    "deepseek": {"input": 0.00014, "output": 0.00028},
}


def _load_price_table() -> Dict[str, Dict[str, float]]:
    cfg_costs: Dict[str, Dict[str, float]] = {}
    try:
        cfg = app_config.load_config()
        raw = cfg.get("cost_per_1k", {})
        if isinstance(raw, dict):
            cfg_costs = raw  # shallow; se accede con get seguro abajo
    except Exception as e:
        print(f"[Costes] No se pudo leer cost_per_1k de config.json: {e}")

    table: Dict[str, Dict[str, float]] = {}
    for key, defaults in DEFAULT_PRICE_PER_1K.items():
        cfg_entry = cfg_costs.get(key, {}) if isinstance(cfg_costs, dict) else {}
        table[key] = {
            "input": _read_price(
                key,
                "input",
                cfg_entry.get("input", defaults.get("input", 0.0)) if isinstance(cfg_entry, dict) else defaults.get("input", 0.0),
            ),
            "output": _read_price(
                key,
                "output",
                cfg_entry.get("output", defaults.get("output", 0.0)) if isinstance(cfg_entry, dict) else defaults.get("output", 0.0),
            ),
        }
    return table


_WARNED_PRICING: Set[str] = set()
_WARNED_MISSING_USAGE: Set[str] = set()
_WARNED_JA_TAGGER: bool = False
_JA_TAGGER_FAILED: bool = False
_WARNED_CONTEXT_NOTE_MISSING_KEY: bool = False
_WARNED_CONTEXT_NOTE_CLIENT: bool = False


def estimate_cost(model_key: str, prompt_tokens: int, completion_tokens: int) -> float:
    # Reload pricing from the effective config each time so a long-lived GUI
    # process does not keep stale zero-cost values after the user updates config.
    prices = _load_price_table().get(model_key, {})
    in_price = float(prices.get("input", 0.0) or 0.0)
    out_price = float(prices.get("output", 0.0) or 0.0)

    if (in_price <= 0 or out_price <= 0) and model_key not in _WARNED_PRICING:
        print(
            f"[Costes] Precios por token no configurados para {DISPLAY_NAMES.get(model_key, model_key)}. "
            f"Define COST_{model_key.upper()}_INPUT_PER_1K y COST_{model_key.upper()}_OUTPUT_PER_1K para costes reales."
        )
        _WARNED_PRICING.add(model_key)

    return (prompt_tokens / 1000.0) * in_price + (completion_tokens / 1000.0) * out_price


def _warn_missing_usage(model_key: str) -> None:
    if model_key in _WARNED_MISSING_USAGE:
        return
    display = DISPLAY_NAMES.get(model_key, model_key)
    print(
        f"[Costes] {display} no devolvió metadatos de uso; se asume coste=0. "
        "Revisa la versión del SDK o las opciones de la API si quieres token counts reales."
    )
    _WARNED_MISSING_USAGE.add(model_key)


def _build_translation_response_format(expected_count: int) -> Dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "subtitle_translations",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "translations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": expected_count,
                        "maxItems": expected_count,
                    }
                },
                "required": ["translations"],
            },
        },
    }


def _build_retry_user_prompt(user_prompt: str, expected_count: int, attempt: int) -> str:
    if attempt <= 1:
        return user_prompt
    return (
        user_prompt
        + "\n\nCORRECCION OBLIGATORIA:\n"
        + f"- Debes devolver EXACTAMENTE {expected_count} elementos en \"translations\".\n"
        + "- No fusiones lineas.\n"
        + "- No dividas lineas.\n"
        + "- No omitas ninguna linea.\n"
        + "- Devuelve SOLO JSON valido."
    )


def _save_translation_debug_response(
    debug_dir: Optional[str],
    model_key: str,
    start_line: int,
    end_line: int,
    attempt: int,
    issue: str,
    chunk: List[str],
    raw_response: str,
    parse_result: Optional[TranslationParseResult] = None,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    if not debug_dir:
        return

    os.makedirs(debug_dir, exist_ok=True)
    safe_issue = re.sub(r"[^a-zA-Z0-9_-]+", "_", issue).strip("_") or "issue"
    file_name = f"{model_key}_{start_line:05d}-{end_line:05d}_attempt{attempt}_{safe_issue}.json"
    path = os.path.join(debug_dir, file_name)

    payload: Dict[str, object] = {
        "model": model_key,
        "start_line": start_line,
        "end_line": end_line,
        "attempt": attempt,
        "issue": issue,
        "expected_count": len(chunk),
        "source_lines": chunk,
        "raw_response": raw_response,
    }
    if parse_result is not None:
        payload.update(
            {
                "parser": parse_result.parser,
                "raw_count": parse_result.raw_count,
                "exact_match": parse_result.exact_match,
                "normalized": parse_result.normalized,
                "used_fallback": parse_result.used_fallback,
                "missing_indices": parse_result.missing_indices,
                "extra_count": parse_result.extra_count,
                "error": parse_result.error,
                "normalized_translations": parse_result.translations,
            }
        )
    if extra:
        payload["extra"] = extra

    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"[DEBUG {DISPLAY_NAMES.get(model_key, model_key)}] Respuesta guardada en {path}")
    except Exception as e:
        print(f"[DEBUG {DISPLAY_NAMES.get(model_key, model_key)}] No se pudo guardar respuesta bruta: {e}")


def _log_translation_count_issue(
    model_key: str,
    start_line: int,
    end_line: int,
    parse_result: TranslationParseResult,
    chunk: Optional[List[str]] = None,
) -> None:
    detail = (
        f"[{DISPLAY_NAMES.get(model_key, model_key)}] Bloque {start_line}-{end_line}: "
        f"esperadas {parse_result.expected_count}, recibidas {parse_result.raw_count} "
        f"(parser={parse_result.parser})."
    )
    if parse_result.missing_indices:
        missing_abs = [start_line + idx for idx in parse_result.missing_indices]
        detail += f" Missing positions/lines: {missing_abs}."
    if parse_result.extra_count:
        detail += f" {parse_result.extra_count} extra translations."
    if parse_result.error:
        detail += f" Error: {parse_result.error}."
    print(detail)

    if parse_result.missing_indices and chunk:
        print(f"[{DISPLAY_NAMES.get(model_key, model_key)}] Source lines returned without translation by the model:")
        for idx in parse_result.missing_indices:
            absolute_line = start_line + idx
            source_text = chunk[idx].strip() if idx < len(chunk) else ""
            print(f"  - line {absolute_line}: {source_text}")

    if parse_result.extra_count and chunk:
        preview_count = min(3, len(chunk))
        if preview_count:
            print(f"[{DISPLAY_NAMES.get(model_key, model_key)}] Context for the block with extra translations:")
            for rel_idx in range(preview_count):
                absolute_line = start_line + rel_idx
                source_text = chunk[rel_idx].strip()
                print(f"  - line {absolute_line}: {source_text}")


def _read_supabase_value(field: str, env_names) -> str:
    for env in env_names:
        val = os.getenv(env)
        if val:
            return val
    try:
        cfg = app_config.load_config()
        supabase_cfg = cfg.get("supabase", {})
        if isinstance(supabase_cfg, dict):
            return str(supabase_cfg.get(field, "") or "")
    except Exception:
        pass
    return ""


SUPABASE_URL = _read_supabase_value("url", ["SUPABASE_URL"])
SUPABASE_SERVICE_KEY = _read_supabase_value(
    "service_key",
    ["SUPABASE_SERVICE_KEY", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_ANON_KEY"],
)
SUPABASE_COST_TABLE = os.getenv("SUPABASE_COST_TABLE", "voicex_api_costs")


def log_cost_summary(run_id: str, usage_by_model: Dict[str, ApiUsage], series_name: str, episode: str) -> None:
    if not usage_by_model:
        return

    print(f"[Costes] Resumen tokens/coste para {series_name or 'serie-desconocida'} / {episode} (run {run_id}):")
    for key in sorted(usage_by_model.keys()):
        usage = usage_by_model[key]
        display = DISPLAY_NAMES.get(key, key)
        print(
            f"[Costes] {display}: prompt={usage.prompt_tokens} completion={usage.completion_tokens} "
            f"total={usage.total_tokens} cost_usd=${usage.cost_usd:.4f}"
        )
    total_tokens = sum(u.total_tokens for u in usage_by_model.values())
    total_cost = sum(u.cost_usd for u in usage_by_model.values())
    print(f"[Costes] TOTAL: tokens={total_tokens} cost_usd=${total_cost:.4f}")


def log_time_cost_breakdown(
    phase_timings: Dict[str, float],
    context_note_usage: ApiUsage,
    model_timings: Dict[str, float],
    usage_by_model: Dict[str, ApiUsage],
    total_elapsed: float,
) -> None:
    print("[Tiempo/coste] Desglose final:")

    asr_seconds = phase_timings.get("asr", 0.0)
    punctuation_seconds = phase_timings.get("punctuation", 0.0)
    embedding_seconds = phase_timings.get("embedding", 0.0)
    context_seconds = phase_timings.get("context_note", 0.0)
    roman_seconds = phase_timings.get("romanization", 0.0)
    displayed_seconds = asr_seconds + punctuation_seconds + embedding_seconds + context_seconds + roman_seconds
    displayed_cost = context_note_usage.cost_usd

    print(f"ASR: {_format_elapsed(asr_seconds)}; {_format_cost(0.0)}")
    print(f"Puntuación: {_format_elapsed(punctuation_seconds)}; {_format_cost(0.0)}")
    print(f"Embedding ASS: {_format_elapsed(embedding_seconds)}; {_format_cost(0.0)}")
    print(f"Prompts explicativos: {_format_elapsed(context_seconds)}; {_format_cost(context_note_usage.cost_usd)}")
    print(f"Romanización: {_format_elapsed(roman_seconds)}; {_format_cost(0.0)}")

    for key in ("gpt", "claude", "gemini", "deepseek"):
        if key not in model_timings and key not in usage_by_model:
            continue
        usage = usage_by_model.get(key, ApiUsage(engine=key, model_name=""))
        seconds = model_timings.get(key, 0.0)
        displayed_seconds += seconds
        displayed_cost += usage.cost_usd
        print(f"{DISPLAY_NAMES.get(key, key)}: {_format_elapsed(seconds)}; {_format_cost(usage.cost_usd)}")

    other_seconds = max(0.0, total_elapsed - displayed_seconds)
    if other_seconds >= 0.05:
        displayed_seconds += other_seconds
        print(f"Otros pasos: {_format_elapsed(other_seconds)}; {_format_cost(0.0)}")

    print(f"Total: {_format_elapsed(displayed_seconds)}; {_format_cost(displayed_cost)}")


def log_time_cost_breakdown_v2(
    phase_timings: Dict[str, float],
    romanization_usage: ApiUsage,
    context_note_usage: ApiUsage,
    model_timings: Dict[str, float],
    usage_by_model: Dict[str, ApiUsage],
    total_elapsed: float,
) -> None:
    print("[Tiempo/coste] Desglose final:")

    asr_seconds = phase_timings.get("asr", 0.0)
    punctuation_seconds = phase_timings.get("punctuation", 0.0)
    embedding_seconds = phase_timings.get("embedding", 0.0)
    context_seconds = phase_timings.get("context_note", 0.0)
    roman_seconds = phase_timings.get("romanization", 0.0)
    displayed_seconds = asr_seconds + punctuation_seconds + embedding_seconds + context_seconds + roman_seconds
    displayed_cost = context_note_usage.cost_usd + romanization_usage.cost_usd

    print(f"ASR: {_format_elapsed(asr_seconds)}; {_format_cost(0.0)}")
    print(f"Puntuación: {_format_elapsed(punctuation_seconds)}; {_format_cost(0.0)}")
    print(f"Embedding ASS: {_format_elapsed(embedding_seconds)}; {_format_cost(0.0)}")
    print(f"Prompts explicativos: {_format_elapsed(context_seconds)}; {_format_cost(context_note_usage.cost_usd)}")
    print(f"Romanization: {_format_elapsed(roman_seconds)}; {_format_cost(romanization_usage.cost_usd)}")

    for key in ("gpt", "claude", "gemini", "deepseek"):
        if key not in model_timings and key not in usage_by_model:
            continue
        usage = usage_by_model.get(key, ApiUsage(engine=key, model_name=""))
        seconds = model_timings.get(key, 0.0)
        displayed_seconds += seconds
        displayed_cost += usage.cost_usd
        print(f"{DISPLAY_NAMES.get(key, key)}: {_format_elapsed(seconds)}; {_format_cost(usage.cost_usd)}")

    other_seconds = max(0.0, total_elapsed - displayed_seconds)
    if other_seconds >= 0.05:
        displayed_seconds += other_seconds
        print(f"Otros pasos: {_format_elapsed(other_seconds)}; {_format_cost(0.0)}")

    print(f"Total: {_format_elapsed(displayed_seconds)}; {_format_cost(displayed_cost)}")


def persist_costs_to_supabase(
    run_id: str,
    series_name: str,
    episode: str,
    lang: str,
    usage_by_model: Dict[str, ApiUsage],
) -> None:
    if not usage_by_model:
        return
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("[Costes] SUPABASE_URL/SUPABASE_SERVICE_KEY no definidos; se omite guardado remoto.")
        return
    if requests is None:
        print("[Costes] Libreria requests no disponible; no se envian datos a Supabase.")
        return

    base_url = SUPABASE_URL.rstrip("/")
    url = f"{base_url}/rest/v1/{SUPABASE_COST_TABLE}"
    ts = datetime.utcnow().isoformat() + "Z"

    rows = []
    for key, usage in usage_by_model.items():
        rows.append(
            {
                "run_id": run_id,
                "series": series_name,
                "episode": episode,
                "lang": lang,
                "engine": key,
                "model_name": usage.model_name,
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "cost_usd": round(usage.cost_usd, 6),
                "currency": "USD",
                "created_at": ts,
            }
        )

    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

    try:
        resp = requests.post(url, headers=headers, json=rows, timeout=15)
        if resp.status_code not in (200, 201):
            print(f"[Costes] Error al escribir en Supabase ({resp.status_code}): {resp.text}")
        else:
            print(f"[Costes] Costes guardados en Supabase ({len(rows)} filas).")
    except Exception as e:
        print(f"[Costes] No se pudo enviar a Supabase: {e}")

# ============================================================
#  ESTADO GLOBAL PARA ANÁLISIS
# ============================================================

_ja_tagger: Optional[Tagger]          = None


def _ensure_ja_tagger() -> Optional[Tagger]:
    global _ja_tagger, _WARNED_JA_TAGGER, _JA_TAGGER_FAILED
    if _ja_tagger is not None:
        return _ja_tagger
    if _JA_TAGGER_FAILED:
        return None
    try:
        _ja_tagger = Tagger()
    except Exception as e:
        _JA_TAGGER_FAILED = True
        if not _WARNED_JA_TAGGER:
            print(
                "[JA] No se pudo iniciar Tagger (fugashi/UniDic). "
                "Se usa kakasi simple para la romanizacion. "
                f"Detalle: {e}"
            )
            _WARNED_JA_TAGGER = True
        _ja_tagger = None
    return _ja_tagger


# ============================================================
#  UTILIDADES: DICCIONARIO YOMITAN
# ============================================================

# ============================================================
#  PUNTUACIÓN (MODELOS LIBRES, SIN GPT)
# ============================================================

def strip_ja_punct(text: str) -> str:
    """Quita solo 「、」 y 「。」 (el modelo japonés las vuelve a insertar)."""
    return re.sub(r"[、。]", "", text)


def strip_zh_punct(text: str) -> str:
    """
    Quita los signos chinos que el modelo sabe restaurar:
    ， 、 。 ？ ！ ；   (full-width)
    """
    return re.sub(r"[ ，、。？！；]", "", text)


def refine_japanese_punctuation_free(lines: List[str]) -> List[str]:
    """
    Restauración de puntuación japonesa (modelo libre).
    Requiere: insert_punctuation.py + weight/punctuation_position_model.pth.
    Si no está disponible, devuelve texto original.
    """
    try:
        from .insert_punctuation import process_long_text
    except Exception as e:
        print(
            "[Puntuación ja] No se pudo cargar insert_punctuation.py o sus pesos "
            "(weight/punctuation_position_model.pth); se usa texto original. "
            f"Detalle: {e}"
        )
        return lines

    out: List[str] = []
    for t in lines:
        txt = (t or "").strip()
        if not txt:
            out.append(t)
            continue

        cleaned = strip_ja_punct(txt)
        try:
            fixed = process_long_text(cleaned)
            out.append(fixed)
        except Exception as e:
            print(f"[JA punctuation] Error processing line, keeping original: {e}")
            out.append(t)

    return out

@lru_cache(maxsize=1)
def _get_zh_punct_components():
    """
    Carga el modelo libre de restauración de puntuación en chino:
    - Modelo: p208p2002/zh-wiki-punctuation-restore
    - Librería: zhpr (DocumentDataset, merge_stride, decode_pred)
    Si falta algo, lanza RuntimeError (lo capturaremos más arriba).
    """
    try:
        from zhpr.predict import DocumentDataset, merge_stride, decode_pred  # type: ignore
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError as e:
        raise RuntimeError(
            f"Para la puntuación en chino necesitas instalar zhpr + transformers: {e}"
        )

    model_name = "p208p2002/zh-wiki-punctuation-restore"
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return DocumentDataset, merge_stride, decode_pred, model, tokenizer


def _restore_zh_line(text: str) -> str:
    """
    Restaura la puntuación de UNA línea de chino usando zh-wiki-punctuation-restore.
    """
    from torch.utils.data import DataLoader

    if not text.strip():
        return text

    DocumentDataset, merge_stride, decode_pred, model, tokenizer = _get_zh_punct_components()

    cleaned = strip_zh_punct(text)

    window_size = 100
    step = 75
    dataset = DocumentDataset(cleaned, window_size=window_size, step=step)
    dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=3)

    def predict_step(batch, model, tokenizer):
        out = []
        input_ids = batch
        encodings = {"input_ids": input_ids}
        output = model(**encodings)

        predicted_token_class_id_batch = output["logits"].argmax(-1)
        for predicted_token_class_ids, ids in zip(predicted_token_class_id_batch, input_ids):
            tokens = tokenizer.convert_ids_to_tokens(ids)

            ids_list = ids.tolist()
            try:
                pad_start = ids_list.index(tokenizer.pad_token_id)
            except ValueError:
                pad_start = len(ids_list)

            ids_list = ids_list[:pad_start]
            tokens = tokens[:pad_start]

            predicted_tokens_classes = [
                model.config.id2label[t.item()] for t in predicted_token_class_ids
            ]
            predicted_tokens_classes = predicted_tokens_classes[:pad_start]

            for token, ner in zip(tokens, predicted_tokens_classes):
                out.append((token, ner))

        return out

    model_pred_out = []
    for batch in dataloader:
        batch_out = predict_step(batch, model, tokenizer)
        for out in batch_out:
            model_pred_out.append(out)

    merge_pred_result = merge_stride(model_pred_out, step)
    decoded = decode_pred(merge_pred_result)
    return "".join(decoded)


def refine_chinese_punctuation_free(lines: List[str]) -> List[str]:
    """
    Restauración de puntuación en chino con modelo libre.
    Si falta zhpr/transformers, devuelve las líneas originales.
    """
    try:
        _get_zh_punct_components()
    except Exception as e:
        print(f"[Puntuación zh] zhpr/transformers no disponibles ({e}); se usa texto original.")
        return lines

    out: List[str] = []
    for t in lines:
        try:
            out.append(_restore_zh_line(t))
        except Exception as e:
            print(f"[ZH punctuation] Error processing line, keeping original: {e}")
            out.append(t)
    return out


def refine_punctuation_free(lines: List[str], lang: str) -> List[str]:
    """
    Modelos libres de puntuación:
    - ja: BERT japonés
    - zh: DESACTIVADO por rendimiento, se usa salida cruda del ASR.
    """
    if not lines:
        return lines

    if lang == "ja":
        return refine_japanese_punctuation_free(lines)

    if lang == "zh":
        print("[Puntuación zh] Pulido desactivado (se usa la salida del modelo de voz).")
        return lines

    return lines


# ============================================================
#  ASR + ROMAJI/PINYIN + ANÁLISIS TIPO DICCIONARIO
# ============================================================
def clean_repetitions(text: str) -> str:
    """
    Reducir repeticiones absurdas de caracteres (ej.: たたたたたたたたた...).
    - Colapsa cualquier carácter repetido más de 4 veces seguidas a 4 repeticiones.
    - Si la línea sigue siendo muy larga y con muy poca variedad de caracteres,
      la recorta para evitar monstruos tipo '痛たたたたたたたたたたたたたたた...'.
    """
    import re

    if not text:
        return text

    # 1) Colapsar rachas de cualquier carácter (incluyendo kana/kanji)
    #    Ej.: "たたたたたたた" → "たたたた"
    def _collapse(match):
        ch = match.group(1)
        return ch * 4  # máximo 4 repeticiones seguidas

    text = re.sub(r"(.)\1{4,}", _collapse, text)

    # 2) Si la línea es muy larga pero con pocos caracteres distintos,
    #    probablemente sea una onomatopeya loca → la recortamos.
    if len(text) > 50:
        unique_chars = set(text)
        if len(unique_chars) < 10:
            # nos quedamos con los primeros 30 caracteres, suficiente para "痛たたたた…"
            text = text[:30]

    return text


@lru_cache(maxsize=1)
def _get_transformers_pipeline():
    """
    Import perezoso del pipeline de Hugging Face.

    Evita fallos al importar este modulo cuando la ejecucion no necesita ASR
    y reduce el impacto de dependencias opcionales resueltas por transformers
    durante la carga de `pipeline`.
    """
    try:
        from transformers import pipeline as hf_pipeline
    except Exception as e:
        raise RuntimeError(
            "No se pudo cargar transformers.pipeline. "
            "En la version empaquetada, verifica que el bundle incluya "
            "torchcodec y su metadata de paquete. "
            f"Detalle: {e}"
        ) from e
    return hf_pipeline

def build_asr_pipeline(lang: str):
    """
    Crea el pipeline de ASR según el idioma elegido.
    lang: "ja" para japonés, "zh" para chino.
    """
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        batch_size = 64
    else:
        device = "cpu"
        dtype = torch.float32
        batch_size = 8

    pipeline = _get_transformers_pipeline()

    if lang == "ja":
        print("[+] Selected language: Japanese (Anime-Whisper)")
        model_name = "litagin/anime-whisper"

        asr = pipeline(
            task="automatic-speech-recognition",
            model=model_name,
            device=device,
            torch_dtype=dtype,
            chunk_length_s=30.0,
            batch_size=batch_size,
        )

    elif lang == "zh":
        print("[+] Selected language: Chinese (BELLE-2/Belle-whisper-large-v3-zh)")
        model_name = "BELLE-2/Belle-whisper-large-v3-zh"

        asr = pipeline(
            task="automatic-speech-recognition",
            model=model_name,
            device=device,
            torch_dtype=dtype,
            chunk_length_s=30.0,
            batch_size=batch_size,
        )

        asr.model.config.forced_decoder_ids = asr.tokenizer.get_decoder_prompt_ids(
            language="zh",
            task="transcribe",
        )

    else:
        raise ValueError(f"Idioma no soportado: {lang}. Usa 'ja' o 'zh'.")

    return asr


def build_romaji_converter():
    """
    Devuelve el objeto kakasi (no el converter legacy),
    para poder usar la API nueva .convert().
    """
    kks = kakasi()
    # Estos setMode siguen funcionando, aunque estén deprecados.
    try:
        kks.setMode("J", "a")  # Kanji → romaji
        kks.setMode("K", "a")  # Katakana → romaji
        kks.setMode("H", "a")  # Hiragana → romaji
    except Exception:
        # Por si en alguna versión futura cambian setMode.
        pass
    return kks

def japanese_to_romaji_pretty(text: str, conv, tagger: Tagger) -> str:
    """
    Convierte una línea de japonés a romaji legible:
    - Usa fugashi para segmentar en "palabras".
    - Usa pykakasi.convert() para romanizar cada palabra.
    - Repara la pequeña っ cuando salta de una palabra a la siguiente.
    - Fusiona patrones frecuentes: "X tte" → "Xtte", "X te iru" → "Xteiru", etc.
    """
    import re

    text = (text or "").strip()
    if not text:
        return ""

    tokens = list(tagger(text))
    romaji_tokens: List[str] = []
    prev_had_small_tsu = False  # si la palabra anterior termina en っ／ッ

    for word in tokens:
        surf = word.surface

        # Intentamos primero la API nueva .convert() (pykakasi >= 2.x)
        r = ""
        try:
            parts = conv.convert(surf)
            # parts es una lista de dicts con claves como "hepburn", "kunrei", etc.
            r = "".join(
                (
                    item.get("hepburn")
                    or item.get("kunrei")
                    or item.get("hira")
                    or item.get("orig")
                    or ""
                )
                if isinstance(item, dict) else str(item)
                for item in parts
            )
        except AttributeError:
            # Fallback: conv es un converter legacy con .do()
            r = conv.do(surf)

        r = (r or "").strip()

        if not r:
            romaji_tokens.append("")
            prev_had_small_tsu = surf.endswith("っ") or surf.endswith("ッ")
            continue

        if prev_had_small_tsu and romaji_tokens:
            # Ajustar la palabra anterior si terminaba en "tsu"/"tu"
            prev = romaji_tokens[-1]

            for suf in ("tsu", "tu"):
                if prev.endswith(suf):
                    prev = prev[:-len(suf)]
                    break

            # Geminar la consonante inicial de la palabra actual (te → tte)
            first = r[0]
            if first.isalpha():
                r = first + r

            romaji_tokens[-1] = prev

        romaji_tokens.append(r)
        prev_had_small_tsu = surf.endswith("っ") or surf.endswith("ッ")

    # Unimos con espacios
    romaji = " ".join(rt for rt in romaji_tokens if rt)

    # Fusiones útiles:
    # X tte → Xtte  (黙って → damatte)
    romaji = re.sub(r"\b([a-z]+)\s+tte\b", r"\1tte", romaji)

    # X te iru → Xteiru (している → shiteiru)
    romaji = re.sub(r"\b([a-z]+)\s+te\s+iru\b", r"\1teiru", romaji)
    # X de iru → Xdeiru
    romaji = re.sub(r"\b([a-z]+)\s+de\s+iru\b", r"\1deiru", romaji)

    # n da yo → ndayo
    romaji = re.sub(r"\bn\s+da\s+yo\b", "ndayo", romaji)

    return romaji

def japanese_to_romaji_line(text: str, conv) -> str:
    """
    Convierte una línea completa de japonés a romaji:
    - usa kakasi.convert para manejar bien っ, 長音, etc.
    - junta todo sin espacios raros (shiteiru, dokidoki...).

    Si la versión de pykakasi es antigua y no soporta convert(),
    cae a conv.do(text).
    """
    import re

    text = (text or "").strip()
    if not text:
        return ""

    # pykakasi >= 2.x
    try:
        parts = conv.convert(text)
    except TypeError:
        # Fallback: vieja API
        return conv.do(text).replace("  ", " ").strip()

    out: List[str] = []
    for item in parts:
        if isinstance(item, dict):
            r = (
                item.get("hepburn")
                or item.get("kana")
                or item.get("hira")
                or item.get("orig")
            )
            if r:
                out.append(r)
        else:
            out.append(str(item))

    # Un solo string, sin espacios internos
    romaji = "".join(out)
    # Limpieza básica de espacios
    romaji = re.sub(r"\s+", "", romaji)
    return romaji


def text_to_pinyin(text: str) -> str:
    """
    Pinyin con tonos en diacríticos (nǐ hǎo), no números.
    """
    syllables = lazy_pinyin(text, style=Style.TONE, errors="ignore")
    return " ".join(syllables).strip()


def _normalize_romanization_output(text: str, lang: str) -> str:
    out = (text or "").strip()
    if not out:
        return ""

    if lang == "ja":
        # Preserve word boundaries for romaji output. We only collapse
        # repeated whitespace so DeepSeek/local romanization keeps readable
        # spacing between particles and words.
        return re.sub(r"\s+", " ", out).strip()

    if lang == "zh":
        out = re.sub(r"\s+", " ", out)
        return out.strip()

    return out


def _romanize_locally(lines: List[str], lang: str) -> List[str]:
    romaji_converter = build_romaji_converter() if lang == "ja" else None
    ja_tagger = _ensure_ja_tagger() if lang == "ja" else None
    out: List[str] = []
    for line in lines:
        romanized = _romanize_source_text(line, lang, romaji_converter, ja_tagger)
        out.append(_normalize_romanization_output(romanized, lang))
    return out


def _build_romanization_system_prompt(lang: str) -> str:
    language_name = "japones" if lang == "ja" else "chino"
    output_name = "romaji" if lang == "ja" else "pinyin"
    return (
        f"Eres un motor de {output_name} para subtitulos en {language_name}.\n"
        "Devuelve SOLO JSON valido, sin explicaciones, sin markdown y sin texto extra.\n"
        "La salida debe tener exactamente la clave \"romanizations\" con el mismo numero "
        "de elementos que la entrada.\n"
        "Reglas:\n"
        f"- Cada elemento de salida corresponde a la linea de entrada con el mismo indice.\n"
        f"- Para {lang == 'ja' and 'japones' or 'chino'} debes conservar numeros, puntuacion y texto latin.\n"
        "- Si una linea empieza por guion de dialogo, conserva el guion.\n"
        "- Conserva los saltos de linea lógicos si la linea contiene \\N.\n"
        "- Deja las lineas vacias como cadenas vacias.\n"
        "- No apliques traduccion ni explicacion; solo romanizacion.\n"
        + (
            "- En japones, devuelve romaji estilo Hepburn con palabras separadas por espacios simples, preferentemente en minusculas.\n"
            "- Separa tambien las particulas y auxiliares como palabras independientes.\n"
            "- Ejemplo: 俺たちのダンジョンが -> oretachi no danjon ga\n"
            "- Ejemplo: 階層主を融合させやがったのか… -> kaisoushu wo yuugou saseyagatta no ka...\n"
            if lang == "ja"
            else "- En chino, devuelve pinyin con tonos en diacriticos y espacios entre silabas.\n"
        )
    )


def _build_romanization_user_prompt(lines: List[str], lang: str) -> str:
    payload = json.dumps({"lines": lines}, ensure_ascii=False)
    return (
        f"Romaniza estas {len(lines)} lineas en {lang}.\n"
        "Entrada JSON:\n"
        f"{payload}\n\n"
        "Devuelve un JSON con esta forma exacta:\n"
        "{\"romanizations\": [\"...\", \"...\"]}\n"
        "No incluyas nada mas."
    )


def _build_romanization_response_format(expected_count: int) -> Dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "subtitle_romanizations",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "romanizations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": expected_count,
                        "maxItems": expected_count,
                    }
                },
                "required": ["romanizations"],
            },
        },
    }


def _log_romanization_count_issue(
    start_line: int,
    end_line: int,
    parse_result: RomanizationParseResult,
    chunk: Optional[List[str]] = None,
) -> None:
    detail = (
        f"[DeepSeek romanization] Bloque {start_line}-{end_line}: "
        f"esperadas {parse_result.expected_count}, recibidas {parse_result.raw_count} "
        f"(parser={parse_result.parser})."
    )
    if parse_result.missing_indices:
        missing_abs = [start_line + idx for idx in parse_result.missing_indices]
        detail += f" Faltan posiciones/lineas: {missing_abs}."
    if parse_result.extra_count:
        detail += f" Sobran {parse_result.extra_count} romanizaciones."
    if parse_result.error:
        detail += f" Error: {parse_result.error}."
    print(detail)

    if parse_result.missing_indices and chunk:
        print("[DeepSeek romanization] Source lines returned without romanization by the model:")
        for idx in parse_result.missing_indices:
            absolute_line = start_line + idx
            source_text = chunk[idx].strip() if idx < len(chunk) else ""
            print(f"  - linea {absolute_line}: {source_text}")


def romanize_with_deepseek(
    src_lines: List[str],
    lang: str,
) -> Tuple[List[str], ApiUsage, Optional[str]]:
    if not src_lines:
        return [], ApiUsage(engine="deepseek", model_name=DEEPSEEK_MODEL), None

    if not DEEPSEEK_API_KEY:
        print("[Romanization] DeepSeek is skipped because DEEPSEEK_API_KEY is missing; using local romanization.")
        return _romanize_locally(src_lines, lang), ApiUsage(engine="deepseek", model_name=DEEPSEEK_MODEL), "missing_key"

    try:
        client = get_deepseek_client()
    except Exception as e:
        print(f"[Romanization] DeepSeek cannot be initialized: {e}. Using local romanization.")
        return _romanize_locally(src_lines, lang), ApiUsage(engine="deepseek", model_name=DEEPSEEK_MODEL), "client_error"

    system_prompt = _build_romanization_system_prompt(lang)
    all_outputs: List[str] = []
    total = len(src_lines)
    usage = ApiUsage(engine="deepseek", model_name=DEEPSEEK_MODEL)
    skipped_reason: Optional[str] = None

    for start in range(0, total, CHUNK_SIZE):
        chunk = src_lines[start:start + CHUNK_SIZE]
        base_user_prompt = _build_romanization_user_prompt(chunk, lang)
        end_line = min(start + CHUNK_SIZE, total)
        print(f"[DeepSeek romanization] Lines {start + 1}-{end_line} of {total}...")

        response = None
        content = ""
        parse_result: Optional[RomanizationParseResult] = None

        for attempt in range(1, MODEL_BLOCK_ATTEMPTS + 1):
            user_prompt = _build_retry_user_prompt(base_user_prompt, len(chunk), attempt)
            try:
                request_kwargs: Dict[str, object] = {
                    "model": DEEPSEEK_MODEL,
                    "temperature": 0.0,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                }

                response = client.chat.completions.create(**request_kwargs)
                content = (response.choices[0].message.content or "").strip()
                parse_result = parse_json_romanizations_result(content, fallback_lines=chunk)
                if parse_result.exact_match:
                    break

                _log_romanization_count_issue(start + 1, end_line, parse_result, chunk)
                if attempt < MODEL_BLOCK_ATTEMPTS:
                    wait_s = min(9, 2 * attempt)
                    print(
                        f"[DeepSeek romanization] Retrying block {start + 1}-{end_line} in {wait_s} s "
                        "porque el numero de romanizaciones no coincide..."
                    )
                    time.sleep(wait_s)
                    continue
                break
            except Exception as e:
                if attempt < MODEL_BLOCK_ATTEMPTS:
                    wait_s = min(9, 2 * attempt)
                    print(
                        f"[DeepSeek romanization] Transient error in block {start + 1}-{end_line}: {e}. "
                        f"Retrying in {wait_s} s..."
                    )
                    time.sleep(wait_s)
                    continue
                print(
                    f"[DeepSeek romanization] Romanization failed; using local romanization for this block. "
                    f"Detalle: {e}"
                )

        if response is None:
            skipped_reason = skipped_reason or "partial_error"
            all_outputs.extend(_romanize_locally(chunk, lang))
            continue

        resp_usage = getattr(response, "usage", None)
        if resp_usage:
            pt = _safe_int(getattr(resp_usage, "prompt_tokens", 0))
            ct = _safe_int(getattr(resp_usage, "completion_tokens", 0))
            usage.prompt_tokens += pt
            usage.completion_tokens += ct
            usage.cost_usd += estimate_cost("deepseek", pt, ct)
        else:
            _warn_missing_usage("deepseek")

        if parse_result is None:
            parse_result = parse_json_romanizations_result(content, fallback_lines=chunk)
        if not parse_result.exact_match:
            skipped_reason = skipped_reason or "partial_error"
            _log_romanization_count_issue(start + 1, end_line, parse_result, chunk)
            all_outputs.extend(_romanize_locally(chunk, lang))
            continue

        all_outputs.extend(
            [_normalize_romanization_output(text, lang) for text in parse_result.romanizations]
        )

    return all_outputs, usage, skipped_reason


def extract_segment(video_path: str, start_ms: int, end_ms: int, out_wav: str, sample_rate: int = 16000):
    start_s = max(0.0, start_ms / 1000.0)
    dur_s = max(0.01, (end_ms - start_ms) / 1000.0)

    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-y",
        "-ss", f"{start_s:.3f}",
        "-i", video_path,
        "-t", f"{dur_s:.3f}",
        "-vn",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", str(sample_rate),
        out_wav,
    ]
    subprocess.run(cmd, check=True)


def transcribe_ass(
    ass_path: str,
    video_path: str,
    pad_ms: int,
    lang: str,
    do_roman_morph: bool,
    romanization_usage: Optional[ApiUsage] = None,
    context_note_usage: Optional[ApiUsage] = None,
    phase_timings: Optional[Dict[str, float]] = None,
) -> pysubs2.SSAFile:
    """
    Carga un .ass, transcribe audio, pule la puntuación con modelos libres
    y opcionalmente añade:
      - romaji/pinyin via DeepSeek
      - nota contextual con Claude Sonnet
    en líneas adicionales (separadas con \\N).
    Además, imprime progreso por línea para que la GUI
    pueda mostrar en qué línea va y el % completado.
    """
    global _ja_tagger

    print("[+] Loading ASS for transcription.")
    subs = pysubs2.load(ass_path, encoding="utf-8")

    events: List[pysubs2.SSAEvent] = []
    audio_paths: List[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        print("[+] Preparing audio segments with ffmpeg.")

        for idx, ev in enumerate(subs):
            if getattr(ev, "is_comment", False):
                continue
            if ev.duration <= 0:
                continue

            start_ms = max(0, ev.start - pad_ms)
            end_ms = ev.end + pad_ms

            seg_path = os.path.join(tmpdir, f"seg_{idx:04d}.wav")

            try:
                extract_segment(video_path, start_ms, end_ms, seg_path)
            except subprocess.CalledProcessError as e:
                print(f"[!] ffmpeg failed on line {idx} ({ev.start}-{ev.end} ms): {e}")
                continue

            events.append(ev)
            audio_paths.append(seg_path)

        if not audio_paths:
            print("[!] No audio segments were generated. Returning the ASS unchanged.")
            return subs

        total = len(events)
        print(f"[+] Prepared segments: {total}")
        print("[+] Loading transcription model (the first run may take a while).")
        asr_started_at = time.time()
        asr = build_asr_pipeline(lang)

        print("[+] Transcribing lines.")
        raw_lines: List[str] = []

        # ASR línea a línea con progreso
        for i, wav_path in enumerate(audio_paths, start=1):
            try:
                res = asr(wav_path)
                if isinstance(res, dict):
                    txt = (res.get("text", "") or "").strip()
                else:
                    txt = str(res).strip()
            except Exception as e:
                print(f"[ASR] Error on line {i}/{total}: {e}")
                txt = ""

            # 🔹 Limpiar repeticiones absurdas tipo 痛たたたたたたた....
            txt = clean_repetitions(txt)

            raw_lines.append(txt)
            snippet = txt.replace("\n", " ")[:60]
            print(f"[Transcription] Line {i}/{total} -> {snippet}")
        _record_phase_time(phase_timings, "asr", time.time() - asr_started_at)

        print("[+] Refining punctuation (free models, no GPT).")
        punctuation_started_at = time.time()
        refined_lines = refine_punctuation_free(raw_lines, lang)
        _record_phase_time(phase_timings, "punctuation", time.time() - punctuation_started_at)

        romanized_lines: List[str] = []
        context_notes: List[str] = []
        if do_roman_morph:
            roman_started_at = time.time()
            romanized_lines, roman_usage, roman_skip = romanize_with_deepseek(refined_lines, lang)
            if romanization_usage is not None:
                merge_api_usage(romanization_usage, roman_usage)
            _record_phase_time(phase_timings, "romanization", time.time() - roman_started_at)
            if roman_skip:
                print(f"[Romanization] Partial fallback reason: {roman_skip}")

            note_started_at = time.time()
            context_notes = build_contextual_notes(refined_lines, lang, usage_accumulator=context_note_usage)
            _record_phase_time(phase_timings, "context_note", time.time() - note_started_at)
            print("[+] Embedding romaji/pinyin and contextual notes into the ASS.")

        embedding_started_at = time.time()
        for i, (ev, text) in enumerate(zip(events, refined_lines), start=1):
            base_text = (text or "").strip()
            if not base_text:
                continue

            lines = [base_text]

            if do_roman_morph:
                romanized = romanized_lines[i - 1] if i - 1 < len(romanized_lines) else ""
                if romanized:
                    lines.append("{" + _ass_sanitize_braces(romanized) + "}")

                context_note = context_notes[i - 1] if i - 1 < len(context_notes) else ""
                if context_note:
                    lines.append(_ass_hide(context_note))

            ev.text = "\\N".join(lines)
            snippet = base_text.replace("\n", " ")[:60]
            print(f"[ASS] Line {i}/{total} -> {snippet}")
        _record_phase_time(phase_timings, "embedding", time.time() - embedding_started_at)

    print("[+] Transcription completed.")
    return subs


def add_roman_morph_to_subs(
    subs: pysubs2.SSAFile,
    lang: str,
    romanization_usage: Optional[ApiUsage] = None,
    context_note_usage: Optional[ApiUsage] = None,
    phase_timings: Optional[Dict[str, float]] = None,
) -> pysubs2.SSAFile:
    """
    Añade romaji/pinyin + nota contextual a un ASS que YA tiene el guion
    (japonés o chino) en la primera línea de cada diálogo.
    - NO exige que las líneas tengan duración > 0.
    - Trabaja sobre cualquier línea Dialogue con texto no vacío.
    - Respeta líneas adicionales ya existentes (por ejemplo, traducciones).
    """

    # Seleccionamos TODAS las líneas de diálogo con texto no vacío,
    # sin importar tiempos ni estilos.
    events = [
        ev for ev in subs
        if getattr(ev, "type", "") == "Dialogue"
        and (getattr(ev, "text", "") or "").strip()
    ]

    total = len(events)
    if total == 0:
        print("[Romaji/Pinyin] No dialogue lines to process.")
        return subs

    print(f"[Romaji/Pinyin] There are {total} dialogue lines to process.")

    base_texts = [_event_source_text(ev) for ev in events]
    roman_started_at = time.time()
    romanized_lines, roman_usage, roman_skip = romanize_with_deepseek(base_texts, lang)
    if romanization_usage is not None:
        merge_api_usage(romanization_usage, roman_usage)
    _record_phase_time(phase_timings, "romanization", time.time() - roman_started_at)
    if roman_skip:
                print(f"[Romanization] Partial fallback reason: {roman_skip}")

    note_started_at = time.time()
    context_notes = build_contextual_notes(base_texts, lang, usage_accumulator=context_note_usage)
    _record_phase_time(phase_timings, "context_note", time.time() - note_started_at)
    print("[+] Embedding romaji/pinyin and contextual notes into the ASS.")

    embedding_started_at = time.time()
    for i, ev in enumerate(events, start=1):
        original_lines, extra_lines = _split_event_original_and_extra(ev)

        # Primera línea = texto base en JA/ZH
        base_text = "\\N".join(line.strip() for line in original_lines if line.strip()).strip()
        if not base_text:
            continue

        # Líneas extra ya existentes (por si ya tenías traducción debajo)

        # Reconstruimos las líneas del evento
        lines = original_lines
        romanized = romanized_lines[i - 1] if i - 1 < len(romanized_lines) else ""
        if romanized:
            lines.append("{" + _ass_sanitize_braces(romanized) + "}")

        context_note = context_notes[i - 1] if i - 1 < len(context_notes) else ""
        if context_note:
            lines.append(_ass_hide(context_note))

        lines.extend(extra_lines)

        ev.text = "\\N".join(lines)

        snippet = base_text.replace("\n", " ")[:60]
        print(f"[ASS] Line {i}/{total} -> {snippet}")
    _record_phase_time(phase_timings, "embedding", time.time() - embedding_started_at)

    return subs

# ============================================================
#  PROMPTS Y PREGUNTAS (solo CLI)
# ============================================================

def ask_language() -> str:
    while True:
        print("Choose the transcription language:")
        print("  [j] Japanese")
        print("  [c] Chinese (Mandarin)")
        choice = input("Choice (j/c): ").strip().lower()

        if choice in ("j", "ja", "jp", "japones", "japonés"):
            return "ja"
        if choice in ("c", "zh", "ch", "chino", "mandarin", "mandarín"):
            return "zh"

        print("Invalid input. Please type 'j' or 'c'.\n")


def ask_series_name() -> str:
    series = input("What series is this? (e.g. Dragon Raja): ").strip()
    if not series:
        series = "esta serie"
    return series


def ask_source_type() -> str:
    print("Does the series have source material?")
    print("  [1] Manga")
    print("  [2] Manhwa")
    print("  [3] Light novel")
    print("  [4] None / I don't know")
    while True:
        choice = input("Choice (1/2/3/4): ").strip()
        if choice == "1":
            return "Manga"
        if choice == "2":
            return "Manhwa"
        if choice == "3":
            return "Light novel"
        if choice == "4":
            return "None"
        print("Invalid input. Type 1, 2, 3, or 4.\n")


def build_system_prompt(lang: str, series_name: str, source_type: str) -> str:
    if lang == "ja":
        src_lang = "japonés"
    elif lang == "zh":
        src_lang = "chino mandarín"
    else:
        src_lang = "japonés o chino mandarín"

    source_sentence = describe_source_type(source_type)

    return (
        f"Eres un traductor profesional del {src_lang} al español de España, "
        "especializado en anime y donghua, y en subtitulación profesional.\n\n"
        f"Estás traduciendo la serie «{series_name}».\n"
        f"{source_sentence}\n\n"
        "Instrucciones de subtitulación:\n"
        "- Las líneas ya están segmentadas como subtítulos; NO las fusiones ni las "
        "dividas. Cada línea de origen debe corresponder exactamente a una línea traducida.\n"
        "- Respeta escrupulosamente el formato ASS: conserva tal cual cualquier código "
        "de estilo o posición (p. ej. {\\i1}, {\\b1}, {\\an8}, {\\c&HFFFFFF&}, \\N, etc.). "
        "No los traduzcas, no los borres, no los muevas; solo traduce el texto natural "
        "alrededor de ellos.\n"
        "- Traduce al español de España, registro oral natural, evitando calcos raros, "
        "pero sin perder información ni matices importantes.\n"
        "- Mantén nombres propios y terminología coherentes entre episodios; cuando exista "
        "una versión oficial del material original, intenta aproximarte a su terminología "
        "sin sacrificar naturalidad.\n\n"
        "Salida:\n"
        "- Para cada lote de N líneas, debes devolver EXCLUSIVAMENTE un JSON con el "
        "siguiente formato:\n"
        "  {\"translations\": [\"traducción de la línea 1\", \"traducción de la línea 2\", ...]}\n"
        "- El array \"translations\" debe tener exactamente el mismo número de entradas "
        "que líneas se te han dado, en el mismo orden.\n"
        "- No añadas ningún otro texto fuera del JSON (ni explicaciones, ni comentarios, "
        "ni formato extra).\n"
    )


def build_user_prompt(chunk_lines: List[str], lang: str, series_name: str, source_type: str) -> str:
    if lang == "ja":
        src_lang = "japonés"
    elif lang == "zh":
        src_lang = "chino mandarín"
    else:
        src_lang = "japonés o chino mandarín"

    lines_str = "\n".join(
        f"{i+1}: {text}" for i, text in enumerate(chunk_lines)
    )
    user_prompt = (
        f"Estás traduciendo subtítulos de la serie «{series_name}».\n"
        f"El idioma original es {src_lang}.\n\n"
        "Devuelve EXCLUSIVAMENTE un JSON con esta forma:\n"
        "{\"translations\": [\"traducción de la línea 1\", \"traducción de la línea 2\", ...]}\n"
        "sin texto adicional.\n\n"
        "Líneas a traducir:\n"
        f"{lines_str}"
    )
    return user_prompt


def get_openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("Falta OPENAI_API_KEY (env o api_keys en config.local.json).")
    return OpenAI(
        api_key=OPENAI_API_KEY,
        timeout=OPENAI_TIMEOUT_S,
        max_retries=OPENAI_MAX_RETRIES,
    )


def analyze_contextual_note_with_claude(
    client: anthropic.Anthropic,
    lines: List[str],
    index: int,
    lang: str,
) -> Tuple[str, ApiUsage]:
    prompt = build_contextual_explanation_prompt(lang, lines, index)
    message = client.messages.create(
        model=CONTEXT_NOTE_MODEL,
        max_tokens=CONTEXT_NOTE_MAX_TOKENS,
        system=(
            "Sigue exactamente el formato solicitado y devuelve solo la nota pedida, "
            "sin encabezados ni texto extra."
        ),
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    content = "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    ).strip()
    if not content:
        raise RuntimeError("Claude no devolvió texto para la nota contextual.")

    usage = ApiUsage(engine="context_note", model_name=CONTEXT_NOTE_MODEL)
    msg_usage = getattr(message, "usage", None)
    if msg_usage:
        pt = _safe_int(getattr(msg_usage, "input_tokens", 0))
        ct = _safe_int(getattr(msg_usage, "output_tokens", 0))
        usage.prompt_tokens += pt
        usage.completion_tokens += ct
        usage.cost_usd += estimate_cost("context_note", pt, ct)
    else:
        _warn_missing_usage("context_note")
    note = parse_contextual_explanation_response(content)
    if contains_japanese_script(note):
        repair_prompt = build_contextual_explanation_repair_prompt(lang, lines, index, note)
        repair_message = client.messages.create(
            model=CONTEXT_NOTE_MODEL,
            max_tokens=CONTEXT_NOTE_MAX_TOKENS,
            system=(
                "Sigue exactamente el formato solicitado y devuelve solo la nota pedida, "
                "sin encabezados ni texto extra."
            ),
            messages=[
                {
                    "role": "user",
                    "content": repair_prompt,
                }
            ],
        )
        repair_content = "".join(
            block.text for block in repair_message.content if getattr(block, "type", None) == "text"
        ).strip()
        if not repair_content:
            raise RuntimeError("Claude no devolvió texto en la corrección de la nota contextual.")

        repair_usage = getattr(repair_message, "usage", None)
        if repair_usage:
            pt = _safe_int(getattr(repair_usage, "input_tokens", 0))
            ct = _safe_int(getattr(repair_usage, "output_tokens", 0))
            usage.prompt_tokens += pt
            usage.completion_tokens += ct
            usage.cost_usd += estimate_cost("context_note", pt, ct)
        else:
            _warn_missing_usage("context_note")

        repaired_note = parse_contextual_explanation_response(repair_content)
        if contains_japanese_script(repaired_note):
            if lang == "ja":
                repaired_note = (
                    "La linea depende del contexto y usa un matiz expresivo propio del japonés."
                )
            else:
                repaired_note = "La linea depende del contexto y tiene un matiz propio del chino."
        note = repaired_note or note
    return note, usage


def build_contextual_notes(
    lines: List[str],
    lang: str,
    usage_accumulator: Optional[ApiUsage] = None,
) -> List[str]:
    global _WARNED_CONTEXT_NOTE_MISSING_KEY, _WARNED_CONTEXT_NOTE_CLIENT

    cleaned_lines = [(line or "").strip() for line in lines]
    if not cleaned_lines:
        return []

    if lang not in {"ja", "zh"}:
        return [""] * len(cleaned_lines)

    if not ANTHROPIC_API_KEY:
        if not _WARNED_CONTEXT_NOTE_MISSING_KEY:
            print("[Context note] Skipping because ANTHROPIC_API_KEY is missing.")
            _WARNED_CONTEXT_NOTE_MISSING_KEY = True
        return [""] * len(cleaned_lines)

    try:
        client = get_claude_client()
    except Exception as e:
        if not _WARNED_CONTEXT_NOTE_CLIENT:
            print(f"[Context note] Claude cannot be initialized: {e}")
            _WARNED_CONTEXT_NOTE_CLIENT = True
        return [""] * len(cleaned_lines)

    total = len(cleaned_lines)
    notes: List[str] = []
    lang_label = "JA" if lang == "ja" else "ZH"

    for idx, line in enumerate(cleaned_lines, start=1):
        if not line:
            notes.append("")
            continue

        print(f"[Context note {lang_label}] Line {idx}/{total}...")
        try:
            note, note_usage = analyze_contextual_note_with_claude(client, cleaned_lines, idx - 1, lang)
            if usage_accumulator is not None:
                merge_api_usage(usage_accumulator, note_usage)
        except Exception as e:
            print(f"[Context note {lang_label}] Error on line {idx}: {e}")
            note = ""
        notes.append(note)

    return notes


def get_deepseek_client() -> OpenAI:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Falta DEEPSEEK_API_KEY. Define la variable de entorno.")
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


def get_claude_client() -> anthropic.Anthropic:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("Falta ANTHROPIC_API_KEY. Define la variable de entorno.")
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def get_gemini_model(lang: str, series_name: str, source_type: str):
    if not GEMINI_API_KEY:
        raise RuntimeError("Falta GEMINI_API_KEY. Define la variable de entorno.")
    system_prompt = build_system_prompt(lang, series_name, source_type)
    if google_genai is not None:
        client = google_genai.Client(api_key=GEMINI_API_KEY)
        return "google-genai", client, system_prompt
    if legacy_genai is not None:
        legacy_genai.configure(api_key=GEMINI_API_KEY)
        return "google-generativeai", legacy_genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=system_prompt,
        ), system_prompt
    raise RuntimeError("No Gemini SDK available. Install google-genai or google-generativeai.")


# ============================================================
#  TRADUCCIÓN POR MODELO
# ============================================================

def translate_with_openai(
    src_lines: List[str],
    lang: str,
    series_name: str,
    source_type: str,
    debug_dir: Optional[str] = None,
) -> Tuple[List[str], ApiUsage, Optional[str]]:
    if not OPENAI_API_KEY:
        print("[GPT-5.5] GPT is skipped because OPENAI_API_KEY is missing (env or config.local.json).")
        return src_lines, ApiUsage(engine="gpt", model_name=OPENAI_MODEL), "missing_key"

    try:
        client = get_openai_client()
    except Exception as e:
        print(f"[GPT-5.5] The client cannot be initialized: {e}. GPT will be skipped.")
        return src_lines, ApiUsage(engine="gpt", model_name=OPENAI_MODEL), "client_error"

    system_prompt = build_system_prompt(lang, series_name, source_type)
    all_translations: List[str] = []
    total = len(src_lines)
    usage = ApiUsage(engine="gpt", model_name=OPENAI_MODEL)
    skipped_reason: Optional[str] = None

    for start in range(0, total, CHUNK_SIZE):
        chunk = src_lines[start:start + CHUNK_SIZE]
        base_user_prompt = build_user_prompt(chunk, lang, series_name, source_type)
        end_line = min(start + CHUNK_SIZE, total)
        print(f"[{DISPLAY_NAMES['gpt']}] Lines {start + 1}-{end_line} of {total}...")

        response = None
        content = ""
        parse_result: Optional[TranslationParseResult] = None
        use_response_format = True

        for attempt in range(1, OPENAI_BLOCK_ATTEMPTS + 1):
            block_started_at = time.time()
            user_prompt = _build_retry_user_prompt(base_user_prompt, len(chunk), attempt)
            print(
                f"[GPT-5.5] Solicitud bloque {start + 1}-{end_line}, "
                f"intento {attempt}/{OPENAI_BLOCK_ATTEMPTS}..."
            )
            try:
                request_kwargs: Dict[str, object] = {
                    "model": OPENAI_MODEL,
                    "timeout": OPENAI_TIMEOUT_S,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                }
                _add_openai_temperature(request_kwargs, 0.1)
                if use_response_format:
                    request_kwargs["response_format"] = _build_translation_response_format(len(chunk))
                response = client.chat.completions.create(**request_kwargs)
                content = (response.choices[0].message.content or "").strip()
                elapsed = time.time() - block_started_at
                print(
                    f"[GPT-5.5] Bloque {start + 1}-{end_line} completado "
                    f"en {elapsed:.1f} s."
                )
                parse_result = parse_json_translations_result(content, fallback_lines=chunk)
                if parse_result.exact_match:
                    break

                _log_translation_count_issue("gpt", start + 1, end_line, parse_result, chunk)
                _save_translation_debug_response(
                    debug_dir,
                    "gpt",
                    start + 1,
                    end_line,
                    attempt,
                    "count_mismatch",
                    chunk,
                    content,
                    parse_result=parse_result,
                )
                if attempt < OPENAI_BLOCK_ATTEMPTS:
                    wait_s = min(12, 3 * attempt)
                    print(
                        f"[GPT-5.5] Retrying block {start + 1}-{end_line} in {wait_s} s "
                        "because the number of translations does not match..."
                    )
                    time.sleep(wait_s)
                    continue
                break
            except Exception as e:
                elapsed = time.time() - block_started_at
                err_text = str(e)
                if use_response_format and "response_format" in err_text.lower():
                    use_response_format = False
                    print(
                        f"[GPT-5.5] The model/SDK rejected response_format for block "
                        f"{start + 1}-{end_line}; se reintenta sin JSON schema."
                    )
                    if attempt < OPENAI_BLOCK_ATTEMPTS:
                        continue
                if attempt < OPENAI_BLOCK_ATTEMPTS:
                    wait_s = min(12, 3 * attempt)
                    print(
                        f"[GPT-5.5] Error/transitorio o timeout en bloque "
                        f"{start + 1}-{end_line} tras {elapsed:.1f} s: {e}. "
                        f"Reintentando en {wait_s} s..."
                    )
                    time.sleep(wait_s)
                else:
                    print(
                        f"[GPT-5.5] Error al traducir; se omite GPT en este bloque. "
                        f"Detalle: {e}"
                    )

        if response is None:
            translations = chunk
            all_translations.extend(translations)
            skipped_reason = skipped_reason or "partial_error"
            continue

        resp_usage = getattr(response, "usage", None)
        if resp_usage:
            pt = _safe_int(getattr(resp_usage, "prompt_tokens", 0))
            ct = _safe_int(getattr(resp_usage, "completion_tokens", 0))
            usage.prompt_tokens += pt
            usage.completion_tokens += ct
            usage.cost_usd += estimate_cost("gpt", pt, ct)
        else:
            _warn_missing_usage("gpt")
        if parse_result is None:
            parse_result = parse_json_translations_result(content, fallback_lines=chunk)
        if not parse_result.exact_match:
            skipped_reason = skipped_reason or "partial_error"
            _log_translation_count_issue("gpt", start + 1, end_line, parse_result, chunk)
            _save_translation_debug_response(
                debug_dir,
                "gpt",
                start + 1,
                end_line,
                OPENAI_BLOCK_ATTEMPTS,
                "final_mismatch",
                chunk,
                content,
                parse_result=parse_result,
            )
        all_translations.extend(parse_result.translations)

    return all_translations, usage, skipped_reason


def translate_with_deepseek(
    src_lines: List[str],
    lang: str,
    series_name: str,
    source_type: str,
    debug_dir: Optional[str] = None,
) -> Tuple[List[str], ApiUsage, Optional[str]]:
    try:
        client = get_deepseek_client()
    except Exception as e:
        print(f"[DeepSeek] DeepSeek is skipped (client not initialized): {e}")
        return src_lines, ApiUsage(engine="deepseek", model_name=DEEPSEEK_MODEL), "client_error"
    system_prompt = build_system_prompt(lang, series_name, source_type)
    all_translations: List[str] = []
    total = len(src_lines)
    usage = ApiUsage(engine="deepseek", model_name=DEEPSEEK_MODEL)
    skipped_reason: Optional[str] = None

    for start in range(0, total, CHUNK_SIZE):
        chunk = src_lines[start:start + CHUNK_SIZE]
        base_user_prompt = build_user_prompt(chunk, lang, series_name, source_type)
        end_line = min(start + CHUNK_SIZE, total)
        print(f"[DeepSeek] Lines {start + 1}-{end_line} of {total}...")

        response = None
        content = ""
        parse_result: Optional[TranslationParseResult] = None

        for attempt in range(1, MODEL_BLOCK_ATTEMPTS + 1):
            user_prompt = _build_retry_user_prompt(base_user_prompt, len(chunk), attempt)
            try:
                response = client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    temperature=0.1,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = (response.choices[0].message.content or "").strip()
                parse_result = parse_json_translations_result(content, fallback_lines=chunk)
                if parse_result.exact_match:
                    break

                _log_translation_count_issue("deepseek", start + 1, end_line, parse_result, chunk)
                _save_translation_debug_response(
                    debug_dir,
                    "deepseek",
                    start + 1,
                    end_line,
                    attempt,
                    "count_mismatch",
                    chunk,
                    content,
                    parse_result=parse_result,
                )
                if attempt < MODEL_BLOCK_ATTEMPTS:
                    wait_s = min(9, 2 * attempt)
                    print(f"[DeepSeek] Retrying block {start + 1}-{end_line} in {wait_s} s...")
                    time.sleep(wait_s)
                    continue
                break
            except Exception as e:
                if attempt < MODEL_BLOCK_ATTEMPTS:
                    wait_s = min(9, 2 * attempt)
                    print(
                        f"[DeepSeek] Error in block {start + 1}-{end_line}: {e}. "
                        f"Retrying in {wait_s} s..."
                    )
                    time.sleep(wait_s)
                    continue
                print(f"[DeepSeek] Translation failed; DeepSeek is skipped for this block. Details: {e}")

        if response is None:
            skipped_reason = skipped_reason or "partial_error"
            all_translations.extend(chunk)
            continue

        resp_usage = getattr(response, "usage", None)
        if resp_usage:
            pt = _safe_int(getattr(resp_usage, "prompt_tokens", 0))
            ct = _safe_int(getattr(resp_usage, "completion_tokens", 0))
            usage.prompt_tokens += pt
            usage.completion_tokens += ct
            usage.cost_usd += estimate_cost("deepseek", pt, ct)
        else:
            _warn_missing_usage("deepseek")
        if parse_result is None:
            parse_result = parse_json_translations_result(content, fallback_lines=chunk)
        if not parse_result.exact_match:
            skipped_reason = skipped_reason or "partial_error"
            _log_translation_count_issue("deepseek", start + 1, end_line, parse_result, chunk)
            _save_translation_debug_response(
                debug_dir,
                "deepseek",
                start + 1,
                end_line,
                MODEL_BLOCK_ATTEMPTS,
                "final_mismatch",
                chunk,
                content,
                parse_result=parse_result,
            )
        all_translations.extend(parse_result.translations)

    return all_translations, usage, skipped_reason


def translate_with_claude(
    src_lines: List[str],
    lang: str,
    series_name: str,
    source_type: str,
    debug_dir: Optional[str] = None,
) -> Tuple[List[str], ApiUsage, Optional[str]]:
    try:
        client = get_claude_client()
    except Exception as e:
        print(f"[Claude] Claude is skipped (client not initialized): {e}")
        return src_lines, ApiUsage(engine="claude", model_name=CLAUDE_MODEL), "client_error"

    system_prompt = build_system_prompt(lang, series_name, source_type)
    all_translations: List[str] = []
    total = len(src_lines)
    usage = ApiUsage(engine="claude", model_name=CLAUDE_MODEL)
    skipped_reason: Optional[str] = None

    for start in range(0, total, CHUNK_SIZE):
        chunk = src_lines[start:start + CHUNK_SIZE]
        base_user_prompt = build_user_prompt(chunk, lang, series_name, source_type)
        end_line = min(start + CHUNK_SIZE, total)
        print(f"[Claude] Lines {start + 1}-{end_line} of {total}...")

        message = None
        content = ""
        parse_result: Optional[TranslationParseResult] = None

        for attempt in range(1, MODEL_BLOCK_ATTEMPTS + 1):
            user_prompt = _build_retry_user_prompt(base_user_prompt, len(chunk), attempt)
            try:
                request_kwargs = {
                    "model": CLAUDE_MODEL,
                    "max_tokens": 4096,
                    "system": system_prompt,
                    "messages": [
                        {
                            "role": "user",
                            "content": user_prompt,
                        }
                    ],
                }
                if CLAUDE_MODEL != "claude-opus-4-7":
                    request_kwargs["temperature"] = 0.1
                message = client.messages.create(**request_kwargs)
            except Exception as e:
                if attempt < MODEL_BLOCK_ATTEMPTS:
                    wait_s = min(9, 2 * attempt)
                    print(
                        f"[Claude] Error in block {start + 1}-{end_line}: {e}. "
                        f"Retrying in {wait_s} s..."
                    )
                    time.sleep(wait_s)
                    continue
                print(f"[Claude] Translation failed; Claude is skipped for this block. Details: {e}")
                break

            content = "".join(
                block.text for block in message.content if getattr(block, "type", None) == "text"
            ).strip()
            if not content:
                if attempt < MODEL_BLOCK_ATTEMPTS:
                    wait_s = min(9, 2 * attempt)
                    print(
                        f"[Claude] Respuesta vacía en bloque {start + 1}-{end_line}. "
                        f"Reintentando en {wait_s} s..."
                    )
                    time.sleep(wait_s)
                    continue
                print("[Claude] Empty response, returning the original lines for this block.")
                break

            parse_result = parse_json_translations_result(content, fallback_lines=chunk)
            if parse_result.exact_match:
                break

            _log_translation_count_issue("claude", start + 1, end_line, parse_result, chunk)
            _save_translation_debug_response(
                debug_dir,
                "claude",
                start + 1,
                end_line,
                attempt,
                "count_mismatch",
                chunk,
                content,
                parse_result=parse_result,
            )
            if attempt < MODEL_BLOCK_ATTEMPTS:
                wait_s = min(9, 2 * attempt)
                print(f"[Claude] Retrying block {start + 1}-{end_line} in {wait_s} s...")
                time.sleep(wait_s)
                continue
            break

        if message is None or not content:
            skipped_reason = skipped_reason or "partial_error"
            all_translations.extend(chunk)
            continue

        msg_usage = getattr(message, "usage", None)
        if msg_usage:
            pt = _safe_int(getattr(msg_usage, "input_tokens", 0))
            ct = _safe_int(getattr(msg_usage, "output_tokens", 0))
            usage.prompt_tokens += pt
            usage.completion_tokens += ct
            usage.cost_usd += estimate_cost("claude", pt, ct)
        else:
            _warn_missing_usage("claude")

        if parse_result is None:
            parse_result = parse_json_translations_result(content, fallback_lines=chunk)
        if not parse_result.exact_match:
            skipped_reason = skipped_reason or "partial_error"
            _log_translation_count_issue("claude", start + 1, end_line, parse_result, chunk)
            _save_translation_debug_response(
                debug_dir,
                "claude",
                start + 1,
                end_line,
                MODEL_BLOCK_ATTEMPTS,
                "final_mismatch",
                chunk,
                content,
                parse_result=parse_result,
            )
        all_translations.extend(parse_result.translations)

    return all_translations, usage, skipped_reason


def translate_with_gemini(
    src_lines: List[str],
    lang: str,
    series_name: str,
    source_type: str,
    debug_dir: Optional[str] = None,
) -> Tuple[List[str], ApiUsage, Optional[str]]:
    """
    Usa Gemini 3 Flash, con bloques más pequeños y max_output_tokens
    limitado para ir algo más rápido/estable.
    """
    try:
        gemini_sdk, model, system_prompt = get_gemini_model(lang, series_name, source_type)
    except Exception as e:
        print(f"[Gemini 3 Flash] Gemini is skipped (client not initialized): {e}")
        return src_lines, ApiUsage(engine="gemini", model_name=GEMINI_MODEL), "client_error"
    all_translations: List[str] = []
    total = len(src_lines)
    usage = ApiUsage(engine="gemini", model_name=GEMINI_MODEL)
    skipped_reason: Optional[str] = None

    for start in range(0, total, GEMINI_CHUNK):
        chunk = src_lines[start:start + GEMINI_CHUNK]
        base_user_prompt = build_user_prompt(chunk, lang, series_name, source_type)
        end_line = min(start + GEMINI_CHUNK, total)
        print(f"[Gemini 3 Flash] Lines {start + 1}-{end_line} of {total}...")

        response = None
        raw = ""
        parse_result: Optional[TranslationParseResult] = None

        for attempt in range(1, MODEL_BLOCK_ATTEMPTS + 1):
            user_prompt = _build_retry_user_prompt(base_user_prompt, len(chunk), attempt)
            try:
                if gemini_sdk == "google-genai":
                    response = model.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=user_prompt,
                        config=google_genai_types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=0.1,
                            max_output_tokens=2000,
                        ),
                    )
                else:
                    response = model.generate_content(
                        user_prompt,
                        generation_config={
                            "temperature": 0.1,
                            "max_output_tokens": 2000,
                        },
                    )
            except Exception as e:
                if attempt < MODEL_BLOCK_ATTEMPTS:
                    wait_s = min(9, 2 * attempt)
                    print(
                        f"[Gemini] API error on lines {start + 1}-{end_line}: {e}. "
                        f"Retrying in {wait_s} s..."
                    )
                    time.sleep(wait_s)
                    continue
                print(f"[Gemini] API error on lines {start + 1}-{end_line}: {e}. "
                      "The original lines are returned for this block.")
                break

            cand = response.candidates[0] if getattr(response, "candidates", None) else None
            if cand is not None:
                print(f"[Gemini DEBUG] finish_reason={getattr(cand, 'finish_reason', None)}")
                print(f"[Gemini DEBUG] safety_ratings={getattr(cand, 'safety_ratings', None)}")

            if not cand or not getattr(cand, "content", None) or not getattr(cand.content, "parts", None):
                finish_reason = getattr(cand, "finish_reason", None) if cand else None
                safety = getattr(cand, "safety_ratings", None) if cand else None
                if attempt < MODEL_BLOCK_ATTEMPTS:
                    wait_s = min(9, 2 * attempt)
                    print(
                        f"[Gemini] Empty or blocked response (finish_reason={finish_reason}). "
                        f"Retrying in {wait_s} s..."
                    )
                    print(f"[Gemini DEBUG] safety_ratings={safety}")
                    time.sleep(wait_s)
                    continue
                print(f"[Gemini] Empty or blocked response (finish_reason={finish_reason}). "
                      "The original lines are returned for this block.")
                print(f"[Gemini DEBUG] safety_ratings={safety}")
                break

            text_parts = [
                getattr(part, "text", "")
                for part in cand.content.parts
                if getattr(part, "text", "")
            ]
            raw = "".join(text_parts).strip()
            if not raw:
                finish_reason = getattr(cand, "finish_reason", None)
                safety = getattr(cand, "safety_ratings", None)
                if attempt < MODEL_BLOCK_ATTEMPTS:
                    wait_s = min(9, 2 * attempt)
                    print(
                        f"[Gemini] Sin texto utilizable (finish_reason={finish_reason}). "
                        f"Reintentando en {wait_s} s..."
                    )
                    print(f"[Gemini DEBUG] safety_ratings={safety}")
                    time.sleep(wait_s)
                    continue
                print(f"[Gemini] Sin texto utilizable (finish_reason={finish_reason}). "
                      "Se devuelven las líneas originales para este bloque.")
                print(f"[Gemini DEBUG] safety_ratings={safety}")
                break

            parse_result = parse_json_translations_result(raw, chunk)
            if parse_result.exact_match:
                break

            _log_translation_count_issue("gemini", start + 1, end_line, parse_result, chunk)
            _save_translation_debug_response(
                debug_dir,
                "gemini",
                start + 1,
                end_line,
                attempt,
                "count_mismatch",
                chunk,
                raw,
                parse_result=parse_result,
                extra={
                    "finish_reason": getattr(cand, "finish_reason", None),
                    "safety_ratings": getattr(cand, "safety_ratings", None),
                },
            )
            if attempt < MODEL_BLOCK_ATTEMPTS:
                wait_s = min(9, 2 * attempt)
                print(f"[Gemini] Retrying block {start + 1}-{end_line} in {wait_s} s...")
                time.sleep(wait_s)
                continue
            break

        if response is None:
            skipped_reason = skipped_reason or "partial_error"
            all_translations.extend(chunk)
            continue

        cand = response.candidates[0] if getattr(response, "candidates", None) else None
        usage_md = getattr(response, "usage_metadata", None)
        if usage_md is None and cand is not None and getattr(cand, "usage_metadata", None):
            usage_md = cand.usage_metadata
        if usage_md:
            pt = _safe_int(getattr(usage_md, "prompt_token_count", 0))
            ct = _safe_int(getattr(usage_md, "candidates_token_count", 0))
            usage.prompt_tokens += pt
            usage.completion_tokens += ct
            usage.cost_usd += estimate_cost("gemini", pt, ct)
        else:
            _warn_missing_usage("gemini")

        if parse_result is None:
            if raw:
                parse_result = parse_json_translations_result(raw, chunk)
            else:
                parse_result = TranslationParseResult(
                    translations=list(chunk),
                    expected_count=len(chunk),
                    raw_count=0,
                    parser="fallback",
                    exact_match=False,
                    normalized=True,
                    used_fallback=True,
                    error="empty_response",
                )
        if not parse_result.exact_match:
            skipped_reason = skipped_reason or "partial_error"
            _log_translation_count_issue("gemini", start + 1, end_line, parse_result, chunk)
            _save_translation_debug_response(
                debug_dir,
                "gemini",
                start + 1,
                end_line,
                MODEL_BLOCK_ATTEMPTS,
                "final_mismatch",
                chunk,
                raw,
                parse_result=parse_result,
                extra={
                    "finish_reason": getattr(cand, "finish_reason", None) if cand else None,
                    "safety_ratings": getattr(cand, "safety_ratings", None) if cand else None,
                },
            )
        all_translations.extend(parse_result.translations)

    return all_translations, usage, skipped_reason


# ============================================================
#  APLICAR TRADUCCIONES + HTML
# ============================================================

def apply_translations_and_save_subs(
    base_subs: pysubs2.SSAFile,
    translations: List[str],
    output_path: str,
):
    subs_out = copy.deepcopy(base_subs)
    events_out = [ev for ev in subs_out if not getattr(ev, "is_comment", False)]

    if len(translations) != len(events_out):
        print("[WARNING] The number of translations differs from the number of lines; the minimum common length will be used.")
    n = min(len(translations), len(events_out))

    for i in range(n):
        ev = events_out[i]
        trans = translations[i].strip()
        if not trans:
            continue

        ev.text = trans

    subs_out.save(output_path, encoding="utf-8-sig")
    print(f"Saved: {output_path}")

def format_morph_cell_html(morph: str) -> str:
    """
    Recibe una cadena del tipo:
      '已经 (d) -> already; ... | 能 (v) -> ...'
    y la convierte en HTML:
      <b>已经 (d)</b> -> already; ...<br>
      <b>能 (v)</b> -> ...<br>
    """
    if "->" not in morph:
        return html.escape(morph).replace("\n", "<br>")

    parts = [p.strip() for p in morph.split("|") if p.strip()]
    if not parts:
        return ""

    html_parts: List[str] = []

    for part in parts:
        # Intentamos separar 'cabeza (pos)' y 'definición'
        m = re.match(r"^(.*?\))\s*->\s*(.*)$", part)
        if m:
            head = m.group(1).strip()   # '已经 (d)'
            gloss = m.group(2).strip()  # 'already; ...'
            html_parts.append(
                "<b>" + html.escape(head) + "</b> -> " +
                html.escape(gloss) + "<br>"
            )
        else:
            # Si no encaja el patrón, lo metemos tal cual con salto de línea
            html_parts.append(html.escape(part) + "<br>")

    return "".join(html_parts)

def generate_html(
    subs: pysubs2.SSAFile,
    translations_by_model: Dict[str, List[str]],
    output_path: str,
):
    """
    Genera un HTML con columnas:
    Texto original | Romaji/Pinyin | Nota contextual | GPT | Claude | Gemini | DeepSeek
    Todas las columnas con el mismo ancho.
    """
    events = [ev for ev in subs if not getattr(ev, "is_comment", False)]

    gpt = translations_by_model.get("gpt", [])
    claude = translations_by_model.get("claude", [])
    gemini = translations_by_model.get("gemini", [])
    deepseek = translations_by_model.get("deepseek", [])

    def safe_get(lst: List[str], idx: int) -> str:
        return lst[idx].strip() if idx < len(lst) else ""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("<!DOCTYPE html><html><head><meta charset='utf-8'>")
        f.write("<title>Resumen de subtítulos</title>")
        f.write(
            "<style>"
            "body{font-family:Segoe UI,Arial,sans-serif;font-size:13px;}"
            "table{border-collapse:collapse;width:100%;table-layout:fixed;}"
            "th,td{border:1px solid #ccc;padding:4px;vertical-align:top;"
            "width:14%;word-wrap:break-word;overflow-wrap:break-word;}"
            "th{background:#f0f0f0;}"
            "tr:nth-child(even){background:#fafafa;}"
            "</style>"
        )
        f.write("</head><body>")
        f.write("<h2>Resumen de subtítulos</h2>")
        f.write("<table><thead><tr>")
        headers = [
            "Texto original",
            "Romaji/Pinyin",
            "Nota contextual",
            "GPT",
            "Claude",
            "Gemini",
            "DeepSeek",
        ]
        for h in headers:
            f.write(f"<th>{html.escape(h)}</th>")
        f.write("</tr></thead><tbody>")

        for i, ev in enumerate(events):
            original_lines, extra_lines = _split_event_original_and_extra(ev)
            hidden_lines = [line for line in extra_lines if _is_hidden_ass_line(line)]
            original = "\n".join(line.strip() for line in original_lines if line.strip())
            roman = _extract_braced_text(hidden_lines[0]) if len(hidden_lines) > 0 else ""
            morph = _extract_braced_text(hidden_lines[1]) if len(hidden_lines) > 1 else ""

            row_vals = [
                original,
                roman,
                morph,
                safe_get(gpt, i),
                safe_get(claude, i),
                safe_get(gemini, i),
                safe_get(deepseek, i),
            ]

            f.write("<tr>")
            for col_idx, v in enumerate(row_vals):
                if col_idx == 2 and v:  # columna "Nota contextual"
                    cell_html = format_morph_cell_html(v)
                    f.write("<td>" + cell_html + "</td>")
                else:
                    f.write("<td>" + html.escape(v).replace("\n", "<br>") + "</td>")
            f.write("</tr>")

        f.write("</tbody></table></body></html>")

    print(f"[+] HTML generado: {output_path}")


# ============================================================
#  ORQUESTADOR: TRADUCCIONES
# ============================================================

def process_all_models_with_subs(
    subs: pysubs2.SSAFile,
    lang: str,
    series_name: str,
    source_type: str,
    base_name: str,
    models: Set[str],
    out_dir: str,
) -> Tuple[Dict[str, List[str]], Dict[str, ApiUsage], Dict[str, float]]:
    def _should_write_output(reason: Optional[str]) -> bool:
        return reason not in {"missing_key", "client_error", "auth_error"}

    # Normalizar por si nos llegan nombres “bonitos”
    norm: Set[str] = set()
    for m in (models or set()):
        key = MODEL_ALIASES.get(str(m).strip().lower(), str(m).strip().lower())
        if key in DISPLAY_NAMES:
            norm.add(key)
    models = norm

    events = [ev for ev in subs if not getattr(ev, "is_comment", False)]
    src_lines: List[str] = []
    for ev in events:
        src_lines.append(_event_source_text(ev))

    total = len(src_lines)
    print(f"There are {total} dialogue lines to translate.")
    if models:
        print("[Models] Running:", ", ".join(DISPLAY_NAMES[m] for m in sorted(models)))
    else:
        print("[Models] None selected; translation will be skipped.")
        return {}, {}, {}

    os.makedirs(out_dir, exist_ok=True)
    debug_dir = os.path.join(out_dir, "_debug", base_name)

    translations_by_model: Dict[str, List[str]] = {}
    usage_by_model: Dict[str, ApiUsage] = {}
    model_timings: Dict[str, float] = {}
    skipped: Dict[str, str] = {}

    if "gpt" in models:
        print(f"=== {DISPLAY_NAMES['gpt']} ===")
        start = time.time()
        gpt_trans, gpt_usage, gpt_skip = translate_with_openai(
            src_lines, lang, series_name, source_type, debug_dir=debug_dir
        )
        elapsed = time.time() - start
        model_timings["gpt"] = elapsed
        translations_by_model["gpt"] = gpt_trans
        usage_by_model["gpt"] = gpt_usage
        if gpt_skip:
            skipped["gpt"] = gpt_skip
        if _should_write_output(gpt_skip):
            gpt_out = os.path.join(out_dir, f"{base_name}_gpt.ass")
            apply_translations_and_save_subs(subs, gpt_trans, gpt_out)
        else:
            print(f"[{DISPLAY_NAMES['gpt']}] No se escribe archivo (motivo: {gpt_skip}).")
        print(f"[{DISPLAY_NAMES['gpt']}] Tiempo total: {elapsed:.1f} s\n")

    if "claude" in models:
        print(f"=== {DISPLAY_NAMES['claude']} ===")
        start = time.time()
        claude_trans, claude_usage, claude_skip = translate_with_claude(
            src_lines, lang, series_name, source_type, debug_dir=debug_dir
        )
        elapsed = time.time() - start
        model_timings["claude"] = elapsed
        translations_by_model["claude"] = claude_trans
        usage_by_model["claude"] = claude_usage
        if claude_skip:
            skipped["claude"] = claude_skip
        if _should_write_output(claude_skip):
            claude_out = os.path.join(out_dir, f"{base_name}_claude.ass")
            apply_translations_and_save_subs(subs, claude_trans, claude_out)
        else:
            print(f"[{DISPLAY_NAMES['claude']}] No se escribe archivo (motivo: {claude_skip}).")
        print(f"[{DISPLAY_NAMES['claude']}] Tiempo total: {elapsed:.1f} s\n")

    if "gemini" in models:
        print(f"=== {DISPLAY_NAMES['gemini']} ===")
        start = time.time()
        gemini_trans, gemini_usage, gemini_skip = translate_with_gemini(
            src_lines, lang, series_name, source_type, debug_dir=debug_dir
        )
        elapsed = time.time() - start
        model_timings["gemini"] = elapsed
        translations_by_model["gemini"] = gemini_trans
        usage_by_model["gemini"] = gemini_usage
        if gemini_skip:
            skipped["gemini"] = gemini_skip
        if _should_write_output(gemini_skip):
            gemini_out = os.path.join(out_dir, f"{base_name}_gemini.ass")
            apply_translations_and_save_subs(subs, gemini_trans, gemini_out)
        else:
            print(f"[{DISPLAY_NAMES['gemini']}] No se escribe archivo (motivo: {gemini_skip}).")
        print(f"[{DISPLAY_NAMES['gemini']}] Tiempo total: {elapsed:.1f} s\n")

    if "deepseek" in models:
        print(f"=== {DISPLAY_NAMES['deepseek']} ===")
        start = time.time()
        deepseek_trans, deepseek_usage, deepseek_skip = translate_with_deepseek(
            src_lines, lang, series_name, source_type, debug_dir=debug_dir
        )
        elapsed = time.time() - start
        model_timings["deepseek"] = elapsed
        translations_by_model["deepseek"] = deepseek_trans
        usage_by_model["deepseek"] = deepseek_usage
        if deepseek_skip:
            skipped["deepseek"] = deepseek_skip
        if _should_write_output(deepseek_skip):
            deepseek_out = os.path.join(out_dir, f"{base_name}_deepseek.ass")
            apply_translations_and_save_subs(subs, deepseek_trans, deepseek_out)
        else:
            print(f"[{DISPLAY_NAMES['deepseek']}] No se escribe archivo (motivo: {deepseek_skip}).")
        print(f"[{DISPLAY_NAMES['deepseek']}] Tiempo total: {elapsed:.1f} s\n")

    if skipped:
        print("[Modelos] Saltados parcial/total:", ", ".join(f"{DISPLAY_NAMES[k]} ({v})" for k, v in skipped.items()))

    return translations_by_model, usage_by_model, model_timings

# ============================================================
#  MAIN
# ============================================================

def main(argv: Optional[List[str]] = None):
    run_started_at = time.time()
    run_started_override = os.getenv("TAKOWORKS_TRANSCRIBER_RUN_STARTED_AT", "").strip()
    if run_started_override:
        try:
            run_started_at = float(run_started_override)
        except Exception:
            pass
    phase_timings: Dict[str, float] = {}
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe an .ass + video to Japanese or Chinese (Anime-Whisper / BELLE-2), "
            "refine punctuation with free models, add romaji/pinyin via DeepSeek and an optional "
            "context note with Claude Sonnet, and translate with GPT, Claude, Gemini, and DeepSeek. "
            "Each output .ass can contain:\n"
            "  - line 1: Japanese/Chinese\n"
            "  - line 2: romaji/pinyin (if enabled)\n"
            "  - line 3: context note (if enabled)\n"
            "  - last line: translation."
        )
    )
    parser.add_argument("ass_in", help="Input .ass file (with synced timing).")
    parser.add_argument("video_in", nargs="?", default="", help="Matching video (optional if --skip-asr).")
    parser.add_argument(
        "--out-dir",
        help="Folder where the output files (.ass, .html) will be saved. "
             "Defaults to the input .ass folder.",
        default=None,
    )
    parser.add_argument(
        "--base-name",
        help="Prefix for the output .ass files (defaults to the input .ass base name).",
        default=None,
    )
    parser.add_argument(
        "--models",
        help=(
            "Comma-separated list of models to run. "
            "Options: GPT-5.5, Claude Opus 4.7, Gemini 3 Flash, DeepSeek V4 Flash (or gpt, claude, gemini, deepseek). "
            "Default: gpt,claude,gemini,deepseek"
        ),
        default="GPT-5.5,Claude Opus 4.7,Gemini 3 Flash,DeepSeek V4 Flash",
    )
    parser.add_argument(
        "--pad-ms",
        type=int,
        default=0,
        help="Padding in milliseconds at the start and end of each line when trimming audio.",
    )
    parser.add_argument(
        "--lang",
        choices=["ja", "zh"],
        help="Original language of the audio or script (ja = Japanese, zh = Chinese Mandarin). "
             "If omitted, you will be prompted in the console.",
    )
    parser.add_argument(
        "--series",
        help="Series name (for example, 'Dragon Raja'). "
             "If omitted, you will be prompted in the console.",
    )
    parser.add_argument(
        "--source-type",
        choices=["Manga", "Manhwa", "Light novel", "None"],
        help="Source material type (manga, manhwa, light novel, none). "
             "If omitted, you will be prompted in the console.",
    )
    parser.add_argument(
        "--do-roman-morph",
        action="store_true",
        help="Add romaji/pinyin via DeepSeek and a context note with Claude Sonnet to the ASS.",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate a summary HTML with original text, romanization, context notes, and translations.",
    )
    parser.add_argument(
        "--skip-asr",
        action="store_true",
        help=(
            "Skip audio transcription. The .ass is assumed to already contain the "
            "Japanese or Chinese transcription in the first line of each subtitle. "
            "You can still add romaji/pinyin via DeepSeek and a context note (--do-roman-morph) "
            "and perform translations."
        ),
    )

    args = parser.parse_args(argv)

    # Normalizamos rutas y carpeta de salida
    ass_in = os.path.abspath(args.ass_in)
    video_in = os.path.abspath(args.video_in) if args.video_in else ""

    if args.out_dir:
        out_dir = os.path.abspath(args.out_dir)
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = os.path.dirname(ass_in) or "."

    base_name = args.base_name or os.path.splitext(os.path.basename(ass_in))[0]
    run_id = str(uuid.uuid4())

    if not args.skip_asr and not video_in:
        raise SystemExit("Falta video_in (obligatorio si NO usas --skip-asr).")

    # Idioma, serie y tipo de material
    if args.lang:
        lang = args.lang
    else:
        lang = ask_language()

    if args.series:
        series_name = args.series
    else:
        series_name = ask_series_name()

    if args.source_type:
        source_type = normalize_source_type(args.source_type)
    else:
        source_type = ask_source_type()

    romanization_usage = ApiUsage(engine="deepseek", model_name=DEEPSEEK_MODEL)
    context_note_usage = ApiUsage(engine="context_note", model_name=CONTEXT_NOTE_MODEL)

    # 1) Obtener subs de partida
    if args.skip_asr:
        print("[+] Skipping audio transcription phase: the text already present in the .ass will be used.")
        subs = pysubs2.load(ass_in, encoding="utf-8")

        if args.do_roman_morph:
            print("[+] Adding romaji/pinyin and a context note on top of the existing script.")
            subs = add_roman_morph_to_subs(
                subs,
                lang,
                romanization_usage=romanization_usage,
                context_note_usage=context_note_usage,
                phase_timings=phase_timings,
            )
        else:
            print("[+] --do-roman-morph is NOT enabled: the script will be used as-is for translation.")

        # Guardamos un intermedio igualmente, para tener copia de trabajo
        asr_suffix = "_ja_asr" if lang == "ja" else "_zh_asr"
        asr_out = os.path.join(out_dir, f"{base_name}{asr_suffix}.ass")
        subs.save(asr_out, encoding="utf-8-sig")
        print(f"[+] Intermediate file (no ASR, only romanization/context note if applicable): {asr_out}\n")

    else:
        print("[+] Running the full pipeline: ASR + punctuation + romaji/pinyin (if applicable).")
        subs = transcribe_ass(
            ass_in,
            video_in,
            pad_ms=args.pad_ms,
            lang=lang,
            do_roman_morph=args.do_roman_morph,
            romanization_usage=romanization_usage,
            context_note_usage=context_note_usage,
            phase_timings=phase_timings,
        )

        # Guardamos ASS intermedio (asr)
        asr_suffix = "_ja_asr" if lang == "ja" else "_zh_asr"
        asr_out = os.path.join(out_dir, f"{base_name}{asr_suffix}.ass")
        subs.save(asr_out, encoding="utf-8-sig")
        print(f"[+] Intermediate file (transcription + punctuation + romanization/context note only): {asr_out}\n")

    # 2) Traducir
    models = normalize_models_arg(args.models)
    translations_by_model, usage_by_model, model_timings = process_all_models_with_subs(
        subs,
        lang,
        series_name,
        source_type,
        base_name,
        models,
        out_dir,
    )

    if romanization_usage.total_tokens > 0:
        usage_by_model["romanization"] = merge_api_usage(
            usage_by_model.get(
                "romanization",
                ApiUsage(engine="romanization", model_name=DEEPSEEK_MODEL),
            ),
            romanization_usage,
        )
        print(
            f"[Costs] DeepSeek V4 Flash romanization: prompt={romanization_usage.prompt_tokens} "
            f"completion={romanization_usage.completion_tokens} "
            f"total={romanization_usage.total_tokens} cost_usd=${romanization_usage.cost_usd:.4f}"
        )

    if context_note_usage.total_tokens > 0:
        usage_by_model["context_note"] = merge_api_usage(
            usage_by_model.get(
                "context_note",
                ApiUsage(engine="context_note", model_name=CONTEXT_NOTE_MODEL),
            ),
            context_note_usage,
        )
        print(
            f"[Costs] Claude Sonnet 4.6 explanatory prompt: prompt={context_note_usage.prompt_tokens} "
            f"completion={context_note_usage.completion_tokens} "
            f"total={context_note_usage.total_tokens} cost_usd=${context_note_usage.cost_usd:.4f}"
        )

    if usage_by_model:
        log_cost_summary(run_id, usage_by_model, series_name, base_name)
        persist_costs_to_supabase(run_id, series_name, base_name, lang, usage_by_model)

    log_time_cost_breakdown_v2(
        phase_timings,
        romanization_usage,
        context_note_usage,
        model_timings,
        usage_by_model,
        time.time() - run_started_at,
    )

    # 3) HTML opcional
    if args.html:
        html_out = os.path.join(out_dir, f"{base_name}_summary.html")
        generate_html(subs, translations_by_model, html_out)


if __name__ == "__main__":
    main()
