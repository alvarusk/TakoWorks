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
from collections import defaultdict
from functools import lru_cache
from ... import config as app_config
from .ass_utils import (
    _ass_hide,
    _ass_hide_prefix,
    _ass_sanitize_braces,
    _ass_unsanitize_braces,
)
from .json_utils import parse_json_translations

try:
    import requests  # type: ignore
except Exception:
    requests = None

import pysubs2
import torch
from transformers import pipeline
from pykakasi import kakasi
from pypinyin import lazy_pinyin, Style

from fugashi import Tagger
import jieba.posseg as pseg

from openai import OpenAI          # OpenAI + DeepSeek (API compatible)
import anthropic                   # Claude
import google.generativeai as genai  # Gemini

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

OPENAI_MODEL   = "gpt-5.1"                  # OpenAI (ajusta si hace falta)
CLAUDE_MODEL   = "claude-opus-4-5-20251101" # Anthropic (ajusta si hace falta)
GEMINI_MODEL   = "gemini-2.5-flash"         # Gemini 2.5 Flash
DEEPSEEK_MODEL = "deepseek-chat"            # DeepSeek (OpenAI-like)

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

# Directorios por defecto para diccionarios Yomitan (puedes sobreescribir con env vars)
YOMI_JA_DIR = os.getenv("YOMI_JA_DIR", r"C:\Transcriber\jpdict")
YOMI_ZH_DIR = os.getenv("YOMI_ZH_DIR", r"C:\Transcriber\cndict")



# ============================================================
#  NOMBRES DE MODELOS (UI) Y NORMALIZACIÓN
# ============================================================

# Nombres que quieres ver en la GUI / logs / HTML
DISPLAY_NAMES = {
    "gpt": "GPT-5",
    "claude": "Claude",
    "gemini": "Gemini 2.5 Flash",
    "deepseek": "DeepSeek",
}

# Alias aceptados en --models (CLI/GUI). Se normalizan a claves internas:
MODEL_ALIASES = {
    "gpt-5": "gpt",
    "gpt5": "gpt",
    "gpt": "gpt",
    "openai": "gpt",
    "chatgpt": "gpt",

    "claude": "claude",
    "anthropic": "claude",

    "gemini 2.5 flash": "gemini",
    "gemini-2.5-flash": "gemini",
    "gemini": "gemini",

    "deepseek": "deepseek",
    "deepseek-chat": "deepseek",
}

def normalize_models_arg(models_str: str) -> Set[str]:
    """
    Acepta tanto: 'gpt,claude' como 'GPT-5,Claude,Gemini 2.5 Flash,DeepSeek'
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
    "gpt": {"input": 0.0, "output": 0.0},
    "claude": {"input": 0.0, "output": 0.0},
    "gemini": {"input": 0.0, "output": 0.0},
    "deepseek": {"input": 0.0, "output": 0.0},
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


_PRICE_TABLE = _load_price_table()
_WARNED_PRICING: Set[str] = set()
_WARNED_MISSING_USAGE: Set[str] = set()
_WARNED_JA_TAGGER: bool = False
_JA_TAGGER_FAILED: bool = False


def estimate_cost(model_key: str, prompt_tokens: int, completion_tokens: int) -> float:
    prices = _PRICE_TABLE.get(model_key, {})
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
_ja_dict_en: Optional[Dict[str, List[str]]] = None  # japonés → glosas EN
_zh_dict_en: Optional[Dict[str, List[str]]] = None  # chino   → glosas EN


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
                "Se usa kakasi simple y se omite analisis diccionario. "
                f"Detalle: {e}"
            )
            _WARNED_JA_TAGGER = True
        _ja_tagger = None
    return _ja_tagger


# ============================================================
#  UTILIDADES: DICCIONARIO YOMITAN
# ============================================================

def load_yomitan_dict(dir_path: str) -> Dict[str, List[str]]:
    """
    Carga un diccionario Yomitan/Yomichan desde un directorio (recursivo) y construye un mapping:
        forma (expression / reading) -> lista de glosas (strings EN).

    Soporta árboles de diccionarios típicos (subcarpetas). Sirve tanto para JA como para ZH.
    """
    mapping: Dict[str, List[str]] = defaultdict(list)

    if not os.path.isdir(dir_path):
        print(f"[yomitan] Directorio no encontrado: {dir_path}")
        return mapping

    print(
        "[yomitan] Cargando diccionario desde "
        f"{dir_path} (term_bank_*, kanji_bank_*, hanzi_bank_*.json)..."
    )

    def collect_strings(obj) -> List[str]:
        """Extrae todos los strings anidados en listas/dicts (Yomitan puede venir muy anidado)."""
        if isinstance(obj, str):
            return [obj]
        if isinstance(obj, list):
            res: List[str] = []
            for x in obj:
                res.extend(collect_strings(x))
            return res
        if isinstance(obj, dict):
            res: List[str] = []
            for v in obj.values():
                res.extend(collect_strings(v))
            return res
        return []

    prefixes = ("term_bank_", "kanji_bank_", "hanzi_bank_")

    # Recursivo: muchos diccionarios vienen en subcarpetas
    for root, _dirs, files in os.walk(dir_path):
        for fname in files:
            if not fname.endswith(".json") or not fname.startswith(prefixes):
                continue

            path = os.path.join(root, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    entries = json.load(f)
            except Exception as e:
                print(f"[yomitan] Error leyendo {path}: {e}")
                continue

            if not isinstance(entries, list):
                continue

            for entry in entries:
                forms = set()
                glosses: List[str] = []

                if isinstance(entry, list):
                    # 0: expresión (hanzi/kanji), 1: lectura
                    if len(entry) >= 1 and isinstance(entry[0], str):
                        forms.add(entry[0])
                    if len(entry) >= 2 and isinstance(entry[1], str):
                        forms.add(entry[1])

                    specific_indices: List[int] = []
                    if len(entry) > 4:
                        specific_indices.append(4)
                    if len(entry) > 5:
                        specific_indices.append(5)

                    cand: List[str] = []
                    for idx in specific_indices:
                        part = entry[idx]
                        if isinstance(part, (list, dict)):
                            cand = [s.strip() for s in collect_strings(part) if s and isinstance(s, str)]
                            if cand:
                                break

                    if not cand:
                        for it in entry:
                            if isinstance(it, (list, dict)):
                                cand = [s.strip() for s in collect_strings(it) if s and isinstance(s, str)]
                                if cand:
                                    break

                    if cand:
                        # preferimos glosas en inglés (letras latinas); si hay frases con espacios, mejor
                        glosses = [s for s in cand if re.search(r"[A-Za-z]", s) and " " in s]
                        if not glosses:
                            glosses = [s for s in cand if re.search(r"[A-Za-z]", s)]
                        if not glosses:
                            glosses = cand

                elif isinstance(entry, dict):
                    term = entry.get("term") or entry.get("expression") or entry.get("kanji")
                    reading = entry.get("reading")
                    if isinstance(term, str):
                        forms.add(term)
                    if isinstance(reading, str):
                        forms.add(reading)

                    defs = (
                        entry.get("glossary")
                        or entry.get("glossaries")
                        or entry.get("definition")
                        or entry.get("meanings")
                    )
                    if defs is not None:
                        cand = [s.strip() for s in collect_strings(defs) if s and isinstance(s, str)]
                        if cand:
                            glosses = [s for s in cand if re.search(r"[A-Za-z]", s) and " " in s]
                            if not glosses:
                                glosses = [s for s in cand if re.search(r"[A-Za-z]", s)]
                            if not glosses:
                                glosses = cand

                if not forms or not glosses:
                    continue

                for form in forms:
                    mapping[form].extend(glosses)

    # Deduplicar glosas por forma
    for k, v in list(mapping.items()):
        seen = set()
        deduped: List[str] = []
        for g in v:
            if g not in seen:
                seen.add(g)
                deduped.append(g)
        mapping[k] = deduped

    print(f"[yomitan] Entradas cargadas: {len(mapping)}")
    return mapping

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
            print(f"[Puntuación ja] Error al procesar línea, se mantiene original: {e}")
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
            print(f"[Puntuación zh] Error al procesar línea, se mantiene original: {e}")
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
def clean_gloss_list(glosses: List[str]) -> List[str]:
    """
    Limpia la lista de glosas que viene del diccionario Yomitan:
    - Quita etiquetas técnicas (div, span, zh-Hant, etc.).
    - Quita cadenas sin letras latinas (para quedarnos con glosas EN).
    """
    META_TOKENS = {
        "structured-content",
        "div",
        "span",
        "ul",
        "li",
        "zh",
        "zh-Hans",
        "zh-Hant",
        "headword",
        "headword-trad",
        "headword-simp",
        "definition",
    }

    cleaned: List[str] = []
    for g in glosses:
        s = (g or "").strip()
        if not s:
            continue

        # etiquetas tipo div, span, zh-Hant, etc.
        if s in META_TOKENS:
            continue

        # nos quedamos solo con cosas que parecen inglés
        if not re.search(r"[A-Za-z]", s):
            continue

        cleaned.append(s)

    return cleaned

def analyze_japanese_morph(text: str) -> str:
    """
    Análisis tipo diccionario JA→EN usando SOLO diccionario Yomitan.
    - Usa fugashi (UniDic) para segmentar.
    - Mira en el diccionario Yomitan japonés-inglés (YOMI_JA_DIR).
    - Si no encuentra glosas con letras inglesas, devuelve "" (sin fallback a GPT).
    Formato de salida:
      表層 (lemma, POS) -> short English gloss | ...
    """
    global _ja_tagger, _ja_dict_en

    text = (text or "").strip()
    if not text:
        return ""

    if not os.path.isdir(YOMI_JA_DIR):
        print(f"[JA dict] Directorio no encontrado: {YOMI_JA_DIR}")
        return ""

    tagger = _ensure_ja_tagger()
    if tagger is None:
        return ""
    if _ja_dict_en is None:
        _ja_dict_en = load_yomitan_dict(YOMI_JA_DIR)
    if _ja_dict_en is not None:
        print(f"[JA dict] Entradas: {len(_ja_dict_en)}")

    tokens_desc: List[str] = []
    interesting_pos = {"名詞", "動詞", "形容詞", "副詞", "助詞"}

    for word in tagger(text):
        surface = word.surface
        lemma = getattr(word.feature, "lemma", surface) or surface
        main_pos = word.pos.split("-")[0]  # p.ej. 名詞-普通名詞-一般 → 名詞

        if main_pos not in interesting_pos:
            continue

        glosses: Optional[List[str]] = None
        for key in (lemma, surface):
            g = _ja_dict_en.get(key) if _ja_dict_en else None
            if g:
                glosses = clean_gloss_list(g)
                if glosses:
                    break

        if not glosses:
            continue

        gloss_str = "; ".join(glosses)
        # si quieres sin límite, no pongas ningún if aquí
        #if len(gloss_str) > 140:
        #    gloss_str = gloss_str[:137] + "..."

        pos_label = {
            "名詞": "n.",
            "動詞": "v.",
            "形容詞": "adj.",
            "副詞": "adv.",
            "助詞": "part.",
        }.get(main_pos, main_pos)

        tokens_desc.append(f"{surface} ({lemma}, {pos_label}) -> {gloss_str}")

    return " | ".join(tokens_desc)


def analyze_chinese_morph(text: str) -> str:
    """
    Análisis tipo diccionario ZH→EN usando SOLO diccionario Yomitan.
    - Usa jieba.posseg para segmentar.
    - Mira en el diccionario Yomitan chino-inglés (YOMI_ZH_DIR).
    - Si no encuentra glosas con letras inglesas, devuelve "" (sin fallback a GPT).
    Formato de salida:
      词 (POS) -> short English gloss | ...
    """
    global _zh_dict_en

    text = (text or "").strip()
    if not text:
        return ""

    if not os.path.isdir(YOMI_ZH_DIR):
        # No hay diccionario -> sin análisis
        return ""

    if _zh_dict_en is None:
        _zh_dict_en = load_yomitan_dict(YOMI_ZH_DIR)
    if _zh_dict_en is not None:
        print(f"[ZH dict] Entradas: {len(_zh_dict_en)}")

    tokens_desc: List[str] = []

    content_pos_prefixes = ("n", "v", "a", "d")
    particle_prefix = "u"

    for w, flag in pseg.cut(text):
        if not w.strip():
            continue

        # filtramos por categorías que suelen ser interesantes
        if not (flag[0] in content_pos_prefixes or flag.startswith(particle_prefix)):
            continue

        glosses = _zh_dict_en.get(w) if _zh_dict_en else None
        if not glosses:
            continue

        glosses = clean_gloss_list(glosses)
        if not glosses:
            continue

        gloss_str = "; ".join(glosses)
        tokens_desc.append(f"{w} ({flag}) -> {gloss_str}")


    return " | ".join(tokens_desc)

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

    if lang == "ja":
        print("[+] Idioma seleccionado: japonés (Anime-Whisper)")
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
        print("[+] Idioma seleccionado: chino (BELLE-2/Belle-whisper-large-v3-zh)")
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
) -> pysubs2.SSAFile:
    """
    Carga un .ass, transcribe audio, pule la puntuación con modelos libres
    y opcionalmente añade:
      - romaji/pinyin
      - análisis tipo diccionario
    en líneas adicionales (separadas con \\N).
    Además, imprime progreso por línea para que la GUI
    pueda mostrar en qué línea va y el % completado.
    """
    global _ja_tagger

    print("[+] Cargando ASS para transcripción.")
    subs = pysubs2.load(ass_path, encoding="utf-8")

    events: List[pysubs2.SSAEvent] = []
    audio_paths: List[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        print("[+] Preparando segmentos de audio con ffmpeg.")

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
                print(f"[!] ffmpeg falló en línea {idx} ({ev.start}–{ev.end} ms): {e}")
                continue

            events.append(ev)
            audio_paths.append(seg_path)

        if not audio_paths:
            print("[!] No se generó ningún segmento de audio. Se devuelve el ASS sin cambios.")
            return subs

        total = len(events)
        print(f"[+] Segmentos preparados: {total}")
        print("[+] Cargando modelo de transcripción (puede tardar la primera vez).")
        asr = build_asr_pipeline(lang)

        print("[+] Transcribiendo líneas.")
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
                print(f"[ASR] Error en línea {i}/{total}: {e}")
                txt = ""

            # 🔹 Limpiar repeticiones absurdas tipo 痛たたたたたたた....
            txt = clean_repetitions(txt)

            raw_lines.append(txt)
            snippet = txt.replace("\n", " ")[:60]
            print(f"[Transcripción] Línea {i}/{total} → {snippet}")

        print("[+] Refinando puntuación (modelos libres, sin GPT).")
        refined_lines = refine_punctuation_free(raw_lines, lang)

        romaji_converter = build_romaji_converter() if (lang == "ja" and do_roman_morph) else None
        ja_tagger = _ensure_ja_tagger() if (lang == "ja" and do_roman_morph) else None

        for i, (ev, text) in enumerate(zip(events, refined_lines), start=1):
            base_text = (text or "").strip()
            if not base_text:
                continue

            lines = [base_text]

            if do_roman_morph:
                if lang == "ja":
                    romaji = ""
                    if romaji_converter is not None:
                        if ja_tagger is not None:
                            romaji = japanese_to_romaji_pretty(base_text, romaji_converter, ja_tagger)
                        else:
                            romaji = japanese_to_romaji_line(base_text, romaji_converter)
                    if romaji:
                        lines.append("{" + _ass_sanitize_braces(romaji) + "}")

                    if ja_tagger is not None:
                        morph_line = analyze_japanese_morph(base_text)
                        if morph_line:
                            lines.append(_ass_hide(morph_line))

                elif lang == "zh":
                    pinyin = text_to_pinyin(base_text)
                    if pinyin:
                        lines.append("{" + _ass_sanitize_braces(pinyin) + "}")
                    morph_line = analyze_chinese_morph(base_text)
                    if morph_line:
                        lines.append(_ass_hide(morph_line))

            ev.text = "\\N".join(lines)
            snippet = base_text.replace("\n", " ")[:60]
            print(f"[Romaji/Pinyin] Línea {i}/{total} → {snippet}")

    print("[+] Transcripción completada.")
    return subs


def add_roman_morph_to_subs(subs: pysubs2.SSAFile, lang: str) -> pysubs2.SSAFile:
    """
    Añade romaji/pinyin + análisis diccionario a un ASS que YA tiene el guion
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
        print("[Romaji/Pinyin] No hay líneas de diálogo sobre las que trabajar.")
        return subs

    print(f"[Romaji/Pinyin] Hay {total} líneas de diálogo sobre las que trabajar.")

    # Preparar recursos según el idioma
    romaji_converter = None
    ja_tagger = None
    if lang == "ja":
        romaji_converter = build_romaji_converter()
        ja_tagger = _ensure_ja_tagger()

    for i, ev in enumerate(events, start=1):
        raw_text = (ev.text or "")
        parts = raw_text.split("\\N")

        # Primera línea = texto base en JA/ZH
        base_text = (parts[0] or "").strip()
        if not base_text:
            continue

        # Líneas extra ya existentes (por si ya tenías traducción debajo)
        extra_lines = parts[1:]

        # Reconstruimos las líneas del evento
        lines = [base_text]

        if lang == "ja":
            # Romaji "bonito"
            romaji = ""
            if romaji_converter is not None:
                if ja_tagger is not None:
                    romaji = japanese_to_romaji_pretty(base_text, romaji_converter, ja_tagger)
                else:
                    romaji = japanese_to_romaji_line(base_text, romaji_converter)
            if romaji:
                lines.append("{" + _ass_sanitize_braces(romaji) + "}")

            # Análisis diccionario JA→EN (si tienes jpdict configurado)
            if ja_tagger is not None:
                morph_line = analyze_japanese_morph(base_text)
                if morph_line:
                    lines.append(_ass_hide(morph_line))

        elif lang == "zh":
            # Pinyin
            pinyin = text_to_pinyin(base_text)
            if pinyin:
                lines.append("{" + _ass_sanitize_braces(pinyin) + "}")

            # Análisis diccionario ZH→EN (si tienes cndict configurado)
            morph_line = analyze_chinese_morph(base_text)
            if morph_line:
                lines.append(_ass_hide(morph_line))

        # Conservamos cualquier línea extra ya existente
        lines.extend(extra_lines)

        ev.text = "\\N".join(lines)

        snippet = base_text.replace("\n", " ")[:60]
        print(f"[Romaji/Pinyin] Línea {i}/{total} → {snippet}")

    return subs

# ============================================================
#  PROMPTS Y PREGUNTAS (solo CLI)
# ============================================================

def ask_language() -> str:
    while True:
        print("Elige idioma para la transcripción:")
        print("  [j] Japonés")
        print("  [c] Chino (mandarín)")
        choice = input("Opción (j/c): ").strip().lower()

        if choice in ("j", "ja", "jp", "japones", "japonés"):
            return "ja"
        if choice in ("c", "zh", "ch", "chino", "mandarin", "mandarín"):
            return "zh"

        print("Entrada no válida. Por favor escribe 'j' o 'c'.\n")


def ask_series_name() -> str:
    series = input("¿De qué serie se trata? (ej.: Dragon Raja): ").strip()
    if not series:
        series = "esta serie"
    return series


def ask_source_type() -> str:
    print("¿La serie tiene material original?")
    print("  [1] Manga")
    print("  [2] Manhwa")
    print("  [3] Novela ligera")
    print("  [4] Nada / no lo sé")
    while True:
        choice = input("Opción (1/2/3/4): ").strip()
        if choice == "1":
            return "Manga"
        if choice == "2":
            return "Manhwa"
        if choice == "3":
            return "Novela ligera"
        if choice == "4":
            return "Nada"
        print("Entrada no válida. Escribe 1, 2, 3 o 4.\n")


def describe_source_type(source_type: str) -> str:
    if source_type == "Manga":
        return "La serie está basada en un manga. Ten en cuenta la terminología y traducciones oficiales del manga cuando sea posible."
    if source_type == "Manhwa":
        return "La serie está basada en un manhwa. Ten en cuenta la terminología y traducciones oficiales del manhwa cuando sea posible."
    if source_type == "Novela ligera":
        return "La serie está basada en una novela ligera. Ten en cuenta la terminología y traducciones oficiales de la novela ligera cuando sea posible."
    return "No hay material original claramente definido o no es relevante; prioriza la coherencia interna de la serie."


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
    return OpenAI(api_key=OPENAI_API_KEY)


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
    genai.configure(api_key=GEMINI_API_KEY)
    system_prompt = build_system_prompt(lang, series_name, source_type)
    return genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system_prompt,
    )


# ============================================================
#  TRADUCCIÓN POR MODELO
# ============================================================

def translate_with_openai(
    src_lines: List[str],
    lang: str,
    series_name: str,
    source_type: str,
) -> Tuple[List[str], ApiUsage, Optional[str]]:
    if not OPENAI_API_KEY:
        print("[GPT-5] Se omite GPT porque falta OPENAI_API_KEY (env o config.local.json).")
        return src_lines, ApiUsage(engine="gpt", model_name=OPENAI_MODEL), "missing_key"

    try:
        client = get_openai_client()
    except Exception as e:
        print(f"[GPT-5] No se puede inicializar el cliente: {e}. Se omite GPT.")
        return src_lines, ApiUsage(engine="gpt", model_name=OPENAI_MODEL), "client_error"

    system_prompt = build_system_prompt(lang, series_name, source_type)
    all_translations: List[str] = []
    total = len(src_lines)
    usage = ApiUsage(engine="gpt", model_name=OPENAI_MODEL)
    skipped_reason: Optional[str] = None

    for start in range(0, total, CHUNK_SIZE):
        chunk = src_lines[start:start + CHUNK_SIZE]
        user_prompt = build_user_prompt(chunk, lang, series_name, source_type)
        end_line = min(start + CHUNK_SIZE, total)
        print(f"[{DISPLAY_NAMES['gpt']}] Líneas {start + 1}-{end_line} de {total}...")

        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[GPT-5] Error al traducir; se omite GPT en este bloque. Detalle: {e}")
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
        translations = parse_json_translations(content, fallback_lines=chunk)
        all_translations.extend(translations)

    return all_translations, usage, skipped_reason


def translate_with_deepseek(
    src_lines: List[str],
    lang: str,
    series_name: str,
    source_type: str,
) -> Tuple[List[str], ApiUsage, Optional[str]]:
    try:
        client = get_deepseek_client()
    except Exception as e:
        print(f"[DeepSeek] Se omite DeepSeek (cliente no inicializado): {e}")
        return src_lines, ApiUsage(engine="deepseek", model_name=DEEPSEEK_MODEL), "client_error"
    system_prompt = build_system_prompt(lang, series_name, source_type)
    all_translations: List[str] = []
    total = len(src_lines)
    usage = ApiUsage(engine="deepseek", model_name=DEEPSEEK_MODEL)
    skipped_reason: Optional[str] = None

    for start in range(0, total, CHUNK_SIZE):
        chunk = src_lines[start:start + CHUNK_SIZE]
        user_prompt = build_user_prompt(chunk, lang, series_name, source_type)
        end_line = min(start + CHUNK_SIZE, total)
        print(f"[DeepSeek] Líneas {start + 1}-{end_line} de {total}...")

        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[DeepSeek] Error al traducir; se omite DeepSeek en este bloque. Detalle: {e}")
            skipped_reason = skipped_reason or "partial_error"
            translations = chunk
            all_translations.extend(translations)
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
        translations = parse_json_translations(content, fallback_lines=chunk)
        all_translations.extend(translations)

    return all_translations, usage, skipped_reason


def translate_with_claude(
    src_lines: List[str],
    lang: str,
    series_name: str,
    source_type: str,
) -> Tuple[List[str], ApiUsage, Optional[str]]:
    try:
        client = get_claude_client()
    except Exception as e:
        print(f"[Claude] Se omite Claude (cliente no inicializado): {e}")
        return src_lines, ApiUsage(engine="claude", model_name=CLAUDE_MODEL), "client_error"

    system_prompt = build_system_prompt(lang, series_name, source_type)
    all_translations: List[str] = []
    total = len(src_lines)
    usage = ApiUsage(engine="claude", model_name=CLAUDE_MODEL)
    skipped_reason: Optional[str] = None

    for start in range(0, total, CHUNK_SIZE):
        chunk = src_lines[start:start + CHUNK_SIZE]
        user_prompt = build_user_prompt(chunk, lang, series_name, source_type)
        end_line = min(start + CHUNK_SIZE, total)
        print(f"[Claude] Líneas {start + 1}-{end_line} de {total}...")

        try:
            message = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4096,
                temperature=0.1,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
            )
        except Exception as e:
            print(f"[Claude] Error al traducir; se omite Claude en este bloque. Detalle: {e}")
            skipped_reason = skipped_reason or "partial_error"
            translations = chunk
            all_translations.extend(translations)
            continue

        content = "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        ).strip()

        if not content:
            print("[Claude] Respuesta vacía, se devuelven líneas originales para este bloque.")
            skipped_reason = skipped_reason or "partial_error"
            translations = chunk
            all_translations.extend(translations)
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

        translations = parse_json_translations(content, fallback_lines=chunk)
        all_translations.extend(translations)

    return all_translations, usage, skipped_reason


def translate_with_gemini(
    src_lines: List[str],
    lang: str,
    series_name: str,
    source_type: str,
) -> Tuple[List[str], ApiUsage, Optional[str]]:
    """
    Usa Gemini 2.5 Flash, con bloques más pequeños y max_output_tokens
    limitado para ir algo más rápido/estable.
    """
    try:
        model = get_gemini_model(lang, series_name, source_type)
    except Exception as e:
        print(f"[Gemini 2.5 Flash] Se omite Gemini (cliente no inicializado): {e}")
        return src_lines, ApiUsage(engine="gemini", model_name=GEMINI_MODEL), "client_error"
    all_translations: List[str] = []
    total = len(src_lines)
    usage = ApiUsage(engine="gemini", model_name=GEMINI_MODEL)
    skipped_reason: Optional[str] = None

    for start in range(0, total, GEMINI_CHUNK):
        chunk = src_lines[start:start + GEMINI_CHUNK]
        user_prompt = build_user_prompt(chunk, lang, series_name, source_type)
        end_line = min(start + GEMINI_CHUNK, total)
        print(f"[Gemini 2.5 Flash] Líneas {start + 1}-{end_line} de {total}...")

        try:
            response = model.generate_content(
                user_prompt,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 2000,  # prueba 800–1500
                },
            )
        except Exception as e:
            print(f"[Gemini] Error de API en líneas {start + 1}-{end_line}: {e}. "
                  f"Se devuelven las líneas originales para este bloque.")
            all_translations.extend(chunk)
            continue

        cand = None
        if getattr(response, "candidates", None):
            cand = response.candidates[0]
            
        # DEBUG: ver por qué Gemini corta la respuesta
        if cand is not None:
            print(f"[Gemini DEBUG] finish_reason={getattr(cand, 'finish_reason', None)}")
            print(f"[Gemini DEBUG] safety_ratings={getattr(cand, 'safety_ratings', None)}")

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

        if not cand or not getattr(cand, "content", None) or not getattr(cand.content, "parts", None):
            finish_reason = getattr(cand, "finish_reason", None) if cand else None
            safety = getattr(cand, "safety_ratings", None) if cand else None
            print(f"[Gemini] Respuesta vacía o bloqueada (finish_reason={finish_reason}). "
                  "Se devuelven las líneas originales para este bloque.")
            print(f"[Gemini DEBUG] safety_ratings={safety}")
            translations = chunk
            skipped_reason = skipped_reason or "partial_error"
        else:
            text_parts = [
                getattr(part, "text", "")
                for part in cand.content.parts
                if getattr(part, "text", "")
            ]
            raw = "".join(text_parts).strip()

            if not raw:
                finish_reason = getattr(cand, "finish_reason", None)
                safety = getattr(cand, "safety_ratings", None)
                print(f"[Gemini] Sin texto utilizable (finish_reason={finish_reason}). "
                      "Se devuelven las líneas originales para este bloque.")
                print(f"[Gemini DEBUG] safety_ratings={safety}")
                translations = chunk
                skipped_reason = skipped_reason or "partial_error"
            else:
                translations = parse_json_translations(raw, chunk)

        all_translations.extend(translations)

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
        print("[AVISO] Nº de traducciones distinto del nº de líneas; se ajustará al mínimo en común.")
    n = min(len(translations), len(events_out))

    for i in range(n):
        ev = events_out[i]
        trans = translations[i].strip()
        if not trans:
            continue

        ev.text = trans

    subs_out.save(output_path, encoding="utf-8-sig")
    print(f"Guardado: {output_path}")

def format_morph_cell_html(morph: str) -> str:
    """
    Recibe una cadena del tipo:
      '已经 (d) -> already; ... | 能 (v) -> ...'
    y la convierte en HTML:
      <b>已经 (d)</b> -> already; ...<br>
      <b>能 (v)</b> -> ...<br>
    """
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
    Texto original | Romaji/Pinyin | Análisis diccionario | GPT | Claude | Gemini | DeepSeek
    Todas las columnas con el mismo ancho.
    """
    events = [ev for ev in subs if not getattr(ev, "is_comment", False)]

    gpt = translations_by_model.get("gpt", [])
    claude = translations_by_model.get("claude", [])
    gemini = translations_by_model.get("gemini", [])
    deepseek = translations_by_model.get("deepseek", [])

    def safe_get(lst: List[str], idx: int) -> str:
        return lst[idx].strip() if idx < len(lst) else ""

    def extract_braced(raw: str) -> str:
        txt = (raw or "").strip()
        if txt.startswith("{") and txt.endswith("}"):
            txt = txt[1:-1]
        return _ass_unsanitize_braces(txt)

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
            "Análisis diccionario",
            "GPT",
            "Claude",
            "Gemini",
            "DeepSeek",
        ]
        for h in headers:
            f.write(f"<th>{html.escape(h)}</th>")
        f.write("</tr></thead><tbody>")

        for i, ev in enumerate(events):
            parts = (ev.text or "").split("\\N")
            original = parts[0].strip() if parts else ""
            roman   = extract_braced(parts[1]) if len(parts) > 1 else ""
            morph   = extract_braced(parts[2]) if len(parts) > 2 else ""

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
                if col_idx == 2 and v:  # columna "Análisis diccionario"
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
) -> Tuple[Dict[str, List[str]], Dict[str, ApiUsage]]:
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
        text = ev.text or ""
        src = text.split("\\N", 1)[0].strip()
        src_lines.append(src)

    total = len(src_lines)
    print(f"Hay {total} líneas de diálogo para traducir.")
    if models:
        print("[Modelos] Ejecutando:", ", ".join(DISPLAY_NAMES[m] for m in sorted(models)))
    else:
        print("[Modelos] Ninguno seleccionado; se omite la traducción.")
        return {}, {}

    os.makedirs(out_dir, exist_ok=True)

    translations_by_model: Dict[str, List[str]] = {}
    usage_by_model: Dict[str, ApiUsage] = {}
    skipped: Dict[str, str] = {}

    if "gpt" in models:
        print(f"=== {DISPLAY_NAMES['gpt']} ===")
        start = time.time()
        gpt_trans, gpt_usage, gpt_skip = translate_with_openai(src_lines, lang, series_name, source_type)
        elapsed = time.time() - start
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
        claude_trans, claude_usage, claude_skip = translate_with_claude(src_lines, lang, series_name, source_type)
        elapsed = time.time() - start
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
        gemini_trans, gemini_usage, gemini_skip = translate_with_gemini(src_lines, lang, series_name, source_type)
        elapsed = time.time() - start
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
        deepseek_trans, deepseek_usage, deepseek_skip = translate_with_deepseek(src_lines, lang, series_name, source_type)
        elapsed = time.time() - start
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

    return translations_by_model, usage_by_model

# ============================================================
#  MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe un .ass + vídeo a japonés o chino (Anime-Whisper / BELLE-2), "
            "pulido de puntuación con modelos libres, añade romaji/pinyin y análisis "
            "tipo diccionario opcional y traduce con GPT, Claude, Gemini y DeepSeek. "
            "Cada .ass de salida puede tener:\n"
            "  - línea 1: japonés/chino\n"
            "  - línea 2: romaji/pinyin (si se activa)\n"
            "  - línea 3: análisis diccionario (si se activa)\n"
            "  - última línea: traducción."
        )
    )
    parser.add_argument("ass_in", help="Archivo .ass de entrada (con tiempos sincronizados).")
    parser.add_argument("video_in", nargs="?", default="", help="Vídeo correspondiente (opcional si --skip-asr).")
    parser.add_argument(
        "--out-dir",
        help="Carpeta donde guardar los archivos de salida (.ass, .html). "
             "Por defecto, la carpeta del .ass de entrada.",
        default=None,
    )
    parser.add_argument(
        "--base-name",
        help="Prefijo para los .ass de salida (por defecto, nombre base del .ass de entrada).",
        default=None,
    )
    parser.add_argument(
        "--models",
        help=(
            "Lista de modelos a ejecutar, separados por comas. "
            "Opciones: GPT-5, Claude, Gemini 2.5 Flash, DeepSeek (o también gpt, claude, gemini, deepseek). "
            "Por defecto: gpt,claude,gemini,deepseek"
        ),
        default="GPT-5,Claude,Gemini 2.5 Flash,DeepSeek",
    )
    parser.add_argument(
        "--pad-ms",
        type=int,
        default=0,
        help="Padding en milisegundos al inicio y final de cada línea al recortar el audio.",
    )
    parser.add_argument(
        "--lang",
        choices=["ja", "zh"],
        help="Idioma original del audio o del guion (ja = japonés, zh = chino mandarín). "
             "Si no se indica, se preguntará por consola.",
    )
    parser.add_argument(
        "--series",
        help="Nombre de la serie (por ejemplo, 'Dragon Raja'). "
             "Si no se indica, se preguntará por consola.",
    )
    parser.add_argument(
        "--source-type",
        choices=["Manga", "Manhwa", "Novela ligera", "Nada"],
        help="Tipo de material original (manga, manhwa, novela ligera, nada). "
             "Si no se indica, se preguntará por consola.",
    )
    parser.add_argument(
        "--do-roman-morph",
        action="store_true",
        help="Añadir romaji/pinyin y análisis diccionario en el ASS.",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generar un HTML resumen con original, romanización, análisis diccionario y traducciones.",
    )
    parser.add_argument(
        "--skip-asr",
        action="store_true",
        help=(
            "Omitir la transcripción de audio. Se asume que el .ass ya contiene la "
            "transcripción en japonés o chino en la primera línea de cada subtítulo. "
            "Aun así se puede añadir romaji/pinyin y análisis diccionario (--do-roman-morph) "
            "y hacer las traducciones."
        ),
    )

    args = parser.parse_args()

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
        source_type = args.source_type
    else:
        source_type = ask_source_type()

    # 1) Obtener subs de partida
    if args.skip_asr:
        print("[+] Omitiendo fase de transcripción de audio: se usará el texto ya presente en el .ass.")
        subs = pysubs2.load(ass_in, encoding="utf-8")

        if args.do_roman_morph:
            print("[+] Añadiendo romaji/pinyin y análisis diccionario sobre el guion existente.")
            subs = add_roman_morph_to_subs(subs, lang)
        else:
            print("[+] --do-roman-morph NO está activado: se usará el guion tal cual para la traducción.")

        # Guardamos un intermedio igualmente, para tener copia de trabajo
        asr_suffix = "_ja_asr" if lang == "ja" else "_zh_asr"
        asr_out = os.path.join(out_dir, f"{base_name}{asr_suffix}.ass")
        subs.save(asr_out, encoding="utf-8-sig")
        print(f"[+] Archivo intermedio (sin ASR, solo romanización/diccionario si procede): {asr_out}\n")

    else:
        print("[+] Ejecutando pipeline completo: ASR + puntuación + romaji/pinyin (si procede).")
        subs = transcribe_ass(
            ass_in,
            video_in,
            pad_ms=args.pad_ms,
            lang=lang,
            do_roman_morph=args.do_roman_morph,
        )

        # Guardamos ASS intermedio (asr)
        asr_suffix = "_ja_asr" if lang == "ja" else "_zh_asr"
        asr_out = os.path.join(out_dir, f"{base_name}{asr_suffix}.ass")
        subs.save(asr_out, encoding="utf-8-sig")
        print(f"[+] Archivo intermedio (solo transcripción + puntuación + romanización/diccionario): {asr_out}\n")

    # 2) Traducir
    models = normalize_models_arg(args.models)
    translations_by_model, usage_by_model = process_all_models_with_subs(
        subs,
        lang,
        series_name,
        source_type,
        base_name,
        models,
        out_dir,
    )

    if usage_by_model:
        log_cost_summary(run_id, usage_by_model, series_name, base_name)
        persist_costs_to_supabase(run_id, series_name, base_name, lang, usage_by_model)

    # 3) HTML opcional
    if args.html:
        html_out = os.path.join(out_dir, f"{base_name}_summary.html")
        generate_html(subs, translations_by_model, html_out)


if __name__ == "__main__":
    main()
