"""Configured token prices and cost estimates, without loading transcription models."""

import os

from .. import config as app_config

DEFAULT_PRICE_PER_1K = {
    "gpt": {"input": 0.002, "output": 0.01},
    "claude": {"input": 0.004, "output": 0.02},
    "context_note": {"input": 0.00075, "output": 0.00375},
    "gemini": {"input": 0.00075, "output": 0.00375},
    "deepseek": {"input": 0.00014, "output": 0.00028},
}


def _read_price(model_key, kind, default):
    env_name = f"COST_{model_key.upper()}_{kind.upper()}_PER_1K"
    raw = os.getenv(env_name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except Exception:
        print(f"[Costes] Valor no valido en {env_name}: {raw}. Se usa {default}.")
        return default


def load_price_table():
    cfg_costs = {}
    try:
        raw = app_config.load_config().get("cost_per_1k", {})
        if isinstance(raw, dict):
            cfg_costs = raw
    except Exception as exc:
        print(f"[Costes] No se pudo leer cost_per_1k de config.json: {exc}")
    table = {}
    for key, defaults in DEFAULT_PRICE_PER_1K.items():
        entry = cfg_costs.get(key, {})
        table[key] = {
            kind: _read_price(key, kind, entry.get(kind, default) if isinstance(entry, dict) else default)
            for kind, default in defaults.items()
        }
    return table


def calculate_cost(prices, prompt_tokens, completion_tokens):
    return (prompt_tokens * float(prices.get("input", 0.0) or 0.0)
            + completion_tokens * float(prices.get("output", 0.0) or 0.0)) / 1000.0
