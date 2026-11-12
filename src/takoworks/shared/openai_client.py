"""Lightweight OpenAI settings shared by translation and final review."""

import os

from ..config import load_config

OPENAI_MODEL = "gpt-6.1-sol"


def create_openai_client():
    from openai import OpenAI

    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        key = str(load_config().get("api_keys", {}).get("openai", "") or "").strip()
    if not key:
        raise RuntimeError("Falta OPENAI_API_KEY (entorno o configuracion de TakoWorks).")
    return OpenAI(api_key=key, timeout=180, max_retries=2)
