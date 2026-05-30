import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from takoworks import paths  # type: ignore
from takoworks.config import load_config, save_config, save_local_config  # type: ignore


def test_series_history_defaults_and_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "app_root", lambda: tmp_path)

    cfg = load_config()
    assert cfg["series_history"] == []
    assert cfg["last"]["series"] == ""

    cfg["last"]["series"] = "The Beginning After The End"
    cfg["series_history"] = ["The Beginning After The End", "Solo Leveling"]
    save_config(cfg)

    reloaded = load_config()
    assert reloaded["last"]["series"] == "The Beginning After The End"
    assert reloaded["series_history"] == ["The Beginning After The End", "Solo Leveling"]


def test_deepl_key_stays_in_local_config(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "app_root", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData"))

    cfg = load_config()
    cfg["api_keys"]["deepl"] = "test-key:fx"
    save_config(cfg)

    portable_text = (tmp_path / "config.json").read_text(encoding="utf-8")
    assert "test-key:fx" not in portable_text

    save_local_config({"api_keys": {"deepl": "test-key:fx"}})
    local_path = tmp_path / "AppData" / "TakoWorks" / "config.local.json"
    assert local_path.exists()
    assert "test-key:fx" in local_path.read_text(encoding="utf-8")
