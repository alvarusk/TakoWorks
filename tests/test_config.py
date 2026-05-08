import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from takoworks import paths  # type: ignore
from takoworks.config import load_config, save_config  # type: ignore


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
