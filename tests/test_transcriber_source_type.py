import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from takoworks.modules.transcriber.source_type import (  # type: ignore
    describe_source_type,
    normalize_source_type,
)


def test_normalize_source_type_accepts_spanish_and_canonical_values():
    assert normalize_source_type("Novela ligera") == "Light novel"
    assert normalize_source_type("Nada") == "None"
    assert normalize_source_type("Light novel") == "Light novel"
    assert normalize_source_type("None") == "None"


def test_describe_source_type_uses_normalized_values():
    assert "light novel" in describe_source_type("Novela ligera").lower()
    assert "manga" in describe_source_type("Manga").lower()
