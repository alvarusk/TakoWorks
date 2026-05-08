import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from takoworks.modules.transcriber.core import (  # type: ignore
    _build_romanization_system_prompt,
    _normalize_romanization_output,
)


def test_japanese_romanization_keeps_word_spacing():
    assert _normalize_romanization_output("oretachi no danjon ga", "ja") == "oretachi no danjon ga"
    assert _normalize_romanization_output("oretachi   no   danjon   ga", "ja") == "oretachi no danjon ga"


def test_romanization_prompt_requests_spaces_for_japanese():
    prompt = _build_romanization_system_prompt("ja")
    assert "espacios simples" in prompt
    assert "oretachi no danjon ga" in prompt
    assert "kaisoushu wo yuugou saseyagatta no ka..." in prompt
