import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from takoworks.modules.transcriber.ass_utils import (  # type: ignore
    _ass_hide,
    _ass_hide_prefix,
    _ass_sanitize_braces,
    _ass_unsanitize_braces,
)


def test_sanitize_and_unsanitize_roundtrip():
    original = "{hola} {mundo}"
    sanitized = _ass_sanitize_braces(original)
    assert sanitized != original
    restored = _ass_unsanitize_braces(sanitized)
    assert restored == original


def test_ass_hide_wraps_with_braces():
    hidden = _ass_hide("texto")
    assert hidden.startswith("{") and hidden.endswith("}")
    assert "texto" in hidden


def test_hide_prefix_hides_multiple_lines():
    existing = "line1\\Nline2"
    result = _ass_hide_prefix(existing)
    # Should wrap each sub-line in braces and keep separator
    assert result.count("{") == 2
    assert "\\N" in result
