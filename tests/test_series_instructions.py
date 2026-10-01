from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from takoworks.modules.transcriber.core import build_system_prompt  # type: ignore
from takoworks.shared.series_instructions import (  # type: ignore
    markdown_terminology_pairs,
    terminology_pairs_csv,
)


def test_markdown_terminology_is_available_to_transcriber_prompt():
    instructions = "| Original | Español |\n|---|---|\n| Star Core | Núcleo Estelar |"
    prompt = build_system_prompt("ja", "Serie", "None", instructions)

    assert "Star Core" in prompt
    assert "Núcleo Estelar" in prompt


def test_markdown_terminology_extracts_tables_and_bullets():
    markdown = """# Terminología

| Original | Español |
|---|---|
| Star Core | Núcleo Estelar |

- Void Fleet -> Flota del Vacío
"""

    assert markdown_terminology_pairs(markdown) == [
        ("Star Core", "Núcleo Estelar"),
        ("Void Fleet", "Flota del Vacío"),
    ]
    assert "Star Core,Núcleo Estelar" in terminology_pairs_csv(markdown)
