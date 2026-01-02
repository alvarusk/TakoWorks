import argparse
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INIT_PATH = ROOT / "src" / "takoworks" / "__init__.py"
README_PATH = ROOT / "README.md"


def get_branch() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
        return out
    except Exception:
        return ""


def read_version() -> str:
    txt = INIT_PATH.read_text(encoding="utf-8")
    m = re.search(r'__version__\s*=\s*"([^"]+)"', txt)
    if not m:
        raise RuntimeError("No se encontró __version__ en __init__.py")
    return m.group(1)


def write_version(new_version: str) -> None:
    # __init__.py
    txt = INIT_PATH.read_text(encoding="utf-8")
    def _repl_init(m: re.Match) -> str:
        return f'{m.group(1)}{new_version}{m.group(3)}'
    txt = re.sub(r'(__version__\s*=\s*")([^"]+)(")', _repl_init, txt)
    INIT_PATH.write_text(txt, encoding="utf-8")

    # README (encabezado)
    if README_PATH.exists():
        rtxt = README_PATH.read_text(encoding="utf-8")
        def _repl_readme(m: re.Match) -> str:
            return f'{m.group(1)}{new_version}{m.group(3)}'
        rtxt = re.sub(r"(TakoWorks \(v)([^)]+)(\))", _repl_readme, rtxt, count=1)
        README_PATH.write_text(rtxt, encoding="utf-8")


def bump(version: str, mode: str) -> str:
    parts = [int(p) for p in version.split(".")]
    if len(parts) == 2:
        parts.append(0)
    if len(parts) != 3:
        raise RuntimeError(f"Versión no válida: {version}")
    major, minor, patch = parts
    if mode == "minor":
        minor += 1
        patch = 0
    else:
        patch += 1
    return f"{major}.{minor}.{patch}"


def main():
    parser = argparse.ArgumentParser(description="Bump de versión TakoWorks.")
    parser.add_argument("--mode", choices=["auto", "minor", "patch"], default="auto")
    parser.add_argument("--dry-run", action="store_true", help="No escribe archivos, solo muestra nueva versión.")
    args = parser.parse_args()

    if os.getenv("SKIP_VERSION_BUMP"):
        print("SKIP_VERSION_BUMP está definido; no se hace bump.")
        return

    branch = get_branch()
    mode = args.mode
    if mode == "auto":
        if branch.endswith("main"):
            mode = "minor"
        else:
            mode = "patch"

    current = read_version()
    new_version = bump(current, mode)

    if args.dry_run:
        print(f"[dry-run] {current} -> {new_version} (mode={mode}, branch={branch or 'unknown'})")
        return

    write_version(new_version)
    print(f"Bumped version {current} -> {new_version} (mode={mode}, branch={branch or 'unknown'})")
    print("Recuerda commitear los cambios.")


if __name__ == "__main__":
    main()
