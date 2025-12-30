import sys
from pathlib import Path

# Ensure src/ is on sys.path when running from repo root
here = Path(__file__).resolve().parent
src = here / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from takoworks.__main__ import main

if __name__ == "__main__":
    main()
