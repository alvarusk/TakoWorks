from __future__ import annotations

from .bootstrap import bootstrap
from .app import run_app


def main():
    cfg = bootstrap()
    run_app(cfg)


if __name__ == "__main__":
    main()
