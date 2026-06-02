from __future__ import annotations

import argparse

from .bootstrap import bootstrap
from .app import run_app


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--tab", default="", help="Initial tab to select in the main window.")
    ap.add_argument("--ass", dest="ass_path", default="", help="ASS file to prefill in Translator.")
    ap.add_argument(
        "--glossary",
        dest="glossary_path",
        default="",
        help="Optional glossary CSV to prefill in Translator.",
    )
    args = ap.parse_args(argv)

    cfg = bootstrap()
    launch_opts = {
        "tab": args.tab,
        "ass": args.ass_path,
        "glossary": args.glossary_path,
    }
    run_app(cfg, launch_opts=launch_opts)


if __name__ == "__main__":
    main()
