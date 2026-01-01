from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from pathlib import Path

from .ui.main_window import MainWindow
from .paths import app_root
from . import __version__


def run_app(cfg: dict) -> None:
    root = tk.Tk()

    icon_path = Path(app_root()) / "assets" / "takoworks_big.ico"
    if icon_path.exists():
        try:
            root.iconbitmap(default=str(icon_path))
        except Exception:
            pass

    root.title(f"TakoWorks v{__version__}")
    root.geometry("980x720")

    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass

    MainWindow(root, cfg).pack(fill="both", expand=True)
    root.mainloop()
