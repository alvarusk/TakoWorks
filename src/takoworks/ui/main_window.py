from __future__ import annotations

from tkinter import ttk

from ..shared.workers import TaskRunner
from .console_widget import ConsoleFrame

from ..modules.settings.panel import SettingsPanel
from ..modules.transcriber.panel import TranscriberPanel
from ..modules.stylizer.panel import StylizerPanel
from ..modules.unworder.panel import UnworderPanel
from ..modules.scanner.panel import ScannerPanel
from ..modules.corrector.panel import CorrectorPanel

class MainWindow(ttk.Frame):
    def __init__(self, parent, cfg: dict):
        super().__init__(parent)
        self.cfg = cfg

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Splitter vertical: arriba tabs, abajo consola
        self.paned = ttk.PanedWindow(self, orient="vertical")
        self.paned.grid(row=0, column=0, sticky="nsew")

        self.notebook = ttk.Notebook(self.paned)
        self.console = ConsoleFrame(self.paned)

        # Añadimos al paned con pesos (stretch)
        self.paned.add(self.notebook, weight=4)
        self.paned.add(self.console, weight=1)

        # Opcional: arrancar con consola más alta (en px)
        # parent.after(50, lambda: self.paned.sashpos(0, int(parent.winfo_height() * 0.70)))
        
        self._restoring = True
        parent.after(100, self._restore_splitter)

        self.runner = TaskRunner(parent, self.console.write)

        self.notebook.add(SettingsPanel(self.notebook, self.runner, self.cfg), text="Settings")
        self.notebook.add(TranscriberPanel(self.notebook, self.runner, self.cfg), text="Transcriber")
        self.notebook.add(StylizerPanel(self.notebook, self.runner, self.cfg), text="Stylizer")
        self.notebook.add(UnworderPanel(self.notebook, self.runner, self.cfg), text="Unworder")
        self.notebook.add(ScannerPanel(self.notebook, self.runner, self.cfg), text="Scanner")
        self.notebook.add(CorrectorPanel(self.notebook, self.runner, self.cfg), text="Corrector")

    def _restore_splitter(self):
        # Si hay valor guardado, úsalo; si no, un 70% para tabs
        saved = int(self.cfg.get("last", {}).get("splitter_pos", 0) or 0)
        h = max(300, self.winfo_toplevel().winfo_height())
        pos = saved if saved > 0 else int(h * 0.70)

        try:
            self.paned.sashpos(0, pos)
        except Exception:
            pass

        self._restoring = False

        # Bind para guardar cuando el usuario suelte el ratón
        self.paned.bind("<ButtonRelease-1>", self._on_splitter_release)
    
    def _on_splitter_release(self, _event=None):
        if getattr(self, "_restoring", False):
            return
        try:
            pos = int(self.paned.sashpos(0))
        except Exception:
            return

        self.cfg.setdefault("last", {})["splitter_pos"] = pos
        from ..config import save_config
        save_config(self.cfg)
