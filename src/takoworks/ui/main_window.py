from __future__ import annotations

from tkinter import ttk

from ..shared.workers import TaskRunner
from .console_widget import ConsoleFrame

from ..modules.settings.panel import SettingsPanel
from ..modules.transcriber.panel import TranscriberPanel
from ..modules.translator.panel import TranslatorPanel
from ..modules.scanner.panel import ScannerPanel

class MainWindow(ttk.Frame):
    def __init__(self, parent, cfg: dict, launch_opts: dict | None = None):
        super().__init__(parent)
        self.cfg = cfg
        self.launch_opts = launch_opts or {}

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
        parent.after(300, self._restore_splitter)

        self.runner = TaskRunner(parent, self.console.write)

        self.notebook.add(TranscriberPanel(self.notebook, self.runner, self.cfg), text="Transcriber")
        self.notebook.add(
            TranslatorPanel(self.notebook, self.runner, self.cfg, launch_opts=self.launch_opts),
            text="Translator",
        )
        self.notebook.add(ScannerPanel(self.notebook, self.runner, self.cfg), text="Scanner")
        self.notebook.add(SettingsPanel(self.notebook, self.runner, self.cfg), text="Settings")
        self._select_initial_tab()

    def _select_initial_tab(self):
        target = str(self.launch_opts.get("tab", "") or "").strip().lower()
        if not target:
            self.notebook.select(self.notebook.tabs()[0])
            return

        mapping = {
            "transcriber": "Transcriber",
            "translator": "Translator",
            "scanner": "Scanner",
            "settings": "Settings",
        }
        wanted = mapping.get(target)
        if not wanted:
            self.notebook.select(self.notebook.tabs()[0])
            return

        for tab_id in self.notebook.tabs():
            try:
                if str(self.notebook.tab(tab_id, "text")) == wanted:
                    self.notebook.select(tab_id)
                    return
            except Exception:
                continue

        self.notebook.select(self.notebook.tabs()[0])

    def _restore_splitter(self):
        if self.paned.winfo_height() < 200:
            self.after(150, self._restore_splitter)
            return

        # En el arranque, dejamos la consola en ~1/3 de la ventana.
        # Si el valor guardado es razonable, lo respetamos; si no, usamos el valor por defecto.
        self.winfo_toplevel().update_idletasks()
        saved = int(self.cfg.get("last", {}).get("splitter_pos", 0) or 0)
        h = max(300, self.winfo_toplevel().winfo_height(), self.winfo_toplevel().winfo_reqheight())
        default_pos = int(h * 0.67)
        min_pos = int(h * 0.20)
        max_pos = int(h * 0.85)
        pos = saved if min_pos <= saved <= max_pos else default_pos

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
