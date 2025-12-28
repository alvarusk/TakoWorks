from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from ...config import save_config


class _ConsoleAdapter:
    def __init__(self, log):
        self._log = log

    def insert(self, _pos, msg):
        # core.log_console espera insert(tk.END, msg)
        self._log(str(msg).rstrip("\n"))

    def see(self, *_args):
        pass

    def update_idletasks(self):
        pass

    def configure(self, **_kwargs):
        pass


class CorrectorPanel(ttk.Frame):
    def __init__(self, parent, runner, cfg: dict):
        super().__init__(parent)
        self.runner = runner
        self.cfg = cfg

        self.inp_var = tk.StringVar(value=cfg["last"].get("ass_in", ""))
        self.settings_var = tk.StringVar(value=cfg.get("corrector_settings", ""))

        opts = cfg.get("corrector_options", {})
        self.v_serve = tk.BooleanVar(value=bool(opts.get("serve", False)))
        self.v_no_open = tk.BooleanVar(value=bool(opts.get("no_open", False)))
        self.v_keep = tk.BooleanVar(value=bool(opts.get("keep_intermediate", False)))

        self._build()

    def _build(self):
        frm = ttk.Frame(self); frm.pack(fill="both", expand=True, padx=10, pady=10)

        row = ttk.Frame(frm); row.pack(fill="x", pady=(0, 6))
        ttk.Label(row, text="Input (.ass/.ssa):").pack(anchor="w")
        ent = ttk.Entry(row, textvariable=self.inp_var); ent.pack(fill="x", pady=2)
        ttk.Button(row, text="Browse…", command=self._browse_in).pack(anchor="e")

        row2 = ttk.Frame(frm); row2.pack(fill="x", pady=(0, 8))
        ttk.Label(row2, text="Settings (optional):").pack(anchor="w")
        ent2 = ttk.Entry(row2, textvariable=self.settings_var); ent2.pack(fill="x", pady=2)
        ttk.Button(row2, text="Browse…", command=self._browse_settings).pack(anchor="e")

        box = ttk.Labelframe(frm, text="Options"); box.pack(fill="x", pady=8)
        ttk.Checkbutton(box, text="Corrector (con diccionario)", variable=self.v_serve).pack(anchor="w", padx=8, pady=2)
        ttk.Checkbutton(box, text="No abrir HTML al terminar", variable=self.v_no_open).pack(anchor="w", padx=8, pady=2)
        ttk.Checkbutton(box, text="Keep intermediate files", variable=self.v_keep).pack(anchor="w", padx=8, pady=2)

        btns = ttk.Frame(frm); btns.pack(fill="x", pady=10)
        self.run_btn = ttk.Button(btns, text="Run", command=self._on_run); self.run_btn.pack(side="left")
        self.cancel_btn = ttk.Button(btns, text="Cancel", state="disabled", command=self._on_cancel); self.cancel_btn.pack(side="left", padx=6)

    def _browse_in(self):
        path = filedialog.askopenfilename(filetypes=[("ASS/SSA", "*.ass *.ssa"), ("All", "*.*")])
        if path:
            self.inp_var.set(path)

    def _browse_settings(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if path:
            self.settings_var.set(path)

    def _on_cancel(self):
        self.runner.cancel()

    def _on_run(self):
        inp = self.inp_var.get().strip()
        if not inp:
            messagebox.showerror("Error", "Selecciona un .ass/.ssa.")
            return

        # Save config
        self.cfg["last"]["ass_in"] = inp
        self.cfg["corrector_settings"] = self.settings_var.get().strip()
        self.cfg["corrector_options"] = {
            "serve": bool(self.v_serve.get()),
            "no_open": bool(self.v_no_open.get()),
            "keep_intermediate": bool(self.v_keep.get()),
        }
        save_config(self.cfg)

        def job(cancel_event, log):
            from . import core
            adapter = _ConsoleAdapter(log)
            rc = core.run_corrector(
                inp,
                serve=bool(self.v_serve.get()),
                no_open=bool(self.v_no_open.get()),
                keep_intermediate=bool(self.v_keep.get()),
                settings_path=self.settings_var.get().strip() or None,
                log=log,
            )
            if rc != 0:
                raise RuntimeError(f"Corrector exited with code {rc}")

        def done(ok, err):
            self.run_btn.configure(state="normal")
            self.cancel_btn.configure(state="disabled")

        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.runner.start("Corrector", job, on_done=done)
