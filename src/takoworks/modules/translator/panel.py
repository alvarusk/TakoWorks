from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ...config import save_config


class TranslatorPanel(ttk.Frame):
    def __init__(self, parent, runner, cfg: dict, launch_opts: dict | None = None):
        super().__init__(parent)
        self.runner = runner
        self.cfg = cfg
        self.launch_opts = launch_opts or {}

        last = cfg.setdefault("last", {})
        initial_ass = self.launch_opts.get("ass") or last.get("translator_ass", "")
        initial_glossary = self.launch_opts.get("glossary") or last.get("translator_glossary", "")
        self.ass_var = tk.StringVar(value=initial_ass)
        self.glossary_var = tk.StringVar(value=initial_glossary)

        self._build()

    def _build(self):
        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, padx=10, pady=10)

        r0 = ttk.Frame(frm)
        r0.pack(fill="x", pady=3)
        ttk.Label(r0, text="Input ASS File").pack(side="left")
        ttk.Entry(r0, textvariable=self.ass_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r0, text="Browse", command=self._pick_ass).pack(side="left")

        r1 = ttk.Frame(frm)
        r1.pack(fill="x", pady=3)
        ttk.Label(r1, text="Glossary CSV (optional)").pack(side="left")
        ttk.Entry(r1, textvariable=self.glossary_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r1, text="Browse", command=self._pick_glossary).pack(side="left")

        note = ttk.LabelFrame(frm, text="Notes")
        note.pack(fill="x", pady=8)
        ttk.Label(
            note,
            text=(
                "- Translates only Dialogue text.\n"
                "- ASS tags and line breaks are preserved.\n"
                "- If a glossary CSV is selected, DeepL creates a temporary glossary for the run and removes it afterward."
            ),
            justify="left",
        ).pack(anchor="w", padx=8, pady=6)

        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=10)
        self.run_btn = ttk.Button(btns, text="Run", command=self._run)
        self.run_btn.pack(side="left")
        self.cancel_btn = ttk.Button(btns, text="Cancel", command=self.runner.cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=6)

    def _pick_ass(self):
        p = filedialog.askopenfilename(filetypes=[("ASS/SSA", "*.ass *.ssa"), ("All files", "*.*")])
        if p:
            self.ass_var.set(p)

    def _pick_glossary(self):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if p:
            self.glossary_var.set(p)

    def _run(self):
        if self.runner.is_busy():
            return

        ass_path = self.ass_var.get().strip()
        glossary_path = self.glossary_var.get().strip()

        if not ass_path or not os.path.isfile(ass_path):
            messagebox.showerror("Error", "Select a valid ASS file.")
            return
        if glossary_path and not os.path.isfile(glossary_path):
            messagebox.showerror("Error", "Select a valid glossary CSV file.")
            return

        ass_path = os.path.abspath(ass_path)
        glossary_path = os.path.abspath(glossary_path) if glossary_path else ""
        out_path = None

        self.cfg["last"]["translator_ass"] = ass_path
        self.cfg["last"]["translator_glossary"] = glossary_path
        save_config(self.cfg)

        def job(cancel_event, log):
            from . import core

            nonlocal out_path
            out_path = core._make_output_path(ass_path)
            log(f"ASS: {ass_path}")
            log(f"Glossary: {glossary_path or '(none)'}")
            log(f"Output: {out_path}")
            core.translate_ass_file(
                ass_path,
                glossary_path,
                out_path,
                cancel_event=cancel_event,
                log=log,
            )

        def done(ok, err):
            self.run_btn.configure(state="normal")
            self.cancel_btn.configure(state="disabled")

        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.runner.start("Translator", job, on_done=done)
