from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from ...config import save_config


class _ConsoleAdapter:
    def __init__(self, log):
        self._log = log

    # stylizer.core.log_console llama a configure(state="normal")
    def configure(self, **_kwargs):
        pass

    # por si algún sitio usa .config(...)
    def config(self, **_kwargs):
        pass

    def insert(self, *_args):
        txt = str(_args[-1])
        for line in txt.splitlines():
            if line.strip():
                self._log(line)

    def see(self, *_args):
        pass

    def update_idletasks(self):
        pass


class UnworderPanel(ttk.Frame):
    def __init__(self, parent, runner, cfg: dict):
        super().__init__(parent)
        self.runner = runner
        self.cfg = cfg

        self.word_var = tk.StringVar(value=cfg["last"].get("word_in", ""))
        self.out_var = tk.StringVar(value=cfg["last"].get("out_dir", ""))

        self.v_apply_stylizer = tk.BooleanVar(value=True)

        self._build()

    def _build(self):
        frm = ttk.Frame(self); frm.pack(fill="both", expand=True, padx=10, pady=10)

        r0 = ttk.Frame(frm); r0.pack(fill="x", pady=3)
        ttk.Label(r0, text="Input Script (DOCX or TXT)").pack(side="left")
        ttk.Entry(r0, textvariable=self.word_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r0, text="Browse", command=self._pick_word).pack(side="left")

        r1 = ttk.Frame(frm); r1.pack(fill="x", pady=3)
        ttk.Label(r1, text="Output Folder").pack(side="left")
        ttk.Entry(r1, textvariable=self.out_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r1, text="Browse", command=self._pick_out).pack(side="left")

        box = ttk.LabelFrame(frm, text="Post-processing")
        box.pack(fill="x", pady=8)
        ttk.Checkbutton(
            box,
            text="Aplicar Stylizer to output ASS (using the last stored options in Stylizer)",
            variable=self.v_apply_stylizer
        ).pack(anchor="w", padx=8, pady=4)

        btns = ttk.Frame(frm); btns.pack(fill="x", pady=10)
        self.run_btn = ttk.Button(btns, text="Run", command=self._run)
        self.run_btn.pack(side="left")
        self.cancel_btn = ttk.Button(btns, text="Cancel", command=self.runner.cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=6)

    def _pick_word(self):
        p = filedialog.askopenfilename(filetypes=[("Word/Texto", "*.docx *.txt"), ("All files", "*.*")])
        if p:
            self.word_var.set(p)
            if not self.out_var.get().strip():
                self.out_var.set(os.path.dirname(p))

    def _pick_out(self):
        p = filedialog.askdirectory()
        if p:
            self.out_var.set(p)

    def _run(self):
        if self.runner.is_busy():
            return

        inp = self.word_var.get().strip()
        out_dir = self.out_var.get().strip() or os.path.dirname(inp)

        if not inp or not os.path.isfile(inp):
            messagebox.showerror("Error", "Select a valid Word/Text file.")
            return
        if not out_dir or not os.path.isdir(out_dir):
            messagebox.showerror("Error", "Select a valid output folder.")
            return

        self.cfg["last"]["word_in"] = inp
        self.cfg["last"]["out_dir"] = out_dir
        save_config(self.cfg)

        def job(cancel_event, log):
            from . import core
            adapter = _ConsoleAdapter(log)

            base = os.path.splitext(os.path.basename(inp))[0]
            out_ass = os.path.join(out_dir, f"{base}.ass")

            log(f"Input: {inp}")
            lines = core.read_lines_from_doc(inp, adapter)
            if not lines:
                raise RuntimeError("No valid lines found (Actor: Texto).")

            ass_content = core.build_ass_content(lines)
            with open(out_ass, "w", encoding="utf-8-sig", errors="replace") as f:
                f.write(ass_content)
            log(f"[OK] ASS generado: {out_ass}")

            if self.v_apply_stylizer.get():
                # Aplicar exactamente lo mismo que Stylizer (últimas opciones guardadas)
                from ..stylizer import core as styl_core
                opts = self.cfg.get("stylizer_options", {})
                base2 = os.path.splitext(os.path.basename(out_ass))[0]
                out_ass2 = os.path.join(out_dir, f"{base2}_OUT.ass")

                lines2 = styl_core.read_file(out_ass)
                if opts.get("add_styles", True):
                    lines2 = styl_core.replace_styles_section(lines2, console=adapter)

                lines2 = styl_core.process_events(
                    lines2,
                    clean_carteles=bool(opts.get("clean_carteles", False)),
                    clean_comments=bool(opts.get("clean_comments", False)),
                    transform_styles=bool(opts.get("transform_styles", True)),
                    clean_text=bool(opts.get("clean_text", False)),
                    console=adapter,
                )
                styl_core.write_file(out_ass2, lines2)
                log(f"[OK] Stylizer applied: {out_ass2}")

        def done(ok, err):
            self.run_btn.configure(state="normal")
            self.cancel_btn.configure(state="disabled")

        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.runner.start("Unworder", job, on_done=done)
