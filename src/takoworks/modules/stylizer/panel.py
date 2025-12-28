from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from ...config import save_config


class _ConsoleAdapter:
    # Adapter para reutilizar log_console() del core (que espera un Text-like)
    def __init__(self, log):
        self._log = log

    def configure(self, **kwargs):  # ignore
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


class StylizerPanel(ttk.Frame):
    def __init__(self, parent, runner, cfg: dict):
        super().__init__(parent)
        self.runner = runner
        self.cfg = cfg

        self.input_var = tk.StringVar(value=cfg["last"].get("ass_in", ""))
        self.out_var = tk.StringVar(value=cfg["last"].get("out_dir", ""))

        opts = cfg.get("stylizer_options", {})
        self.v_clean_carteles = tk.BooleanVar(value=bool(opts.get("clean_carteles", False)))
        self.v_clean_comments = tk.BooleanVar(value=bool(opts.get("clean_comments", False)))
        self.v_add_styles = tk.BooleanVar(value=bool(opts.get("add_styles", True)))
        self.v_transform = tk.BooleanVar(value=bool(opts.get("transform_styles", True)))
        self.v_clean_text = tk.BooleanVar(value=bool(opts.get("clean_text", False)))
        self.v_remove_linebreaks = tk.BooleanVar(value=bool(opts.get("remove_linebreaks", False)))

        self._build()

    def _build(self):
        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, padx=10, pady=10)

        # Input
        r0 = ttk.Frame(frm); r0.pack(fill="x", pady=3)
        ttk.Label(r0, text="Input ASS File").pack(side="left")
        ttk.Entry(r0, textvariable=self.input_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r0, text="Browse", command=self._pick_ass).pack(side="left")

        # Output
        r1 = ttk.Frame(frm); r1.pack(fill="x", pady=3)
        ttk.Label(r1, text="Output Folder").pack(side="left")
        ttk.Entry(r1, textvariable=self.out_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r1, text="Browse", command=self._pick_out).pack(side="left")

        # Options
        box = ttk.LabelFrame(frm, text="Options")
        box.pack(fill="x", pady=8)

        ttk.Checkbutton(box, text="Add/replace styles", variable=self.v_add_styles).pack(anchor="w", padx=8, pady=2)
        ttk.Checkbutton(box, text="Transform styles", variable=self.v_transform).pack(anchor="w", padx=8, pady=2)
        ttk.Checkbutton(box, text="Clean typesetting", variable=self.v_clean_carteles).pack(anchor="w", padx=8, pady=2)
        ttk.Checkbutton(box, text="Clean comments", variable=self.v_clean_comments).pack(anchor="w", padx=8, pady=2)
        ttk.Checkbutton(box, text="Remove linebreaks", variable=self.v_remove_linebreaks).pack(anchor="w", padx=8, pady=2)
        ttk.Checkbutton(box, text="Clean text", variable=self.v_clean_text).pack(anchor="w", padx=8, pady=2)

        # Buttons
        btns = ttk.Frame(frm); btns.pack(fill="x", pady=10)
        self.run_btn = ttk.Button(btns, text="Run", command=self._run)
        self.run_btn.pack(side="left")
        self.cancel_btn = ttk.Button(btns, text="Cancel", command=self.runner.cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=6)

    def _pick_ass(self):
        p = filedialog.askopenfilename(filetypes=[("ASS files", "*.ass")])
        if p:
            self.input_var.set(p)
            if not self.out_var.get().strip():
                self.out_var.set(os.path.dirname(p))

    def _pick_out(self):
        p = filedialog.askdirectory()
        if p:
            self.out_var.set(p)

    def _run(self):
        if self.runner.is_busy():
            return

        inp = self.input_var.get().strip()
        out_dir = self.out_var.get().strip() or os.path.dirname(inp)

        if not inp or not os.path.isfile(inp):
            messagebox.showerror("Error", "Select a valid ASS file.")
            return
        if not out_dir or not os.path.isdir(out_dir):
            messagebox.showerror("Error", "Select a valid output folder.")
            return

        # Guarda opciones para que Unworder pueda “aplicar Stylizer igual”
        self.cfg["stylizer_options"] = {
            "clean_carteles": bool(self.v_clean_carteles.get()),
            "clean_comments": bool(self.v_clean_comments.get()),
            "add_styles": bool(self.v_add_styles.get()),
            "transform_styles": bool(self.v_transform.get()),
            "clean_text": bool(self.v_clean_text.get()),
            "remove_linebreaks": bool(self.v_remove_linebreaks.get()),
        }
        self.cfg["last"]["ass_in"] = inp
        self.cfg["last"]["out_dir"] = out_dir
        save_config(self.cfg)

        def job(cancel_event, log):
            from . import core  # tu core.py actual
            adapter = _ConsoleAdapter(log)

            base = os.path.splitext(os.path.basename(inp))[0]
            out_path = os.path.join(out_dir, f"{base}_OUT.ass")

            lines = core.read_file(inp)

            if self.v_add_styles.get():
                lines = core.replace_styles_section(lines, console=adapter)

            lines = core.process_events(
                lines,
                clean_carteles=bool(self.v_clean_carteles.get()),
                clean_comments=bool(self.v_clean_comments.get()),
                transform_styles=bool(self.v_transform.get()),
                clean_text=bool(self.v_clean_text.get()),
                remove_linebreaks=bool(self.v_remove_linebreaks.get()),
                console=adapter,
            )

            core.write_file(out_path, lines)
            log(f"[OK] Saved in: {out_path}")

        def done(ok, err):
            self.run_btn.configure(state="normal")
            self.cancel_btn.configure(state="disabled")

        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.runner.start("Stylizer", job, on_done=done)
