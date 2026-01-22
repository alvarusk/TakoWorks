from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from ...config import save_config


class TransfererPanel(ttk.Frame):
    def __init__(self, parent, runner, cfg: dict):
        super().__init__(parent)
        self.runner = runner
        self.cfg = cfg

        last = cfg.get("last", {})
        self.src_var = tk.StringVar(value=last.get("transferer_src", ""))
        self.dst_var = tk.StringVar(value=last.get("transferer_dst", ""))

        self._build()

    def _build(self):
        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, padx=10, pady=10)

        r0 = ttk.Frame(frm); r0.pack(fill="x", pady=3)
        ttk.Label(r0, text="Source script (with text)").pack(side="left")
        ttk.Entry(r0, textvariable=self.src_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r0, text="Browse", command=self._pick_src).pack(side="left")

        r1 = ttk.Frame(frm); r1.pack(fill="x", pady=3)
        ttk.Label(r1, text="Timed script (empty lines)").pack(side="left")
        ttk.Entry(r1, textvariable=self.dst_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r1, text="Browse", command=self._pick_dst).pack(side="left")

        btns = ttk.Frame(frm); btns.pack(fill="x", pady=10)
        self.run_btn = ttk.Button(btns, text="Run", command=self._run)
        self.run_btn.pack(side="left")
        self.cancel_btn = ttk.Button(btns, text="Cancel", command=self.runner.cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=6)

    def _pick_src(self):
        p = filedialog.askopenfilename(filetypes=[("ASS/SSA", "*.ass *.ssa"), ("All files", "*.*")])
        if p:
            self.src_var.set(p)

    def _pick_dst(self):
        p = filedialog.askopenfilename(filetypes=[("ASS/SSA", "*.ass *.ssa"), ("All files", "*.*")])
        if p:
            self.dst_var.set(p)

    def _default_out_path(self, dst_path: str) -> str:
        out_dir = os.path.dirname(dst_path) or "."
        base = os.path.splitext(os.path.basename(dst_path))[0]
        out_base = base if base.lower().endswith("_merged") else f"{base}_merged"
        out_path = os.path.join(out_dir, f"{out_base}.ass")
        if os.path.normcase(out_path) == os.path.normcase(dst_path):
            out_path = os.path.join(out_dir, f"{base}_transfer.ass")
        return out_path

    def _run(self):
        if self.runner.is_busy():
            return

        src = self.src_var.get().strip()
        dst = self.dst_var.get().strip()

        if not src or not os.path.isfile(src):
            messagebox.showerror("Error", "Select a valid source ASS file.")
            return
        if not dst or not os.path.isfile(dst):
            messagebox.showerror("Error", "Select a valid timed ASS file.")
            return

        src_path = os.path.abspath(src)
        dst_path = os.path.abspath(dst)

        if os.path.normcase(src_path) == os.path.normcase(dst_path):
            messagebox.showerror("Error", "Source and timed files must be different.")
            return

        out_path = self._default_out_path(dst_path)

        self.cfg.setdefault("last", {})["transferer_src"] = src_path
        self.cfg.setdefault("last", {})["transferer_dst"] = dst_path
        save_config(self.cfg)

        def job(cancel_event, log):
            from . import transferer as core

            if cancel_event.is_set():
                raise RuntimeError("Cancelado")

            log(f"Source: {src_path}")
            log(f"Timed: {dst_path}")
            log(f"Output: {out_path}")

            src_n, dst_n, direct_unique, fallback_n = core.build_merged_ass(
                src_path,
                dst_path,
                out_path,
            )

            log(f"[stats] src_useful={src_n} dst_useful={dst_n} direct_used_unique={direct_unique} fallback_merged={fallback_n}")
            log("[note] Target timings (Start/End) are NOT modified.")
            log('[note] Ignored: Name=="CARTEL" or Style startswith "Cart_".')

        def done(ok, err):
            self.run_btn.configure(state="normal")
            self.cancel_btn.configure(state="disabled")

        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.runner.start("Transferer", job, on_done=done)
