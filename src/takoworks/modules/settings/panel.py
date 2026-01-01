from __future__ import annotations

import os
from tkinter import ttk, filedialog, messagebox
import tkinter as tk
import subprocess
from pathlib import Path

from ...config import save_config
from ... import __version__


def _add_to_path(*dirs: str) -> None:
    current = os.environ.get("PATH", "")
    parts = [p for p in current.split(os.pathsep) if p]
    for d in dirs:
        if d and d not in parts:
            parts.insert(0, d)
    os.environ["PATH"] = os.pathsep.join(parts)


class SettingsPanel(ttk.Frame):
    def __init__(self, parent, runner, cfg: dict):
        super().__init__(parent)
        self.runner = runner
        self.cfg = cfg

        self.jpdict_var = tk.StringVar(value=cfg.get("jpdict_dir", ""))
        self.cndict_var = tk.StringVar(value=cfg.get("cndict_dir", ""))
        self.ffmpeg_var = tk.StringVar(value=cfg.get("ffmpeg_dir", ""))
        self.yomitoku_var = tk.StringVar(value=cfg.get("yomitoku_dir", ""))

        self._build()

    def _build(self):
        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(frm, text="Paths (stored in config.json)").pack(anchor="w", pady=(0, 8))

        self._row(frm, "Japanese Dictionary Folder", self.jpdict_var, self._pick_dir)
        self._row(frm, "SChinese Dictionary Folder", self.cndict_var, self._pick_dir)
        self._row(frm, "FFMPEG Folder (ffmpeg.exe)", self.ffmpeg_var, self._pick_dir)
        self._row(frm, "YomiToku Folder (yomitoku.exe)", self.yomitoku_var, self._pick_dir)

        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=10)

        ttk.Button(btns, text="Save & Apply", command=self._save_apply).pack(side="left")
        ttk.Button(btns, text="Apply & Don't Save", command=self._apply_only).pack(side="left", padx=6)

        ttk.Separator(frm).pack(fill="x", pady=10)

        ttk.Label(frm, text="Notes:").pack(anchor="w")
        ttk.Label(
            frm,
            text="- Transcriber reads YOMI_JA_DIR/YOMI_ZH_DIR when importing.\n"
                 "- As it's imported when run, changing here before the run works.\n"
                 "- PATH is applied to sub-processes (ffmpeg/yomitoku).",
            justify="left"
        ).pack(anchor="w", pady=6)

        ttk.Separator(frm).pack(fill="x", pady=10)

        ttk.Label(frm, text=f"Updates (current v{__version__})").pack(anchor="w")
        ttk.Button(frm, text="Check updates (git pull origin <branch>)", command=self._check_updates).pack(anchor="w", pady=4)

    def _row(self, parent, label, var, browse_cmd):
        r = ttk.Frame(parent)
        r.pack(fill="x", pady=3)
        ttk.Label(r, text=label).pack(side="left")
        ttk.Entry(r, textvariable=var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r, text="Browse", command=lambda: browse_cmd(var)).pack(side="left")

    def _pick_dir(self, var):
        p = filedialog.askdirectory()
        if p:
            var.set(p)

    def _apply_only(self):
        self._apply_env()

    def _save_apply(self):
        self.cfg["jpdict_dir"] = self.jpdict_var.get().strip()
        self.cfg["cndict_dir"] = self.cndict_var.get().strip()
        self.cfg["ffmpeg_dir"] = self.ffmpeg_var.get().strip()
        self.cfg["yomitoku_dir"] = self.yomitoku_var.get().strip()

        save_config(self.cfg)
        self._apply_env()
        messagebox.showinfo("OK", "Settings saved and applied.")

    def _apply_env(self):
        jp = self.jpdict_var.get().strip()
        zh = self.cndict_var.get().strip()
        ff = self.ffmpeg_var.get().strip()
        yo = self.yomitoku_var.get().strip()

        if jp:
            os.environ["YOMI_JA_DIR"] = jp
        if zh:
            os.environ["YOMI_ZH_DIR"] = zh

        _add_to_path(ff, yo)

    def _check_updates(self):
        repo_root = Path(__file__).resolve().parents[4]

        def _git(args):
            res = subprocess.run(["git"] + args, cwd=repo_root, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(res.stderr.strip() or res.stdout.strip() or "git error")
            return res.stdout.strip()

        try:
            _git(["rev-parse", "--is-inside-work-tree"])
        except Exception as e:
            messagebox.showerror("Update", f"No es un repo git o git no está disponible.\n\n{e}")
            return

        try:
            _git(["fetch", "origin"])
            branch = _git(["rev-parse", "--abbrev-ref", "HEAD"]) or "main"
            remote = f"origin/{branch}"
            pending = _git(["log", "HEAD.." + remote, "--oneline"])
            if not pending:
                messagebox.showinfo("Update", "Ya estás al día con " + remote)
                return
            show = "\n".join(pending.splitlines()[:10])
            if messagebox.askyesno("Update", f"HAY cambios en {remote}:\n\n{show}\n\n¿Hacer git pull origin {branch}?"):
                _git(["pull", "origin", branch])
                messagebox.showinfo("Update", "Actualizado. Reinicia TakoWorks para aplicar cambios.")
        except Exception as e:
            messagebox.showerror("Update", f"Error al actualizar:\n\n{e}")
