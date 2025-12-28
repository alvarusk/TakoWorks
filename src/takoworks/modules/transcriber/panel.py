from __future__ import annotations

import inspect
import io
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from ...config import save_config


class _LogWriter(io.TextIOBase):
    def __init__(self, log):
        self._log = log
        self._buf = ""

    def write(self, s):
        if not s:
            return 0
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._log(line)
        return len(s)

    def flush(self):
        if self._buf.strip():
            self._log(self._buf.strip())
        self._buf = ""


class TranscriberPanel(ttk.Frame):
    def __init__(self, parent, runner, cfg: dict):
        super().__init__(parent)
        self.runner = runner
        self.cfg = cfg

        self.ass_var = tk.StringVar(value=cfg["last"].get("ass_in", ""))
        self.video_var = tk.StringVar(value=cfg["last"].get("video_in", ""))
        self.out_var = tk.StringVar(value=cfg["last"].get("out_dir", ""))

        self.lang_var = tk.StringVar(value="ja")
        self.series_var = tk.StringVar(value="")
        self.source_var = tk.StringVar(value="nada")

        self.v_skip_asr = tk.BooleanVar(value=False)
        self.v_do_roman = tk.BooleanVar(value=True)
        self.v_html = tk.BooleanVar(value=False)

        self.v_gpt = tk.BooleanVar(value=True)
        self.v_claude = tk.BooleanVar(value=True)
        self.v_gemini = tk.BooleanVar(value=True)
        self.v_deepseek = tk.BooleanVar(value=True)

        self._build()
        self._toggle_video()

    def _build(self):
        frm = ttk.Frame(self); frm.pack(fill="both", expand=True, padx=10, pady=10)

        r0 = ttk.Frame(frm); r0.pack(fill="x", pady=3)
        ttk.Label(r0, text="Input ASS File").pack(side="left")
        ttk.Entry(r0, textvariable=self.ass_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r0, text="Browse", command=self._pick_ass).pack(side="left")

        r1 = ttk.Frame(frm); r1.pack(fill="x", pady=3)
        ttk.Label(r1, text="Video").pack(side="left")
        self.video_entry = ttk.Entry(r1, textvariable=self.video_var)
        self.video_entry.pack(side="left", fill="x", expand=True, padx=6)
        self.video_btn = ttk.Button(r1, text="Browse", command=self._pick_video)
        self.video_btn.pack(side="left")

        r2 = ttk.Frame(frm); r2.pack(fill="x", pady=3)
        ttk.Label(r2, text="Output Folder").pack(side="left")
        ttk.Entry(r2, textvariable=self.out_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r2, text="Browse", command=self._pick_out).pack(side="left")

        meta = ttk.LabelFrame(frm, text="Metadata")
        meta.pack(fill="x", pady=8)

        rr = ttk.Frame(meta); rr.pack(fill="x", pady=3, padx=8)
        ttk.Label(rr, text="Language").pack(side="left")
        ttk.Combobox(rr, textvariable=self.lang_var, values=["ja", "zh"], width=6, state="readonly").pack(side="left", padx=6)
        ttk.Label(rr, text="Source").pack(side="left", padx=(12, 0))
        ttk.Combobox(rr, textvariable=self.source_var, values=["Manga", "Manhwa", "Novela ligera", "Nada"], width=12, state="readonly").pack(side="left", padx=6)
        ttk.Label(rr, text="Series").pack(side="left", padx=(12, 0))
        ttk.Entry(rr, textvariable=self.series_var).pack(side="left", fill="x", expand=True, padx=6)

        opts = ttk.LabelFrame(frm, text="Options")
        opts.pack(fill="x", pady=8)

        ttk.Checkbutton(opts, text="Ignore transcription (skip ASR; no video required)", variable=self.v_skip_asr, command=self._toggle_video).pack(anchor="w", padx=8, pady=2)
        ttk.Checkbutton(opts, text="Add romaji/pinyin + dictionary", variable=self.v_do_roman).pack(anchor="w", padx=8, pady=2)
        ttk.Checkbutton(opts, text="Generate summary HTML", variable=self.v_html).pack(anchor="w", padx=8, pady=2)

        models = ttk.LabelFrame(frm, text="Translation Models")
        models.pack(fill="x", pady=8)

        rowm = ttk.Frame(models); rowm.pack(anchor="w", padx=8, pady=4)
        ttk.Checkbutton(rowm, text="GPT-5", variable=self.v_gpt).pack(side="left", padx=6)
        ttk.Checkbutton(rowm, text="Claude", variable=self.v_claude).pack(side="left", padx=6)
        ttk.Checkbutton(rowm, text="Gemini 2.5 Flash", variable=self.v_gemini).pack(side="left", padx=6)
        ttk.Checkbutton(rowm, text="DeepSeek", variable=self.v_deepseek).pack(side="left", padx=6)

        btns = ttk.Frame(frm); btns.pack(fill="x", pady=10)
        self.run_btn = ttk.Button(btns, text="Run", command=self._run)
        self.run_btn.pack(side="left")
        self.cancel_btn = ttk.Button(btns, text="Cancel", command=self.runner.cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=6)

    def _toggle_video(self):
        skip = bool(self.v_skip_asr.get())
        state = "disabled" if skip else "normal"
        self.video_entry.configure(state=state)
        self.video_btn.configure(state=state)

    def _pick_ass(self):
        p = filedialog.askopenfilename(filetypes=[("ASS files", "*.ass")])
        if p:
            self.ass_var.set(p)
            if not self.out_var.get().strip():
                self.out_var.set(os.path.dirname(p))

    def _pick_video(self):
        p = filedialog.askopenfilename(filetypes=[("Video", "*.mkv *.mp4 *.avi *.mov *.ts"), ("All files", "*.*")])
        if p:
            self.video_var.set(p)

    def _pick_out(self):
        p = filedialog.askdirectory()
        if p:
            self.out_var.set(p)

    def _run(self):
        if self.runner.is_busy():
            return

        ass_in = self.ass_var.get().strip()
        if not ass_in or not os.path.isfile(ass_in):
            messagebox.showerror("Error", "Select a valid ASS file.")
            return

        skip_asr = bool(self.v_skip_asr.get())
        video_in = self.video_var.get().strip()
        if not skip_asr and (not video_in or not os.path.isfile(video_in)):
            messagebox.showerror("Error", "Select a valid video or check Ignore transcription.")
            return

        out_dir = self.out_var.get().strip() or os.path.dirname(ass_in)

        models = []
        if self.v_gpt.get(): models.append("GPT-5")
        if self.v_claude.get(): models.append("Claude")
        if self.v_gemini.get(): models.append("Gemini 2.5 Flash")
        if self.v_deepseek.get(): models.append("DeepSeek")
        models_str = ",".join(models)  # puede ser ""

        self.cfg["last"]["ass_in"] = ass_in
        self.cfg["last"]["video_in"] = video_in
        self.cfg["last"]["out_dir"] = out_dir
        save_config(self.cfg)

        def job(cancel_event, log):
            from . import core

            argv = [ass_in]
            # Si tu transcriber core aún exige video posicional, ponemos algo “dummy”
            argv.append(video_in if video_in else ass_in)

            argv += ["--out-dir", out_dir]
            argv += ["--lang", self.lang_var.get()]
            argv += ["--source-type", self.source_var.get()]
            if self.series_var.get().strip():
                argv += ["--series", self.series_var.get().strip()]
            if models_str != "":
                argv += ["--models", models_str]
            else:
                argv += ["--models", ""]  # allow none (requiere que tu core lo tolere)

            if self.v_do_roman.get():
                argv += ["--do-roman-morph"]
            if self.v_html.get():
                argv += ["--html"]
            if skip_asr:
                argv += ["--skip-asr"]

            old_argv = sys.argv
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = _LogWriter(log)
            sys.stderr = _LogWriter(log)
            try:
                # Si has cambiado main(argv=None), esto lo usará.
                if "argv" in inspect.signature(core.main).parameters:
                    core.main(argv)
                else:
                    sys.argv = ["transcriber"] + argv
                    core.main()
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                sys.stdout, sys.stderr = old_out, old_err
                sys.argv = old_argv

        def done(ok, err):
            self.run_btn.configure(state="normal")
            self.cancel_btn.configure(state="disabled")

        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.runner.start("Transcriber", job, on_done=done)
