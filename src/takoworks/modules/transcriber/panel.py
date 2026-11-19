from __future__ import annotations

import inspect
import io
import os
import sys
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from ...config import save_config
from .source_type import normalize_source_type

SERIES_HISTORY_LIMIT = 25


def _clean_series_history(values):
    if not isinstance(values, list):
        return []

    history = []
    seen = set()
    for item in values:
        text = str(item).strip()
        if not text or text in seen:
            continue
        history.append(text)
        seen.add(text)
    return history


def _prepend_series_history(values, series):
    series = series.strip()
    if not series:
        return _clean_series_history(values)

    history = [item for item in _clean_series_history(values) if item != series]
    history.insert(0, series)
    return history[:SERIES_HISTORY_LIMIT]


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
        self.instructions_var = tk.StringVar(value=cfg["last"].get("transcriber_instructions_md", ""))
        self.pdf_var = tk.StringVar(value=cfg["last"].get("transcriber_reference_pdf", ""))
        self.purfview_var = tk.StringVar(value=cfg["last"].get("transcriber_purfview_exe", ""))
        self.asr_engine_var = tk.StringVar(value=cfg["last"].get("transcriber_asr_engine", "purfview"))
        self.v_verify_pdf = tk.BooleanVar(value=cfg["last"].get("transcriber_verify_pdf", False))

        self.lang_var = tk.StringVar(value="ja")
        self.series_history = _clean_series_history(cfg.get("series_history", []))
        initial_series = cfg["last"].get("series", "")
        self.series_var = tk.StringVar(value=initial_series)
        self.source_var = tk.StringVar(value="None")

        self.v_skip_asr = tk.BooleanVar(value=True)
        self.v_do_roman = tk.BooleanVar(value=True)
        self.v_context_note = tk.BooleanVar(value=True)
        self.v_html = tk.BooleanVar(value=True)
        self.v_final_pass = tk.BooleanVar(value=cfg["last"].get("transcriber_final_pass", False))

        self.v_gpt = tk.BooleanVar(value=True)
        self.v_claude = tk.BooleanVar(value=True)
        self.v_gemini = tk.BooleanVar(value=True)
        self.v_deepseek = tk.BooleanVar(value=True)

        self._build()
        self._toggle_video()

    def _build(self):
        canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)
        frm = ttk.Frame(canvas, padding=10)
        window = canvas.create_window((0, 0), window=frm, anchor="nw")
        frm.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))

        r0 = ttk.Frame(frm); r0.pack(fill="x", pady=3)
        ttk.Label(r0, text="Input ASS File").pack(side="left")
        ttk.Entry(r0, textvariable=self.ass_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r0, text="Browse", command=self._pick_ass).pack(side="left")

        r1 = ttk.Frame(frm); r1.pack(fill="x", pady=3)
        ttk.Label(r1, text="Video / audio").pack(side="left")
        self.video_entry = ttk.Entry(r1, textvariable=self.video_var)
        self.video_entry.pack(side="left", fill="x", expand=True, padx=6)
        self.video_btn = ttk.Button(r1, text="Browse", command=self._pick_video)
        self.video_btn.pack(side="left")

        r2 = ttk.Frame(frm); r2.pack(fill="x", pady=3)
        ttk.Label(r2, text="Output Folder").pack(side="left")
        ttk.Entry(r2, textvariable=self.out_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r2, text="Browse", command=self._pick_out).pack(side="left")

        r3 = ttk.Frame(frm); r3.pack(fill="x", pady=3)
        ttk.Label(r3, text="Series terminology / instructions (.md)").pack(side="left")
        ttk.Entry(r3, textvariable=self.instructions_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r3, text="Browse", command=self._pick_instructions).pack(side="left")

        initial = ttk.LabelFrame(frm, text="Initial transcription / PDF verification")
        initial.pack(fill="x", pady=4)
        engine_row = ttk.Frame(initial); engine_row.pack(fill="x", padx=8, pady=3)
        ttk.Label(engine_row, text="ASR engine").pack(side="left")
        self.engine_combo = ttk.Combobox(engine_row, textvariable=self.asr_engine_var,
                                        values=["purfview", "legacy"], width=12, state="readonly")
        self.engine_combo.pack(side="left", padx=6)
        self.engine_combo.bind("<<ComboboxSelected>>", lambda _event: self._toggle_video())
        ttk.Label(engine_row, text="Purfview: large-v3-turbo (ja / zh); Gen_ styles only").pack(side="left")
        exe_row = ttk.Frame(initial); exe_row.pack(fill="x", padx=8, pady=3)
        ttk.Label(exe_row, text="Faster Whisper XXL executable").pack(side="left")
        self.purfview_entry = ttk.Entry(exe_row, textvariable=self.purfview_var)
        self.purfview_entry.pack(side="left", fill="x", expand=True, padx=6)
        self.purfview_btn = ttk.Button(exe_row, text="Browse", command=self._pick_purfview)
        self.purfview_btn.pack(side="left")
        ttk.Checkbutton(initial, text="Verify episode PDF: OpenAI, then Gemini (translation models)",
                        variable=self.v_verify_pdf, command=self._toggle_video).pack(anchor="w", padx=8)
        pdf_row = ttk.Frame(initial); pdf_row.pack(fill="x", padx=8, pady=3)
        ttk.Label(pdf_row, text="Episode PDF").pack(side="left")
        self.pdf_entry = ttk.Entry(pdf_row, textvariable=self.pdf_var)
        self.pdf_entry.pack(side="left", fill="x", expand=True, padx=6)
        self.pdf_btn = ttk.Button(pdf_row, text="Browse", command=self._pick_pdf)
        self.pdf_btn.pack(side="left")

        meta = ttk.LabelFrame(frm, text="Metadata")
        meta.pack(fill="x", pady=8)

        rr = ttk.Frame(meta); rr.pack(fill="x", pady=3, padx=8)
        ttk.Label(rr, text="Language").pack(side="left")
        ttk.Combobox(rr, textvariable=self.lang_var, values=["ja", "zh"], width=6, state="readonly").pack(side="left", padx=6)
        ttk.Label(rr, text="Source").pack(side="left", padx=(12, 0))
        ttk.Combobox(rr, textvariable=self.source_var, values=["Manga", "Manhwa", "Light novel", "None"], width=12, state="readonly").pack(side="left", padx=6)
        ttk.Label(rr, text="Series").pack(side="left", padx=(12, 0))
        self.series_combo = ttk.Combobox(rr, textvariable=self.series_var, values=self.series_history, width=40)
        self.series_combo.pack(side="left", fill="x", expand=True, padx=6)

        opts = ttk.LabelFrame(frm, text="Options")
        opts.pack(fill="x", pady=8)

        ttk.Checkbutton(opts, text="Ignore transcription (skip ASR; no video required)", variable=self.v_skip_asr, command=self._toggle_video).pack(anchor="w", padx=8, pady=2)
        ttk.Checkbutton(opts, text="Add romaji/pinyin via DeepSeek", variable=self.v_do_roman).pack(anchor="w", padx=8, pady=2)
        ttk.Checkbutton(opts, text="Add contextual note", variable=self.v_context_note).pack(anchor="w", padx=8, pady=2)
        ttk.Checkbutton(opts, text="Generate summary HTML", variable=self.v_html).pack(anchor="w", padx=8, pady=2)
        ttk.Checkbutton(opts, text="Final Pass", variable=self.v_final_pass).pack(anchor="w", padx=8, pady=2)

        models = ttk.LabelFrame(frm, text="Translation Models")
        models.pack(fill="x", pady=8)

        rowm = ttk.Frame(models); rowm.pack(anchor="w", padx=8, pady=4)
        ttk.Checkbutton(rowm, text="GPT-6.1 Sol", variable=self.v_gpt).pack(side="left", padx=6)
        ttk.Checkbutton(rowm, text="Claude Opus 5.5", variable=self.v_claude).pack(side="left", padx=6)
        ttk.Checkbutton(rowm, text="Gemini 3.8 Flash", variable=self.v_gemini).pack(side="left", padx=6)
        ttk.Checkbutton(rowm, text="DeepSeek V4.1 Flash", variable=self.v_deepseek).pack(side="left", padx=6)

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
        self.engine_combo.configure(state="disabled" if skip else "readonly")
        purfview_state = "normal" if not skip and self.asr_engine_var.get() == "purfview" else "disabled"
        self.purfview_entry.configure(state=purfview_state)
        self.purfview_btn.configure(state=purfview_state)
        pdf_state = "normal" if self.v_verify_pdf.get() else "disabled"
        self.pdf_entry.configure(state=pdf_state)
        self.pdf_btn.configure(state=pdf_state)

    def _pick_purfview(self):
        p = filedialog.askopenfilename(filetypes=[("Faster Whisper XXL", "*.exe"), ("All files", "*.*")])
        if p:
            self.purfview_var.set(p)

    def _pick_pdf(self):
        p = filedialog.askopenfilename(filetypes=[("Episode PDF", "*.pdf")])
        if p:
            self.pdf_var.set(p)

    def _pick_ass(self):
        p = filedialog.askopenfilename(filetypes=[("ASS files", "*.ass")])
        if p:
            self.ass_var.set(p)
            self.out_var.set(os.path.dirname(p))

    def _pick_video(self):
        p = filedialog.askopenfilename(filetypes=[
            ("Video / audio", "*.mkv *.mp4 *.avi *.mov *.ts *.mka *.wav *.mp3 *.m4a *.flac *.ogg"),
            ("All files", "*.*"),
        ])
        if p:
            self.video_var.set(p)

    def _pick_out(self):
        p = filedialog.askdirectory()
        if p:
            self.out_var.set(p)

    def _pick_instructions(self):
        p = filedialog.askopenfilename(filetypes=[("Markdown", "*.md"), ("All files", "*.*")])
        if p:
            self.instructions_var.set(p)

    def _remember_series(self, series: str):
        history = _prepend_series_history(self.cfg.get("series_history", []), series)
        self.cfg["series_history"] = history
        self.series_history = history
        self.series_combo.configure(values=history)

    def _run(self):
        if self.runner.is_busy():
            return

        ass_in = self.ass_var.get().strip()
        if not ass_in or not os.path.isfile(ass_in):
            messagebox.showerror("Error", "Select a valid ASS file.")
            return

        skip_asr = bool(self.v_skip_asr.get())
        verify_pdf = bool(self.v_verify_pdf.get())
        reference_pdf = self.pdf_var.get().strip()
        asr_engine = self.asr_engine_var.get()
        purfview_exe = self.purfview_var.get().strip()
        if verify_pdf and (not os.path.isfile(reference_pdf) or not reference_pdf.lower().endswith(".pdf")):
            messagebox.showerror("Error", "Select a valid episode PDF for both verification passes.")
            return
        if not skip_asr and asr_engine == "purfview":
            from .initial_pass import resolve_purfview
            try:
                purfview_exe = resolve_purfview(purfview_exe)
            except ValueError as exc:
                messagebox.showerror("Error", str(exc))
                return
        video_in = self.video_var.get().strip()
        if not skip_asr and (not video_in or not os.path.isfile(video_in)):
            messagebox.showerror("Error", "Select a valid video or check Ignore transcription.")
            return

        out_dir = self.out_var.get().strip() or os.path.dirname(ass_in)
        instructions_md = self.instructions_var.get().strip()
        if instructions_md and not os.path.isfile(instructions_md):
            messagebox.showerror("Error", "Select a valid Markdown instructions file.")
            return

        models = []
        if self.v_gpt.get(): models.append("GPT-6.1 Sol")
        if self.v_claude.get(): models.append("Claude Opus 5.5")
        if self.v_gemini.get(): models.append("Gemini 3.8 Flash")
        if self.v_deepseek.get(): models.append("DeepSeek V4.1 Flash")
        models_str = ",".join(models)  # puede ser ""

        series_name = self.series_var.get().strip()
        self.cfg["last"]["series"] = series_name
        if series_name:
            self._remember_series(series_name)

        self.cfg["last"]["ass_in"] = ass_in
        self.cfg["last"]["video_in"] = video_in
        self.cfg["last"]["out_dir"] = out_dir
        self.cfg["last"]["transcriber_instructions_md"] = os.path.abspath(instructions_md) if instructions_md else ""
        final_pass = bool(self.v_final_pass.get())
        self.cfg["last"]["transcriber_final_pass"] = final_pass
        self.cfg["last"]["transcriber_reference_pdf"] = reference_pdf
        self.cfg["last"]["transcriber_verify_pdf"] = verify_pdf
        self.cfg["last"]["transcriber_asr_engine"] = asr_engine
        self.cfg["last"]["transcriber_purfview_exe"] = purfview_exe
        save_config(self.cfg)

        run_started_at = time.time()

        def job(cancel_event, log):
            old_run_started_at = os.environ.get("TAKOWORKS_TRANSCRIBER_RUN_STARTED_AT")
            os.environ["TAKOWORKS_TRANSCRIBER_RUN_STARTED_AT"] = str(run_started_at)
            argv = [ass_in]
            # Si tu transcriber core aún exige video posicional, ponemos algo “dummy”
            argv.append(video_in if video_in else ass_in)

            argv += ["--out-dir", out_dir]
            argv += ["--asr-engine", asr_engine]
            if purfview_exe:
                argv += ["--purfview-exe", purfview_exe]
            if verify_pdf:
                argv += ["--reference-pdf", reference_pdf]
            argv += ["--lang", self.lang_var.get()]
            argv += ["--source-type", normalize_source_type(self.source_var.get())]
            if series_name:
                argv += ["--series", series_name]
            if instructions_md:
                argv += ["--instructions-md", os.path.abspath(instructions_md)]
            if models_str != "":
                argv += ["--models", models_str]
            else:
                argv += ["--models", ""]  # allow none (requiere que tu core lo tolere)

            if self.v_do_roman.get():
                argv += ["--do-roman"]
            if self.v_context_note.get():
                argv += ["--do-context-note"]
            if self.v_html.get():
                argv += ["--html"]
            if skip_asr:
                argv += ["--skip-asr"]
            if final_pass:
                argv += ["--final-pass"]

            old_argv = sys.argv
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = _LogWriter(log)
            sys.stderr = _LogWriter(log)
            try:
                from . import core

                # Si has cambiado main(argv=None), esto lo usará.
                if "argv" in inspect.signature(core.main).parameters:
                    core.main(argv, cancel_event=cancel_event)
                else:
                    sys.argv = ["transcriber"] + argv
                    core.main()
            finally:
                if old_run_started_at is None:
                    os.environ.pop("TAKOWORKS_TRANSCRIBER_RUN_STARTED_AT", None)
                else:
                    os.environ["TAKOWORKS_TRANSCRIBER_RUN_STARTED_AT"] = old_run_started_at
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
