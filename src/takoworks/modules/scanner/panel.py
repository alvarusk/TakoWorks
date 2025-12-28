from __future__ import annotations

import os
from typing import Optional
import tempfile
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import fitz  # PyMuPDF

from ...config import save_config


class _CutSelection:
    def __init__(self):
        self.upper_ratio: Optional[float] = None  # 0..1
        self.lower_ratio: Optional[float] = None  # 0..1


class _PreviewWindow(tk.Toplevel):
    """
    Vista previa del PDF para elegir upper/lower cut con dos clicks.
    - Click 1: upper
    - Click 2: lower
    """
    def __init__(self, master, pdf_path: Path, selection: _CutSelection):
        super().__init__(master)
        self.title(f"Vista previa - {pdf_path.name}")
        self.pdf_path = pdf_path
        self.selection = selection

        self.doc = fitz.open(str(pdf_path))
        self.page_index = 0

        self.upper_px = None
        self.lower_px = None
        self.img_w = None
        self.img_h = None
        self.preview_png = None
        self.photo = None

        self.upper_line = None
        self.lower_line = None

        self._build_ui()
        self._render_page()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        top = ttk.Frame(self, padding=8)
        top.grid(row=0, column=0, sticky="we")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        ttk.Button(top, text="◀ Página", command=self._prev_page).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(top, text="Página ▶", command=self._next_page).grid(row=0, column=1, padx=(0, 10))

        self.lbl = ttk.Label(top, text="")
        self.lbl.grid(row=0, column=2, sticky="w")

        ttk.Button(top, text="Reset cortes", command=self._reset_cuts).grid(row=0, column=3, padx=(10, 0))
        ttk.Button(top, text="Usar estos cortes", command=self._accept).grid(row=0, column=4, padx=(10, 0))

        frame = ttk.Frame(self)
        frame.grid(row=1, column=0, sticky="nsew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(frame, bg="#1e1e1e")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        vs = ttk.Scrollbar(frame, orient="vertical", command=self.canvas.yview)
        hs = ttk.Scrollbar(frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        vs.grid(row=0, column=1, sticky="ns")
        hs.grid(row=1, column=0, sticky="we")

        self.canvas.bind("<Button-1>", self._on_click)

        ttk.Label(
            self,
            text="Click 1 = upper cut · Click 2 = lower cut (puedes navegar páginas)"
        ).grid(row=2, column=0, sticky="w", padx=8, pady=(6, 8))

    def _render_page(self):
        page = self.doc.load_page(self.page_index)
        rect = page.rect

        target_w = 900
        zoom = min(2.0, max(0.3, target_w / float(rect.width)))
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)

        tmp_dir = Path(tempfile.gettempdir()) / "takoworks_scanner_preview"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        self.preview_png = tmp_dir / f"preview_{os.getpid()}_{self.page_index+1:03d}.png"
        pix.save(str(self.preview_png))

        self.photo = tk.PhotoImage(file=str(self.preview_png))
        self.img_w = self.photo.width()
        self.img_h = self.photo.height()

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.config(scrollregion=(0, 0, self.img_w, self.img_h))

        self.lbl.config(text=f"Página {self.page_index+1}/{len(self.doc)}")
        self._redraw_from_selection()

    def _redraw_from_selection(self):
        self.upper_px = int(self.selection.upper_ratio * self.img_h) if self.selection.upper_ratio is not None else None
        self.lower_px = int(self.selection.lower_ratio * self.img_h) if self.selection.lower_ratio is not None else None
        self._draw_lines()

    def _draw_lines(self):
        if self.upper_line:
            self.canvas.delete(self.upper_line); self.upper_line = None
        if self.lower_line:
            self.canvas.delete(self.lower_line); self.lower_line = None
        if self.upper_px is not None:
            self.upper_line = self.canvas.create_line(0, self.upper_px, self.img_w, self.upper_px, fill="red", width=2)
        if self.lower_px is not None:
            self.lower_line = self.canvas.create_line(0, self.lower_px, self.img_w, self.lower_px, fill="cyan", width=2)

    def _on_click(self, evt):
        y = int(self.canvas.canvasy(evt.y))
        y = max(0, min(self.img_h - 1, y))

        if self.upper_px is None or (self.upper_px is not None and self.lower_px is not None):
            self.upper_px = y
            self.lower_px = None
        else:
            self.lower_px = y
            if self.lower_px < self.upper_px:
                self.upper_px, self.lower_px = self.lower_px, self.upper_px

        self._draw_lines()

    def _reset_cuts(self):
        self.selection.upper_ratio = None
        self.selection.lower_ratio = None
        self.upper_px = None
        self.lower_px = None
        self._draw_lines()

    def _accept(self):
        if self.upper_px is None or self.lower_px is None:
            messagebox.showwarning("Faltan cortes", "Haz dos clicks: upper y lower.")
            return
        if (self.lower_px - self.upper_px) < 10:
            messagebox.showwarning("Corte demasiado pequeño", "La distancia entre upper y lower es muy pequeña.")
            return
        self.selection.upper_ratio = self.upper_px / float(self.img_h)
        self.selection.lower_ratio = self.lower_px / float(self.img_h)
        self.destroy()

    def _prev_page(self):
        if self.page_index > 0:
            self.page_index -= 1
            self._render_page()

    def _next_page(self):
        if self.page_index < len(self.doc) - 1:
            self.page_index += 1
            self._render_page()

    def _on_close(self):
        try:
            self.doc.close()
        except Exception:
            pass
        self.destroy()


class ScannerPanel(ttk.Frame):
    def __init__(self, parent, runner, cfg: dict):
        super().__init__(parent)
        self.runner = runner
        self.cfg = cfg

        self.input_var = tk.StringVar(value=cfg["last"].get("pdf_in", ""))
        self.batch_var = tk.BooleanVar(value=bool(cfg["last"].get("scanner_batch", False)))
        self.out_var = tk.StringVar(value=cfg["last"].get("out_dir", ""))

        self.xlsx_var = tk.StringVar(value="")  # se llena tras correr

        self.yomitoku_var = tk.StringVar(value=cfg["last"].get("yomitoku_exe", ""))

        self.cut = _CutSelection()
        self.cut.upper_ratio = cfg["last"].get("cut_upper_ratio", None)
        self.cut.lower_ratio = cfg["last"].get("cut_lower_ratio", None)

        self.reuse_images_var = tk.BooleanVar(value=bool(cfg["last"].get("reuse_images", True)))
        self.reuse_md_var = tk.BooleanVar(value=bool(cfg["last"].get("reuse_md", True)))
        self.cleanup_var = tk.BooleanVar(value=bool(cfg["last"].get("scanner_cleanup", False)))

        self.out_excel_var = tk.BooleanVar(value=bool(cfg["last"].get("scanner_out_excel", True)))
        self.out_ass_var = tk.BooleanVar(value=bool(cfg["last"].get("scanner_out_ass", True)))
        self.out_txt_var = tk.BooleanVar(value=bool(cfg["last"].get("scanner_out_txt", False)))

        self._build()

    def _build(self):
        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, padx=10, pady=10)

        # Input (PDF or folder)
        r0 = ttk.Frame(frm); r0.pack(fill="x", pady=3)
        ttk.Label(r0, text="Input (PDF o carpeta)").pack(side="left")
        ttk.Entry(r0, textvariable=self.input_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r0, text="PDF…", command=self._pick_pdf).pack(side="left")
        ttk.Button(r0, text="Carpeta…", command=self._pick_folder).pack(side="left", padx=(6, 0))

        r0b = ttk.Frame(frm); r0b.pack(fill="x", pady=(0, 6))
        ttk.Checkbutton(r0b, text="Batch (procesar carpeta)", variable=self.batch_var).pack(side="left")

        # Output folder
        r1 = ttk.Frame(frm); r1.pack(fill="x", pady=3)
        ttk.Label(r1, text="Output Folder").pack(side="left")
        ttk.Entry(r1, textvariable=self.out_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r1, text="Browse", command=self._pick_out).pack(side="left")

        # YomiToku exe (optional)
        r2 = ttk.Frame(frm); r2.pack(fill="x", pady=3)
        ttk.Label(r2, text="YomiToku exe (opcional)").pack(side="left")
        ttk.Entry(r2, textvariable=self.yomitoku_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r2, text="Browse", command=self._pick_yomitoku).pack(side="left")

        # Cuts
        cuts = ttk.LabelFrame(frm, text="Cortes (upper/lower)")
        cuts.pack(fill="x", pady=10)

        rr = ttk.Frame(cuts); rr.pack(fill="x", padx=8, pady=6)
        ttk.Button(rr, text="Vista previa + elegir cortes…", command=self._open_preview).pack(side="left")
        self.cuts_label = ttk.Label(rr, text=self._cuts_label_text())
        self.cuts_label.pack(side="left", padx=(10, 0))

        rr2 = ttk.Frame(cuts); rr2.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Label(rr2, text="Upper ratio").pack(side="left")
        self.upper_entry = ttk.Entry(rr2, width=8)
        self.upper_entry.pack(side="left", padx=(6, 14))
        ttk.Label(rr2, text="Lower ratio").pack(side="left")
        self.lower_entry = ttk.Entry(rr2, width=8)
        self.lower_entry.pack(side="left", padx=6)

        # preload ratios in entries
        self.upper_entry.insert(0, "" if self.cut.upper_ratio is None else f"{float(self.cut.upper_ratio):.3f}")
        self.lower_entry.insert(0, "" if self.cut.lower_ratio is None else f"{float(self.cut.lower_ratio):.3f}")

        # Reuse / cleanup
        reuse = ttk.LabelFrame(frm, text="Reusar / limpieza")
        reuse.pack(fill="x", pady=6)
        rru = ttk.Frame(reuse); rru.pack(fill="x", padx=8, pady=6)
        ttk.Checkbutton(rru, text="Reusar imágenes ya cortadas (_images)", variable=self.reuse_images_var).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(rru, text="Reusar MD ya extraídos (_yomi_md)", variable=self.reuse_md_var).pack(side="left")
        rru2 = ttk.Frame(reuse); rru2.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Checkbutton(rru2, text="Eliminar cropped PDF al terminar (mantiene _images/_yomi_md como caché)", variable=self.cleanup_var).pack(side="left")

        # Outputs
        outs = ttk.LabelFrame(frm, text="Salidas")
        outs.pack(fill="x", pady=6)
        roo = ttk.Frame(outs); roo.pack(fill="x", padx=8, pady=6)
        ttk.Checkbutton(roo, text="Excel", variable=self.out_excel_var).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(roo, text="ASS (Gen_Main)", variable=self.out_ass_var).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(roo, text="TXT", variable=self.out_txt_var).pack(side="left", padx=(0, 10))

        # Buttons
        btns = ttk.Frame(frm); btns.pack(fill="x", pady=10)
        self.btn_run = ttk.Button(btns, text="Ejecutar Scanner", command=self._run_scanner)
        self.btn_run.pack(side="left")

        self.btn_excel_ass = ttk.Button(btns, text="Excel → ASS (Gen_Main)", command=self._run_excel_to_ass)
        self.btn_excel_ass.pack(side="left", padx=6)

        self.btn_cancel = ttk.Button(btns, text="Cancel", command=self.runner.cancel, state="disabled")
        self.btn_cancel.pack(side="left", padx=6)

        # Excel field (shows last xlsx)
        r3 = ttk.Frame(frm); r3.pack(fill="x", pady=3)
        ttk.Label(r3, text="Excel (último)").pack(side="left")
        ttk.Entry(r3, textvariable=self.xlsx_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r3, text="Browse", command=self._pick_xlsx).pack(side="left")

    def _cuts_label_text(self) -> str:
        if self.cut.upper_ratio is None or self.cut.lower_ratio is None:
            return "Cortes: (no definidos) · necesarios si NO reusas intermedios"
        return f"Cortes: upper={float(self.cut.upper_ratio):.3f} · lower={float(self.cut.lower_ratio):.3f}"

    def _sync_ratio_entries(self):
        # manual override if user typed
        u = self.upper_entry.get().strip()
        l = self.lower_entry.get().strip()
        self.cut.upper_ratio = float(u) if u else self.cut.upper_ratio
        self.cut.lower_ratio = float(l) if l else self.cut.lower_ratio

    def _open_preview(self):
        path = Path(self.input_var.get().strip())
        if not path.exists():
            messagebox.showwarning("Aviso", "Selecciona un PDF o una carpeta primero.")
            return

        if path.is_dir():
            pdfs = sorted([p for p in path.glob("*.pdf") if p.is_file()])
            if not pdfs:
                messagebox.showwarning("Aviso", "No encontré PDFs en esa carpeta.")
                return
            pdf_path = pdfs[0]
        else:
            pdf_path = path

        win = _PreviewWindow(self, pdf_path, self.cut)
        self.wait_window(win)

        # update entries and label
        self.upper_entry.delete(0, "end")
        self.lower_entry.delete(0, "end")
        self.upper_entry.insert(0, "" if self.cut.upper_ratio is None else f"{float(self.cut.upper_ratio):.3f}")
        self.lower_entry.insert(0, "" if self.cut.lower_ratio is None else f"{float(self.cut.lower_ratio):.3f}")
        self.cuts_label.config(text=self._cuts_label_text())

    def _pick_pdf(self):
        p = filedialog.askopenfilename(filetypes=[("PDF", "*.pdf")])
        if p:
            self.input_var.set(p)
            self.batch_var.set(False)
            if not self.out_var.get().strip():
                self.out_var.set(os.path.dirname(p))

    def _pick_folder(self):
        p = filedialog.askdirectory()
        if p:
            self.input_var.set(p)
            self.batch_var.set(True)
            if not self.out_var.get().strip():
                self.out_var.set(p)

    def _pick_xlsx(self):
        p = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx")])
        if p:
            self.xlsx_var.set(p)
            if not self.out_var.get().strip():
                self.out_var.set(os.path.dirname(p))

    def _pick_out(self):
        p = filedialog.askdirectory()
        if p:
            self.out_var.set(p)

    def _pick_yomitoku(self):
        p = filedialog.askopenfilename(filetypes=[("Executable", "*.exe"), ("All", "*.*")])
        if p:
            self.yomitoku_var.set(p)

    def _lock_ui(self, running: bool):
        state = "disabled" if running else "normal"
        self.btn_run.configure(state=state)
        self.btn_excel_ass.configure(state=state)
        self.btn_cancel.configure(state="normal" if running else "disabled")

    def _run_scanner(self):
        if self.runner.is_busy():
            return

        self._sync_ratio_entries()

        inp = self.input_var.get().strip()
        out_dir = self.out_var.get().strip() or (os.path.dirname(inp) if inp else "")

        if not inp:
            messagebox.showerror("Error", "Selecciona un PDF o una carpeta.")
            return
        if not os.path.exists(inp):
            messagebox.showerror("Error", "Ruta de entrada no válida.")
            return
        if not out_dir or not os.path.isdir(out_dir):
            messagebox.showerror("Error", "Selecciona un output folder válido.")
            return

        if not (self.out_excel_var.get() or self.out_ass_var.get() or self.out_txt_var.get()):
            messagebox.showwarning("Aviso", "Selecciona al menos una salida (Excel/ASS/TXT).")
            return

        # cortes obligatorios si no reusamos
        if not self.reuse_images_var.get() and not self.reuse_md_var.get():
            if self.cut.upper_ratio is None or self.cut.lower_ratio is None:
                messagebox.showwarning("Faltan cortes", "Abre la vista previa y selecciona upper/lower cut.")
                return

        # persist config
        self.cfg["last"]["pdf_in"] = inp
        self.cfg["last"]["out_dir"] = out_dir
        self.cfg["last"]["scanner_batch"] = bool(self.batch_var.get())
        self.cfg["last"]["yomitoku_exe"] = self.yomitoku_var.get().strip()

        self.cfg["last"]["cut_upper_ratio"] = self.cut.upper_ratio
        self.cfg["last"]["cut_lower_ratio"] = self.cut.lower_ratio

        self.cfg["last"]["reuse_images"] = bool(self.reuse_images_var.get())
        self.cfg["last"]["reuse_md"] = bool(self.reuse_md_var.get())
        self.cfg["last"]["scanner_cleanup"] = bool(self.cleanup_var.get())

        self.cfg["last"]["scanner_out_excel"] = bool(self.out_excel_var.get())
        self.cfg["last"]["scanner_out_ass"] = bool(self.out_ass_var.get())
        self.cfg["last"]["scanner_out_txt"] = bool(self.out_txt_var.get())
        save_config(self.cfg)

        # run
        result = {"last_xlsx": ""}

        def job(cancel_event, log):
            from . import core
            res = core.run_scanner(
                input_path=inp,
                out_dir=out_dir,
                batch=bool(self.batch_var.get()),
                upper_ratio=self.cut.upper_ratio,
                lower_ratio=self.cut.lower_ratio,
                reuse_images=bool(self.reuse_images_var.get()),
                reuse_md=bool(self.reuse_md_var.get()),
                make_excel=bool(self.out_excel_var.get()),
                make_ass=bool(self.out_ass_var.get()),
                make_txt=bool(self.out_txt_var.get()),
                cleanup=bool(self.cleanup_var.get()),
                yomitoku_exe=self.yomitoku_var.get().strip() or None,
                log=log,
                cancel_event=cancel_event,
            )

            # si es un solo PDF y se generó Excel, rellena el campo "último"
            if (not bool(self.batch_var.get())) and res:
                only = next(iter(res.values()))
                if only and getattr(only, "xlsx", None):
                    result["last_xlsx"] = only.xlsx

        def done(ok, err):
            self._lock_ui(False)
            if not ok:
                messagebox.showerror("Scanner", str(err) if err else "Error")
            if not ok:
                messagebox.showerror("Scanner", str(err) if err else "Error")
                return
            if result["last_xlsx"]:
                self.xlsx_var.set(result["last_xlsx"])

        self._lock_ui(True)
        self.runner.start("Scanner: OCR → Excel/ASS/TXT", job, on_done=done)

    def _run_excel_to_ass(self):
        if self.runner.is_busy():
            return

        xlsx = self.xlsx_var.get().strip()
        out_dir = self.out_var.get().strip() or (os.path.dirname(xlsx) if xlsx else "")

        if not xlsx or not os.path.isfile(xlsx):
            messagebox.showerror("Error", "Select a valid Excel file.")
            return
        if not out_dir or not os.path.isdir(out_dir):
            messagebox.showerror("Error", "Select a valid output folder.")
            return

        self.cfg["last"]["out_dir"] = out_dir
        save_config(self.cfg)

        def job(cancel_event, log):
            from . import core
            ass_path = core.excel_to_ass_with_styles(
                xlsx_path=xlsx,
                out_dir=out_dir,
                style_name="Gen_Main",
                log=log,
                cancel_event=cancel_event,
            )
            log(f"[OK] ASS: {ass_path}")

        def done(ok, err):
            self._lock_ui(False)
            if not ok:
                messagebox.showerror("Scanner", str(err) if err else "Error")

        self._lock_ui(True)
        self.runner.start("Scanner: Excel → ASS", job, on_done=done)
