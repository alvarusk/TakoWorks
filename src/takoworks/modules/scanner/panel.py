from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple
import json
import tempfile
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, TclError

import cv2
import fitz  # PyMuPDF
import numpy as np

from ...config import save_config


ICON_PREV = "\u2190"
ICON_NEXT = "\u2192"
ICON_ROT_LEFT = "\u27F2"
ICON_ROT_RIGHT = "\u27F3"
ICON_ACTOR = "\U0001F464"
ICON_DIALOG = "\U0001F4AC"
ICON_SUBTRACT = "\U0001F9F9"
ICON_UNDO = "X"
ICON_CUT_UP = "\u2702\u2191"
ICON_CUT_DOWN = "\u2702\u2193"
ICON_SAVE = "\U0001F4BE"
ICON_LOAD = "\u2912"


def _pixmap_to_bgr(pix: fitz.Pixmap) -> np.ndarray:
    """
    Conversión robusta de fitz.Pixmap -> BGR.
    Maneja CMYK/DeviceN, alfa y stride (relleno por fila).
    """
    base = pix
    if pix.colorspace is None or (pix.colorspace.n not in (1, 3)) or pix.n not in (1, 3, 4):
        base = fitz.Pixmap(fitz.csRGB, pix)
    elif pix.n in (3, 4) and pix.colorspace.n != 3:
        base = fitz.Pixmap(fitz.csRGB, pix)

    n = base.n
    h, w = base.h, base.w
    stride = base.stride

    buf = np.frombuffer(base.samples, dtype=np.uint8)
    arr = buf.reshape((h, stride))
    arr = arr[:, : w * n].copy()
    arr = arr.reshape((h, w, n))

    if n == 1:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif n >= 4 or base.alpha:
        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3] if arr.shape[2] > 3 else np.full((h, w), 255, dtype=np.uint8)
        bg = np.full_like(rgb, 255, dtype=np.uint8)
        rgb = ((rgb.astype(np.uint16) * alpha[..., None] + bg.astype(np.uint16) * (255 - alpha[..., None])) // 255).astype(np.uint8)
        arr = rgb

    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr


def _rotate_bgr(img: np.ndarray, deg: float) -> np.ndarray:
    if img is None or img.size == 0:
        return img
    if abs(deg) < 1e-3:
        return img
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(center, deg, 1.0)
    cos = abs(m[0, 0])
    sin = abs(m[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    m[0, 2] += (new_w / 2) - center[0]
    m[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(img, m, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))


class _SelectionState:
    def __init__(self):
        self.rects_by_page: Dict[int, List[Tuple[float, float, float, float]]] = {}
        self.rot_deg_by_page: Dict[int, float] = {}
        self.actor_band_by_page: Dict[int, Tuple[float, float]] = {}
        self.skip_rects_by_page: Dict[int, List[Tuple[float, float, float, float]]] = {}
        self.upper_ratio: Optional[float] = None
        self.lower_ratio: Optional[float] = None


class _PreviewWindow(tk.Toplevel):
    """
    Vista previa del PDF con selección rectangular libre, rotación por página y banda Actor (tecla X).
    """
    def __init__(self, master, pdf_path: Path, selection: _SelectionState):
        super().__init__(master)
        self.title(f"Vista previa - {pdf_path.name}")
        self.pdf_path = pdf_path
        self.selection = selection

        self.doc = fitz.open(str(pdf_path))
        self.page_index = 0

        self.img_w = None
        self.img_h = None
        self.photo = None
        self.preview_png = None
        self.bgr = None

        self.drag_start = None
        self.temp_rect = None
        self.drag_mode = "rect"  # rect | actor_band

        self.drag_start_skip = None
        self.temp_rect_skip = None

        self._build_ui()
        self._render_page()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Key>", self._on_key)
        self.bind("<Control-Alt-Right>", lambda _e: self._next_page())
        self.bind("<Control-Alt-Left>", lambda _e: self._prev_page())
        self.bind("<Control-Alt-e>", lambda _e: self._bump_rot(0.5))
        self.bind("<Control-Alt-E>", lambda _e: self._bump_rot(0.5))
        self.bind("<Control-Alt-q>", lambda _e: self._bump_rot(-0.5))
        self.bind("<Control-Alt-Q>", lambda _e: self._bump_rot(-0.5))
        self.bind("<Control-z>", lambda _e: self._undo_last())
        self.bind("<Control-Z>", lambda _e: self._undo_last())
        self.bind("<Alt-Shift-F1>", lambda _e: self._set_actor_mode())
        self.bind("<Alt-Shift-F2>", lambda _e: self._set_dialog_mode())
        self.bind("<Alt-Shift-F3>", lambda _e: self._set_skip_mode())

    def _build_ui(self):
        top = ttk.Frame(self, padding=8)
        top.grid(row=0, column=0, sticky="we")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        ttk.Button(top, text=ICON_PREV, command=self._prev_page).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(top, text=ICON_NEXT, command=self._next_page).grid(row=0, column=1, padx=(0, 10))

        self.lbl = ttk.Label(top, text="")
        self.lbl.grid(row=0, column=2, sticky="w")

        actions = ttk.Frame(top)
        actions.grid(row=0, column=3, padx=(10, 0))
        ttk.Button(actions, text=ICON_ROT_LEFT, command=lambda: self._bump_rot(-0.5)).pack(side="left", padx=(0, 4))
        ttk.Button(actions, text=ICON_ROT_RIGHT, command=lambda: self._bump_rot(0.5)).pack(side="left", padx=(0, 4))
        ttk.Button(actions, text=ICON_ACTOR, command=self._set_actor_mode).pack(side="left", padx=(0, 4))
        ttk.Button(actions, text=ICON_DIALOG, command=self._set_dialog_mode).pack(side="left", padx=(0, 4))
        ttk.Button(actions, text=ICON_SUBTRACT, command=self._set_skip_mode).pack(side="left", padx=(0, 4))
        ttk.Button(actions, text=ICON_UNDO, command=self._undo_last).pack(side="left", padx=(0, 4))
        ttk.Button(actions, text=ICON_CUT_UP, command=self._set_cut_upper_mode).pack(side="left", padx=(0, 4))
        ttk.Button(actions, text=ICON_CUT_DOWN, command=self._set_cut_lower_mode).pack(side="left", padx=(0, 4))
        ttk.Button(actions, text=ICON_SAVE, command=self._save_selection).pack(side="left", padx=(8, 4))
        ttk.Button(actions, text=ICON_LOAD, command=self._load_selection).pack(side="left", padx=(0, 4))
        ttk.Button(top, text="Restablecer", command=self._reset_page).grid(row=0, column=4, padx=(10, 0))
        ttk.Button(top, text="Cerrar", command=self._accept_and_close).grid(row=0, column=5, padx=(10, 0))

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

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<ButtonPress-3>", self._on_press_skip)
        self.canvas.bind("<B3-Motion>", self._on_drag_skip)
        self.canvas.bind("<ButtonRelease-3>", self._on_release_skip)
        self.canvas.focus_set()

        ttk.Label(
            self,
            text="Arrastra para dialogos (verde). Click derecho: restar zona (rojo). Banda Actor: boton o Alt+Shift+F1. Dialogo: Alt+Shift+F2. Restar zona: Alt+Shift+F3. Teclas: Ctrl+Alt+Flecha izq/der paginas, Ctrl+Alt+Q/E rotar, R restablecer rotacion, Ctrl+Z deshacer."
        ).grid(row=2, column=0, sticky="w", padx=8, pady=(6, 8))

    def _current_page(self) -> int:
        return self.page_index + 1

    def _render_page(self):
        page = self.doc.load_page(self.page_index)
        rect = page.rect

        target_w = 1000
        zoom = min(2.5, max(0.3, target_w / float(rect.width)))
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=True)

        bgr = _pixmap_to_bgr(pix)
        deg = float(self.selection.rot_deg_by_page.get(self._current_page(), 0.0) or 0.0)
        bgr = _rotate_bgr(bgr, deg)
        self.bgr = bgr

        tmp_dir = Path(tempfile.gettempdir()) / "takoworks_scanner_preview"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        self.preview_png = tmp_dir / f"preview_{os.getpid()}_{self.page_index+1:03d}.png"
        ok, buf = cv2.imencode(".png", bgr)
        if ok:
            self.preview_png.write_bytes(buf.tobytes())
        else:
            raise RuntimeError("No pude renderizar vista previa.")

        self.photo = tk.PhotoImage(file=str(self.preview_png))
        self.img_w = self.photo.width()
        self.img_h = self.photo.height()

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.config(scrollregion=(0, 0, self.img_w, self.img_h))

        self.lbl.config(text=f"Página {self.page_index+1}/{len(self.doc)}  · rot={deg:.1f}°  · rects={len(self.selection.rects_by_page.get(self._current_page(), []))}")
        self._draw_overlays()

    def _draw_overlays(self):
        self.canvas.delete("overlay")
        rects = self.selection.rects_by_page.get(self._current_page(), [])
        for (x0r, y0r, x1r, y1r) in rects:
            x0 = int(x0r * self.img_w)
            x1 = int(x1r * self.img_w)
            y0 = int(y0r * self.img_h)
            y1 = int(y1r * self.img_h)
            self.canvas.create_rectangle(x0, y0, x1, y1, outline="#3ad35c", width=2, fill="#3ad35c", stipple="gray50", tags="overlay")

        band = self.selection.actor_band_by_page.get(self._current_page())
        if band:
            y0 = int(band[0] * self.img_h)
            y1 = int(band[1] * self.img_h)
            self.canvas.create_rectangle(0, y0, self.img_w, y1, outline="#a455ff", width=2, fill="#a455ff", stipple="gray50", tags="overlay")

        skips = self.selection.skip_rects_by_page.get(self._current_page(), [])
        for (x0r, y0r, x1r, y1r) in skips:
            x0 = int(x0r * self.img_w)
            x1 = int(x1r * self.img_w)
            y0 = int(y0r * self.img_h)
            y1 = int(y1r * self.img_h)
            self.canvas.create_rectangle(x0, y0, x1, y1, outline="#ff4d4d", width=2, fill="#ff4d4d", stipple="gray50", tags="overlay")

        if self.selection.upper_ratio is not None:
            y = int(self.selection.upper_ratio * self.img_h)
            self.canvas.create_line(0, y, self.img_w, y, fill="#ffc107", width=2, tags="overlay")
        if self.selection.lower_ratio is not None:
            y = int(self.selection.lower_ratio * self.img_h)
            self.canvas.create_line(0, y, self.img_w, y, fill="#00bcd4", width=2, tags="overlay")

        if self.temp_rect:
            self.canvas.tag_raise(self.temp_rect)

    def _on_press(self, evt):
        if self.drag_mode == "skip_rect":
            self._on_press_skip(evt)
            return
        self.drag_start = (int(self.canvas.canvasx(evt.x)), int(self.canvas.canvasy(evt.y)))
        if self.temp_rect:
            self.canvas.delete(self.temp_rect)
            self.temp_rect = None

    def _on_drag(self, evt):
        if self.drag_mode == "skip_rect":
            self._on_drag_skip(evt)
            return
        if not self.drag_start:
            return
        x1 = int(self.canvas.canvasx(evt.x))
        y1 = int(self.canvas.canvasy(evt.y))
        x0, y0 = self.drag_start
        if self.temp_rect:
            self.canvas.coords(self.temp_rect, x0, y0, x1, y1)
        else:
            self.temp_rect = self.canvas.create_rectangle(x0, y0, x1, y1, outline="#33b5e5", dash=(4, 2), tags="overlay")

    def _on_release(self, evt):
        if self.drag_mode == "skip_rect":
            self._on_release_skip(evt)
            self.drag_mode = "rect"
            return
        if not self.drag_start:
            return
        x0, y0 = self.drag_start
        x1 = int(self.canvas.canvasx(evt.x))
        y1 = int(self.canvas.canvasy(evt.y))
        self.drag_start = None

        x0, x1 = sorted((max(0, min(self.img_w, x0)), max(0, min(self.img_w, x1))))
        y0, y1 = sorted((max(0, min(self.img_h, y0)), max(0, min(self.img_h, y1))))
        if self.drag_mode == "set_upper":
            self.selection.upper_ratio = y1 / float(self.img_h)
            self.drag_mode = "rect"
        elif self.drag_mode == "set_lower":
            self.selection.lower_ratio = y1 / float(self.img_h)
            self.drag_mode = "rect"
        else:
            if (x1 - x0) < 5 or (y1 - y0) < 5:
                if self.temp_rect:
                    self.canvas.delete(self.temp_rect)
                    self.temp_rect = None
                return

            if self.drag_mode == "actor_band":
                y0r = y0 / float(self.img_h)
                y1r = y1 / float(self.img_h)
                self.selection.actor_band_by_page[self._current_page()] = (y0r, y1r)
                self.drag_mode = "rect"
            else:
                x0r = x0 / float(self.img_w)
                x1r = x1 / float(self.img_w)
                y0r = y0 / float(self.img_h)
                y1r = y1 / float(self.img_h)
                rects = self.selection.rects_by_page.setdefault(self._current_page(), [])
                rects.append((x0r, y0r, x1r, y1r))

        if self.temp_rect:
            self.canvas.delete(self.temp_rect)
            self.temp_rect = None
        self._draw_overlays()
        self._render_page()

    def _on_press_skip(self, evt):
        self.drag_start_skip = (int(self.canvas.canvasx(evt.x)), int(self.canvas.canvasy(evt.y)))
        if self.temp_rect_skip:
            self.canvas.delete(self.temp_rect_skip)
            self.temp_rect_skip = None

    def _on_drag_skip(self, evt):
        if not self.drag_start_skip:
            return
        x1 = int(self.canvas.canvasx(evt.x))
        y1 = int(self.canvas.canvasy(evt.y))
        x0, y0 = self.drag_start_skip
        if self.temp_rect_skip:
            self.canvas.coords(self.temp_rect_skip, x0, y0, x1, y1)
        else:
            self.temp_rect_skip = self.canvas.create_rectangle(x0, y0, x1, y1, outline="#ff4d4d", dash=(4, 2), tags="overlay")

    def _on_release_skip(self, evt):
        if not self.drag_start_skip:
            return
        x0, y0 = self.drag_start_skip
        x1 = int(self.canvas.canvasx(evt.x))
        y1 = int(self.canvas.canvasy(evt.y))
        self.drag_start_skip = None

        x0, x1 = sorted((max(0, min(self.img_w, x0)), max(0, min(self.img_w, x1))))
        y0, y1 = sorted((max(0, min(self.img_h, y0)), max(0, min(self.img_h, y1))))
        if (x1 - x0) < 5 or (y1 - y0) < 5:
            if self.temp_rect_skip:
                self.canvas.delete(self.temp_rect_skip)
                self.temp_rect_skip = None
            return

        x0r = x0 / float(self.img_w)
        x1r = x1 / float(self.img_w)
        y0r = y0 / float(self.img_h)
        y1r = y1 / float(self.img_h)
        skips = self.selection.skip_rects_by_page.setdefault(self._current_page(), [])
        skips.append((x0r, y0r, x1r, y1r))

        if self.temp_rect_skip:
            self.canvas.delete(self.temp_rect_skip)
            self.temp_rect_skip = None
        self._render_page()

    def _on_key(self, evt):
        key = (evt.keysym or "").lower()
        if key == "a":
            self._prev_page()
        elif key == "d":
            self._next_page()
        elif key == "q":
            self._bump_rot(-0.5)
        elif key == "e":
            self._bump_rot(0.5)
        elif key == "r":
            self.selection.rot_deg_by_page.pop(self._current_page(), None)
            self._render_page()
        elif key == "x":
            self.drag_mode = "actor_band"
        elif key == "z":
            self._undo_last()

    def _bump_rot(self, delta: float):
        cur = float(self.selection.rot_deg_by_page.get(self._current_page(), 0.0) or 0.0)
        self.selection.rot_deg_by_page[self._current_page()] = cur + delta
        self._render_page()

    def _set_actor_mode(self):
        self.drag_mode = "actor_band"

    def _set_dialog_mode(self):
        self.drag_mode = "rect"

    def _set_skip_mode(self):
        self.drag_mode = "skip_rect"

    def _undo_last(self):
        rects = self.selection.rects_by_page.get(self._current_page(), [])
        if rects:
            rects.pop()
            if not rects:
                self.selection.rects_by_page.pop(self._current_page(), None)
            self._render_page()
            return
        skips = self.selection.skip_rects_by_page.get(self._current_page(), [])
        if skips:
            skips.pop()
            if not skips:
                self.selection.skip_rects_by_page.pop(self._current_page(), None)
            self._render_page()

    def _set_cut_upper_mode(self):
        self.drag_mode = "set_upper"

    def _set_cut_lower_mode(self):
        self.drag_mode = "set_lower"

    def _save_selection(self):
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not p:
            return
        data = {
            "rects_by_page": self.selection.rects_by_page,
            "actor_band_by_page": self.selection.actor_band_by_page,
            "rot_deg_by_page": self.selection.rot_deg_by_page,
            "skip_rects_by_page": self.selection.skip_rects_by_page,
            "upper_ratio": self.selection.upper_ratio,
            "lower_ratio": self.selection.lower_ratio,
        }
        Path(p).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_selection(self):
        p = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not p:
            return
        try:
            data = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            messagebox.showerror("Error", "No se pudo leer el archivo de selección.")
            return
        def _norm_dict(d):
            return {int(k): v for k, v in d.items()}
        self.selection.rects_by_page = _norm_dict(data.get("rects_by_page", {}))
        self.selection.actor_band_by_page = _norm_dict(data.get("actor_band_by_page", {}))
        self.selection.rot_deg_by_page = {int(k): float(v) for k, v in data.get("rot_deg_by_page", {}).items()}
        self.selection.skip_rects_by_page = _norm_dict(data.get("skip_rects_by_page", {}))
        self.selection.upper_ratio = data.get("upper_ratio", None)
        self.selection.lower_ratio = data.get("lower_ratio", None)
        self._render_page()

    def _reset_page(self):
        self.selection.rects_by_page.pop(self._current_page(), None)
        self.selection.actor_band_by_page.pop(self._current_page(), None)
        self.selection.skip_rects_by_page.pop(self._current_page(), None)
        self.selection.rot_deg_by_page.pop(self._current_page(), None)
        self._render_page()

    def _prev_page(self):
        if self.page_index > 0:
            self.page_index -= 1
            self._render_page()

    def _next_page(self):
        if self.page_index < len(self.doc) - 1:
            self.page_index += 1
            self._render_page()

    def _accept_and_close(self):
        self.destroy()

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

        self.selection = _SelectionState()
        self.selection.upper_ratio = cfg["last"].get("cut_upper_ratio", None)
        self.selection.lower_ratio = cfg["last"].get("cut_lower_ratio", None)

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

        # Manual selection
        sel_frame = ttk.LabelFrame(frm, text="Selección manual (rectángulos + actor band + rotación)")
        sel_frame.pack(fill="x", pady=8)
        rr = ttk.Frame(sel_frame); rr.pack(fill="x", padx=8, pady=6)
        ttk.Button(rr, text="Vista previa / marcar rects…", command=self._open_preview).pack(side="left")
        self.sel_label = ttk.Label(rr, text=self._selection_label_text())
        self.sel_label.pack(side="left", padx=(10, 0))

        # Legacy cuts
        cuts = ttk.LabelFrame(frm, text="Cortes legacy (upper/lower) — usado solo si no hay rectángulos")
        cuts.pack(fill="x", pady=6)
        rr2 = ttk.Frame(cuts); rr2.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Label(rr2, text="Upper ratio").pack(side="left")
        self.upper_entry = ttk.Entry(rr2, width=8)
        self.upper_entry.pack(side="left", padx=(6, 14))
        ttk.Label(rr2, text="Lower ratio").pack(side="left")
        self.lower_entry = ttk.Entry(rr2, width=8)
        self.lower_entry.pack(side="left", padx=6)
        self.upper_entry.insert(0, "" if self.selection.upper_ratio is None else f"{float(self.selection.upper_ratio):.3f}")
        self.lower_entry.insert(0, "" if self.selection.lower_ratio is None else f"{float(self.selection.lower_ratio):.3f}")

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

    def _selection_label_text(self) -> str:
        pages = len(self.selection.rects_by_page)
        band_pages = len(self.selection.actor_band_by_page)
        skip_pages = len(self.selection.skip_rects_by_page)
        return f"Rects: {pages} págs · Banda Actor: {band_pages} · Restas: {skip_pages}"

    def _sync_ratio_entries(self):
        u = self.upper_entry.get().strip()
        l = self.lower_entry.get().strip()
        self.selection.upper_ratio = float(u) if u else self.selection.upper_ratio
        self.selection.lower_ratio = float(l) if l else self.selection.lower_ratio

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

        win = _PreviewWindow(self, pdf_path, self.selection)
        self.wait_window(win)

        # update entries and labels
        try:
            if not self.winfo_exists():
                return
            if self.upper_entry.winfo_exists():
                self.upper_entry.delete(0, "end")
                self.upper_entry.insert(0, "" if self.selection.upper_ratio is None else f"{float(self.selection.upper_ratio):.3f}")
            if self.lower_entry.winfo_exists():
                self.lower_entry.delete(0, "end")
                self.lower_entry.insert(0, "" if self.selection.lower_ratio is None else f"{float(self.selection.lower_ratio):.3f}")
            if self.sel_label.winfo_exists():
                self.sel_label.config(text=self._selection_label_text())
        except TclError:
            # si la UI se cerró mientras estaba abierta la vista previa, ignoramos
            return

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

        if not self.selection.rects_by_page:
            messagebox.showwarning("Faltan rectangulos", "Abre la vista previa y marca rectangulos.")
            return

        # persist config
        self.cfg["last"]["pdf_in"] = inp
        self.cfg["last"]["out_dir"] = out_dir
        self.cfg["last"]["scanner_batch"] = bool(self.batch_var.get())
        self.cfg["last"]["yomitoku_exe"] = self.yomitoku_var.get().strip()

        self.cfg["last"]["cut_upper_ratio"] = self.selection.upper_ratio
        self.cfg["last"]["cut_lower_ratio"] = self.selection.lower_ratio

        manual_selection = bool(self.selection.rects_by_page or self.selection.skip_rects_by_page)
        use_reuse_images = bool(self.reuse_images_var.get()) if not manual_selection else False
        use_reuse_md = bool(self.reuse_md_var.get()) if not manual_selection else False
        self.cfg["last"]["reuse_images"] = bool(self.reuse_images_var.get())
        self.cfg["last"]["reuse_md"] = bool(self.reuse_md_var.get())
        self.cfg["last"]["scanner_cleanup"] = bool(self.cleanup_var.get())

        self.cfg["last"]["scanner_out_excel"] = bool(self.out_excel_var.get())
        self.cfg["last"]["scanner_out_ass"] = bool(self.out_ass_var.get())
        self.cfg["last"]["scanner_out_txt"] = bool(self.out_txt_var.get())
        save_config(self.cfg)

        result = {"last_xlsx": ""}

        def job(cancel_event, log):
            from . import core
            res = core.run_scanner(
                input_path=inp,
                out_dir=out_dir,
                batch=bool(self.batch_var.get()),
                upper_ratio=self.selection.upper_ratio,
                lower_ratio=self.selection.lower_ratio,
                reuse_images=use_reuse_images,
                reuse_md=use_reuse_md,
                make_excel=bool(self.out_excel_var.get()),
                make_ass=bool(self.out_ass_var.get()),
                make_txt=bool(self.out_txt_var.get()),
                cleanup=bool(self.cleanup_var.get()),
                yomitoku_exe=self.yomitoku_var.get().strip() or None,
                rects_by_page=self.selection.rects_by_page,
                rot_deg_by_page=self.selection.rot_deg_by_page,
                skip_rects_by_page=self.selection.skip_rects_by_page,
                actor_band_by_page=self.selection.actor_band_by_page,
                log=log,
                cancel_event=cancel_event,
            )

            if (not bool(self.batch_var.get())) and res:
                only = next(iter(res.values()))
                if only and getattr(only, "xlsx", None):
                    result["last_xlsx"] = only.xlsx

        def done(ok, err):
            self._lock_ui(False)
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
