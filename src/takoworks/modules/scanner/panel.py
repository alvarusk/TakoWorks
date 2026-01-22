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


class _SelectionState:
    def __init__(self):
        self.skip_polys_by_page: Dict[int, List[List[Tuple[float, float]]]] = {}
        self.skip_circles_by_page: Dict[int, List[Tuple[float, float, float]]] = {}
        self.suppress_pages: set[int] = set()
        self.crop_rect: Optional[Tuple[float, float, float, float]] = None
        self.crop_rect_by_page: Dict[int, Tuple[float, float, float, float]] = {}
        self.vertical_cuts: List[float] = []
        self.vertical_cuts_by_page: Dict[int, List[float]] = {}


class _PreviewWindow(tk.Toplevel):
    """
    Vista previa del PDF para borrar zonas y marcar recortes rectangulares.
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

        self.drag_mode = "erase_poly"  # erase_poly | crop_rect | vertical_cut
        self.crop_dragging = False
        self.crop_resize_edge = None
        self.crop_start_xy = None
        self.crop_start_rect = None
        self.crop_temp_id = None
        self.temp_poly = None
        self.brush_points: List[Tuple[int, int]] = []
        self.brush_dots: List[int] = []
        self.brush_radius = 20
        self.brush_var = tk.IntVar(value=self.brush_radius)

        self._build_ui()
        self._render_page()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Key>", self._on_key)
        self.bind("<Control-Alt-Right>", lambda _e: self._next_page())
        self.bind("<Control-Alt-Left>", lambda _e: self._prev_page())
        self.bind("<Control-z>", lambda _e: self._undo_last())
        self.bind("<Control-Z>", lambda _e: self._undo_last())

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
        ttk.Button(actions, text="Borrar zona", command=self._set_erase_mode).pack(side="left", padx=(0, 4))
        ttk.Label(actions, text="Pincel").pack(side="left", padx=(6, 2))
        ttk.Scale(actions, from_=5, to=80, orient="horizontal", variable=self.brush_var, command=self._on_brush_change).pack(side="left")
        ttk.Button(actions, text="Recorte", command=self._set_crop_mode).pack(side="left", padx=(0, 4))
        ttk.Button(actions, text="Corte vertical", command=self._set_vcut_mode).pack(side="left", padx=(0, 4))
        ttk.Button(actions, text="Deshacer", command=self._undo_last).pack(side="left", padx=(0, 4))
        ttk.Button(actions, text="Suprimir pagina", command=self._toggle_suppress_page).pack(side="left", padx=(6, 0))
        ttk.Button(actions, text="Guardar", command=self._save_selection).pack(side="left", padx=(6, 0))
        ttk.Button(actions, text="Cargar", command=self._load_selection).pack(side="left", padx=(6, 0))
        ttk.Button(actions, text="Exportar PDF", command=self._export_pdf).pack(side="left", padx=(6, 0))
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
        self.canvas.bind("<Motion>", self._on_move)
        self.canvas.focus_set()

        ttk.Label(
            self,
            text=(
                "Arrastra para borrar con pincel (rojo). "
                "Recorte: pulsa el boton y arrastra para crear un rectangulo. "
                "Corte vertical: haz clic para crear una linea. "
                "Pasa el raton por los lados para ajustar el recorte. "
                "Teclas: Ctrl+Alt+Flecha izq/der paginas, Ctrl+Z deshacer."
            )
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

        self._draw_overlays()
        self._update_label()

    def _draw_overlays(self):
        self.canvas.delete("overlay")
        polys = self.selection.skip_polys_by_page.get(self._current_page(), [])
        for poly in polys:
            if len(poly) < 3:
                continue
            coords: List[int] = []
            for xr, yr in poly:
                x = int(xr * self.img_w)
                y = int(yr * self.img_h)
                coords.extend([x, y])
            self.canvas.create_polygon(
                coords,
                outline="#ff4d4d",
                width=2,
                fill="#ff4d4d",
                stipple="gray50",
                tags="overlay",
            )
        circles = self.selection.skip_circles_by_page.get(self._current_page(), [])
        for (xr, yr, rr) in circles:
            x = int(xr * self.img_w)
            y = int(yr * self.img_h)
            r = int(rr * min(self.img_w, self.img_h))
            self.canvas.create_oval(
                x - r,
                y - r,
                x + r,
                y + r,
                outline="#ff4d4d",
                width=2,
                fill="#ff4d4d",
                stipple="gray50",
                tags="overlay",
            )

        if self.selection.crop_rect is not None:
            self._draw_crop_rect(self.selection.crop_rect, color="#00bcd4", width=2)
        page_crop = self.selection.crop_rect_by_page.get(self._current_page())
        if page_crop is not None:
            self._draw_crop_rect(page_crop, color="#ff9800", width=2)

        cuts = self._active_vertical_cuts()
        if cuts:
            for xr in cuts:
                x = int(xr * self.img_w)
                self.canvas.create_line(x, 0, x, self.img_h, fill="#ff9800", width=2, dash=(4, 2), tags="overlay")

        if self.temp_poly:
            self.canvas.tag_raise(self.temp_poly)
        self._update_label()

    def _draw_crop_rect(self, rect: Tuple[float, float, float, float], color: str, width: int = 2) -> None:
        left, right, top, bottom = rect
        x0 = int(left * self.img_w)
        x1 = int(right * self.img_w)
        y0 = int(top * self.img_h)
        y1 = int(bottom * self.img_h)
        self.canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=width, tags="overlay")

    def _update_label(self):
        poly_count = len(self.selection.skip_polys_by_page.get(self._current_page(), []))
        circle_count = len(self.selection.skip_circles_by_page.get(self._current_page(), []))
        suppressed = " | SUPRIMIDA" if self._current_page() in self.selection.suppress_pages else ""
        crop_note = " | crop=pagina" if self._current_page() in self.selection.crop_rect_by_page else ""
        cut_count = len(self._active_vertical_cuts())
        cut_note = f" | cortes={cut_count}" if cut_count else ""
        self.lbl.config(
            text=(
                f"Pagina {self.page_index+1}/{len(self.doc)} | "
                f"borrados={poly_count + circle_count}{suppressed}{crop_note}{cut_note}"
            )
        )

    def _on_press(self, evt):
        x = int(self.canvas.canvasx(evt.x))
        y = int(self.canvas.canvasy(evt.y))
        if self.drag_mode == "vertical_cut":
            if not self.img_w:
                return
            xr = max(0.0, min(1.0, x / float(self.img_w)))
            self._store_vertical_cut(xr)
            self._draw_overlays()
            return
        if self.drag_mode == "crop_rect":
            edge = self._hit_test_crop_edge(x, y)
            if edge:
                self.crop_resize_edge = edge
                self.crop_start_rect = self._active_crop_rect()
                self.crop_start_xy = (x, y)
                if self.crop_start_rect is not None:
                    x0, x1, y0, y1 = self._rect_to_px(self.crop_start_rect)
                    self._start_temp_rect(x0, y0, x1, y1)
                return
            self.crop_dragging = True
            self.crop_start_xy = (x, y)
            self._start_temp_rect(x, y, x, y)
            return

        self.brush_points = [(x, y)]
        if self.temp_poly:
            self.canvas.delete(self.temp_poly)
            self.temp_poly = None
        r = self.brush_radius
        dot = self.canvas.create_oval(x - r, y - r, x + r, y + r, outline="#ff4d4d", fill="#ff4d4d", stipple="gray50", tags="overlay")
        self.brush_dots = [dot]

    def _on_drag(self, evt):
        if self.drag_mode == "vertical_cut":
            return
        if self.drag_mode == "crop_rect":
            x = int(self.canvas.canvasx(evt.x))
            y = int(self.canvas.canvasy(evt.y))
            if self.crop_dragging and self.crop_start_xy:
                x0, y0 = self.crop_start_xy
                self._update_temp_rect(x0, y0, x, y)
                return
            if self.crop_resize_edge and self.crop_start_rect is not None:
                x0, x1, y0, y1 = self._rect_to_px(self.crop_start_rect)
                if self.crop_resize_edge == "left":
                    x0 = x
                elif self.crop_resize_edge == "right":
                    x1 = x
                elif self.crop_resize_edge == "top":
                    y0 = y
                elif self.crop_resize_edge == "bottom":
                    y1 = y
                self._update_temp_rect(x0, y0, x1, y1)
                return
            return

        if not self.brush_points:
            return
        x = int(self.canvas.canvasx(evt.x))
        y = int(self.canvas.canvasy(evt.y))
        self.brush_points.append((x, y))
        r = self.brush_radius
        dot = self.canvas.create_oval(x - r, y - r, x + r, y + r, outline="#ff4d4d", fill="#ff4d4d", stipple="gray50", tags="overlay")
        self.brush_dots.append(dot)

    def _on_release(self, evt):
        if self.drag_mode == "vertical_cut":
            return
        if self.drag_mode == "crop_rect":
            if not (self.crop_dragging or self.crop_resize_edge):
                return
            x = int(self.canvas.canvasx(evt.x))
            y = int(self.canvas.canvasy(evt.y))
            rect = None
            if self.crop_dragging and self.crop_start_xy:
                x0, y0 = self.crop_start_xy
                rect = self._px_to_rect(x0, x, y0, y)
            elif self.crop_resize_edge and self.crop_start_rect is not None:
                x0, x1, y0, y1 = self._rect_to_px(self.crop_start_rect)
                if self.crop_resize_edge == "left":
                    x0 = x
                elif self.crop_resize_edge == "right":
                    x1 = x
                elif self.crop_resize_edge == "top":
                    y0 = y
                elif self.crop_resize_edge == "bottom":
                    y1 = y
                rect = self._px_to_rect(x0, x1, y0, y1)

            if rect is not None:
                self._store_crop_rect(rect)
            self.crop_dragging = False
            self.crop_resize_edge = None
            self.crop_start_xy = None
            self.crop_start_rect = None
            self._clear_temp_rect()
            self._draw_overlays()
            return

        if not self.brush_points:
            return
        circles = self.selection.skip_circles_by_page.setdefault(self._current_page(), [])
        rr = self.brush_radius / float(min(self.img_w, self.img_h))
        for x, y in self.brush_points:
            xr = max(0.0, min(1.0, x / float(self.img_w)))
            yr = max(0.0, min(1.0, y / float(self.img_h)))
            circles.append((xr, yr, rr))

        self.brush_points = []
        for dot in self.brush_dots:
            self.canvas.delete(dot)
        self.brush_dots = []
        self._draw_overlays()

    def _on_key(self, evt):
        key = (evt.keysym or "").lower()
        if key == "a":
            self._prev_page()
        elif key == "d":
            self._next_page()
        elif key == "r":
            self._reset_page()
        elif key == "z":
            self._undo_last()

    def _on_move(self, evt):
        if self.drag_mode == "vertical_cut":
            self.canvas.config(cursor="sb_h_double_arrow")
            return
        if self.drag_mode != "crop_rect":
            return
        x = int(self.canvas.canvasx(evt.x))
        y = int(self.canvas.canvasy(evt.y))
        edge = self._hit_test_crop_edge(x, y)
        if edge in ("left", "right"):
            self.canvas.config(cursor="sb_h_double_arrow")
        elif edge in ("top", "bottom"):
            self.canvas.config(cursor="sb_v_double_arrow")
        else:
            self.canvas.config(cursor="crosshair")

    def _set_erase_mode(self):
        self.drag_mode = "erase_poly"
        if self.crop_temp_id is not None:
            self.canvas.delete(self.crop_temp_id)
            self.crop_temp_id = None
        self.canvas.config(cursor="")
        self._on_brush_change()

    def _set_crop_mode(self):
        self.drag_mode = "crop_rect"
        self.canvas.config(cursor="crosshair")

    def _set_vcut_mode(self):
        self.drag_mode = "vertical_cut"
        if self.crop_temp_id is not None:
            self.canvas.delete(self.crop_temp_id)
            self.crop_temp_id = None
        self.canvas.config(cursor="sb_h_double_arrow")

    def _active_crop_rect(self) -> Optional[Tuple[float, float, float, float]]:
        page = self._current_page()
        return self.selection.crop_rect_by_page.get(page) or self.selection.crop_rect

    def _active_vertical_cuts(self) -> List[float]:
        page = self._current_page()
        return self.selection.vertical_cuts_by_page.get(page, [])

    def _store_crop_rect(self, rect: Tuple[float, float, float, float]) -> None:
        page = self._current_page()
        if page in self.selection.crop_rect_by_page:
            self.selection.crop_rect_by_page[page] = rect
        elif self.selection.crop_rect is None or page == 1:
            self.selection.crop_rect = rect
        else:
            self.selection.crop_rect_by_page[page] = rect

    def _store_vertical_cut(self, xr: float) -> None:
        page = self._current_page()
        cuts = self.selection.vertical_cuts_by_page.setdefault(page, [])
        cuts.append(float(xr))
        cuts.sort()

    def _rect_to_px(self, rect: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
        left, right, top, bottom = rect
        x0 = int(left * self.img_w)
        x1 = int(right * self.img_w)
        y0 = int(top * self.img_h)
        y1 = int(bottom * self.img_h)
        return x0, x1, y0, y1

    def _px_to_rect(self, x0: int, x1: int, y0: int, y1: int) -> Optional[Tuple[float, float, float, float]]:
        x0 = max(0, min(self.img_w, x0))
        x1 = max(0, min(self.img_w, x1))
        y0 = max(0, min(self.img_h, y0))
        y1 = max(0, min(self.img_h, y1))
        if abs(x1 - x0) < 4 or abs(y1 - y0) < 4:
            return None
        left = min(x0, x1) / float(self.img_w)
        right = max(x0, x1) / float(self.img_w)
        top = min(y0, y1) / float(self.img_h)
        bottom = max(y0, y1) / float(self.img_h)
        return (left, right, top, bottom)

    def _hit_test_crop_edge(self, x: int, y: int) -> Optional[str]:
        rect = self._active_crop_rect()
        if rect is None:
            return None
        x0, x1, y0, y1 = self._rect_to_px(rect)
        pad = 6
        if (y0 - pad) <= y <= (y1 + pad):
            if abs(x - x0) <= pad:
                return "left"
            if abs(x - x1) <= pad:
                return "right"
        if (x0 - pad) <= x <= (x1 + pad):
            if abs(y - y0) <= pad:
                return "top"
            if abs(y - y1) <= pad:
                return "bottom"
        return None

    def _start_temp_rect(self, x0: int, y0: int, x1: int, y1: int) -> None:
        if self.crop_temp_id is not None:
            self.canvas.delete(self.crop_temp_id)
        self.crop_temp_id = self.canvas.create_rectangle(
            x0, y0, x1, y1,
            outline="#66ffcc",
            width=2,
            dash=(4, 2),
            tags="overlay",
        )

    def _update_temp_rect(self, x0: int, y0: int, x1: int, y1: int) -> None:
        if self.crop_temp_id is None:
            self._start_temp_rect(x0, y0, x1, y1)
            return
        self.canvas.coords(self.crop_temp_id, x0, y0, x1, y1)

    def _clear_temp_rect(self) -> None:
        if self.crop_temp_id is not None:
            self.canvas.delete(self.crop_temp_id)
            self.crop_temp_id = None

    def _on_brush_change(self, *_args):
        try:
            self.brush_radius = max(2, int(self.brush_var.get()))
        except Exception:
            self.brush_radius = 20

    def _undo_last(self):
        page = self._current_page()
        cuts = self.selection.vertical_cuts_by_page.get(page)
        if cuts:
            cuts.pop()
            if not cuts:
                self.selection.vertical_cuts_by_page.pop(page, None)
            self._draw_overlays()
            return
        circles = self.selection.skip_circles_by_page.get(self._current_page(), [])
        if circles:
            circles.pop()
            if not circles:
                self.selection.skip_circles_by_page.pop(self._current_page(), None)
            self._draw_overlays()
            return
        polys = self.selection.skip_polys_by_page.get(self._current_page(), [])
        if polys:
            polys.pop()
            if not polys:
                self.selection.skip_polys_by_page.pop(self._current_page(), None)
            self._draw_overlays()

    def _save_selection(self):
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not p:
            return
        data = {
            "skip_polys_by_page": self.selection.skip_polys_by_page,
            "skip_circles_by_page": self.selection.skip_circles_by_page,
            "suppress_pages": sorted(self.selection.suppress_pages),
            "crop_rect": self.selection.crop_rect,
            "crop_rect_by_page": self.selection.crop_rect_by_page,
            "vertical_cuts": [],
            "vertical_cuts_by_page": self.selection.vertical_cuts_by_page,
        }
        Path(p).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_selection(self):
        p = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not p:
            return
        try:
            data = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            messagebox.showerror("Error", "No se pudo leer el archivo de seleccion.")
            return

        def _norm_dict(d):
            return {int(k): v for k, v in d.items()}

        self.selection.skip_polys_by_page = _norm_dict(data.get("skip_polys_by_page", {}))
        self.selection.skip_circles_by_page = _norm_dict(data.get("skip_circles_by_page", {}))
        self.selection.suppress_pages = set(int(x) for x in data.get("suppress_pages", []))

        crop_rect = data.get("crop_rect")
        if crop_rect is None:
            upper = data.get("upper_ratio")
            lower = data.get("lower_ratio")
            if upper is not None and lower is not None:
                crop_rect = (0.0, 1.0, float(upper), float(lower))
        self.selection.crop_rect = tuple(crop_rect) if crop_rect is not None else None

        crop_by_page = data.get("crop_rect_by_page", {})
        self.selection.crop_rect_by_page = {int(k): tuple(v) for k, v in crop_by_page.items()}
        self.selection.vertical_cuts = []
        cuts_by_page = data.get("vertical_cuts_by_page", {})
        self.selection.vertical_cuts_by_page = {int(k): [float(x) for x in v] for k, v in cuts_by_page.items()}
        legacy_cuts = [float(x) for x in data.get("vertical_cuts", [])]
        if legacy_cuts and 1 not in self.selection.vertical_cuts_by_page:
            self.selection.vertical_cuts_by_page[1] = legacy_cuts
        self._render_page()

    def _export_pdf(self):
        default_name = f"{self.pdf_path.stem}_recortado.pdf"
        p = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
            initialdir=str(self.pdf_path.parent),
            initialfile=default_name,
        )
        if not p:
            return
        out_path = Path(p)
        if out_path.resolve() == self.pdf_path.resolve():
            messagebox.showerror("Exportar PDF", "Elige otro nombre para no sobrescribir el original.")
            return
        try:
            self._export_pdf_with_selection(out_path)
        except Exception as exc:
            messagebox.showerror("Exportar PDF", str(exc))
            return
        messagebox.showinfo("Exportar PDF", f"PDF exportado: {p}")

    def _export_pdf_with_selection(self, out_path: Path) -> None:
        from . import core as scan_core

        dpi = int(getattr(scan_core, "OCR_DPI", 250))
        zoom = dpi / 72.0

        out_doc = fitz.open()
        try:
            for page_index in range(len(self.doc)):
                page_num = page_index + 1
                if page_num in self.selection.suppress_pages:
                    continue

                page = self.doc.load_page(page_index)
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=True)
                bgr = _pixmap_to_bgr(pix)

                polys = self.selection.skip_polys_by_page.get(page_num, [])
                circles = self.selection.skip_circles_by_page.get(page_num, [])
                bgr = scan_core.apply_skip_marks(bgr, polys, circles)

                crop_rect = self.selection.crop_rect_by_page.get(page_num) or self.selection.crop_rect
                if crop_rect is not None:
                    x0, x1, y0, y1 = self._crop_rect_bounds(crop_rect, bgr.shape[1], bgr.shape[0])
                    bgr = bgr[y0:y1, x0:x1].copy()

                if bgr is None or bgr.size == 0:
                    continue

                ok, buf = cv2.imencode(".png", bgr)
                if not ok:
                    raise RuntimeError(f"No pude codificar pagina {page_num}.")

                h, w = bgr.shape[:2]
                page_w = w * 72.0 / dpi
                page_h = h * 72.0 / dpi
                new_page = out_doc.new_page(width=page_w, height=page_h)
                new_page.insert_image(fitz.Rect(0, 0, page_w, page_h), stream=buf.tobytes())

            if out_doc.page_count == 0:
                raise RuntimeError("No hay paginas para exportar.")

            out_doc.save(str(out_path), deflate=True, garbage=4)
        finally:
            out_doc.close()

    def _crop_rect_bounds(
        self,
        rect: Tuple[float, float, float, float],
        w: int,
        h: int,
    ) -> Tuple[int, int, int, int]:
        left, right, top, bottom = rect
        left = max(0.0, min(1.0, float(left)))
        right = max(0.0, min(1.0, float(right)))
        top = max(0.0, min(1.0, float(top)))
        bottom = max(0.0, min(1.0, float(bottom)))

        x0 = max(0, min(w, int(w * left)))
        x1 = max(0, min(w, int(w * right)))
        y0 = max(0, min(h, int(h * top)))
        y1 = max(0, min(h, int(h * bottom)))

        if x1 <= x0 + 1 or y1 <= y0 + 1:
            return 0, w, 0, h
        return x0, x1, y0, y1

    def _reset_page(self):
        self.selection.skip_polys_by_page.pop(self._current_page(), None)
        self.selection.skip_circles_by_page.pop(self._current_page(), None)
        self.selection.suppress_pages.discard(self._current_page())
        self.selection.crop_rect_by_page.pop(self._current_page(), None)
        self.selection.vertical_cuts_by_page.pop(self._current_page(), None)
        self._draw_overlays()
    
    def _toggle_suppress_page(self):
        page = self._current_page()
        if page in self.selection.suppress_pages:
            self.selection.suppress_pages.discard(page)
        else:
            self.selection.suppress_pages.add(page)
        self._draw_overlays()

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

        last_pdf = cfg["last"].get("pdf_in", "")
        self.input_var = tk.StringVar(value=last_pdf)
        out_default = os.path.dirname(last_pdf) if last_pdf else cfg["last"].get("out_dir", "")
        self.out_var = tk.StringVar(value=out_default)

        self.selection = _SelectionState()
        crop_left = cfg["last"].get("cut_left_ratio", None)
        crop_right = cfg["last"].get("cut_right_ratio", None)
        crop_top = cfg["last"].get("cut_top_ratio", None)
        crop_bottom = cfg["last"].get("cut_bottom_ratio", None)
        if crop_top is None and crop_bottom is None:
            legacy_top = cfg["last"].get("cut_upper_ratio", None)
            legacy_bottom = cfg["last"].get("cut_lower_ratio", None)
            if legacy_top is not None and legacy_bottom is not None:
                crop_top = legacy_top
                crop_bottom = legacy_bottom
        if crop_left is None and crop_right is None and crop_top is not None and crop_bottom is not None:
            crop_left, crop_right = 0.0, 1.0
        if None not in (crop_left, crop_right, crop_top, crop_bottom):
            self.selection.crop_rect = (float(crop_left), float(crop_right), float(crop_top), float(crop_bottom))

        self.out_excel_var = tk.BooleanVar(value=bool(cfg["last"].get("scanner_out_excel", True)))
        self.out_txt_var = tk.BooleanVar(value=bool(cfg["last"].get("scanner_out_txt", True)))

        self._build()

    def _build(self):
        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, padx=10, pady=10)

        # Input (PDF)
        r0 = ttk.Frame(frm); r0.pack(fill="x", pady=3)
        ttk.Label(r0, text="Input (PDF)").pack(side="left")
        ttk.Entry(r0, textvariable=self.input_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r0, text="PDF...", command=self._pick_pdf).pack(side="left")

        # Output folder (same as input)
        r1 = ttk.Frame(frm); r1.pack(fill="x", pady=3)
        ttk.Label(r1, text="Output (misma carpeta)").pack(side="left")
        ttk.Entry(r1, textvariable=self.out_var, state="readonly").pack(side="left", fill="x", expand=True, padx=6)

        # Cropping / viewer
        sel_frame = ttk.LabelFrame(frm, text="Cropping (visor PDF)")
        sel_frame.pack(fill="x", pady=8)
        rr = ttk.Frame(sel_frame); rr.pack(fill="x", padx=8, pady=6)
        ttk.Button(rr, text="Abrir visor", command=self._open_preview).pack(side="left")
        self.sel_label = ttk.Label(rr, text=self._selection_label_text())
        self.sel_label.pack(side="left", padx=(10, 0))

        # Outputs
        outs = ttk.LabelFrame(frm, text="Output")
        outs.pack(fill="x", pady=6)
        roo = ttk.Frame(outs); roo.pack(fill="x", padx=8, pady=6)
        ttk.Checkbutton(roo, text="Excel", variable=self.out_excel_var).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(roo, text="TXT", variable=self.out_txt_var).pack(side="left", padx=(0, 10))

        # Buttons
        btns = ttk.Frame(frm); btns.pack(fill="x", pady=10)
        self.btn_run = ttk.Button(btns, text="Ejecutar Scanner", command=self._run_scanner)
        self.btn_run.pack(side="left")

        self.btn_cancel = ttk.Button(btns, text="Cancel", command=self.runner.cancel, state="disabled")
        self.btn_cancel.pack(side="left", padx=6)

    def _selection_label_text(self) -> str:
        skip_pages = len(self.selection.skip_polys_by_page) + len(self.selection.skip_circles_by_page)
        sup_pages = len(self.selection.suppress_pages)
        crop_pages = len(self.selection.crop_rect_by_page)
        cut_pages = len(self.selection.vertical_cuts_by_page)
        return (
            f"Zonas borradas: {skip_pages} pags | Suprimidas: {sup_pages} | "
            f"Crop paginas: {crop_pages} | Cortes: {cut_pages}"
        )

    def _sync_ratio_entries(self):
        return

    def _open_preview(self):
        path = Path(self.input_var.get().strip())
        if not path.exists():
            messagebox.showwarning("Aviso", "Selecciona un PDF primero.")
            return

        if path.is_dir():
            messagebox.showwarning("Aviso", "Selecciona un PDF, no una carpeta.")
            return
        else:
            pdf_path = path

        win = _PreviewWindow(self, pdf_path, self.selection)
        self.wait_window(win)

        # update entries and labels
        try:
            if not self.winfo_exists():
                return
            if self.sel_label.winfo_exists():
                self.sel_label.config(text=self._selection_label_text())
        except TclError:


            # si la UI se cerró mientras estaba abierta la vista previa, ignoramos
            return

    def _pick_pdf(self):
        p = filedialog.askopenfilename(filetypes=[("PDF", "*.pdf")])
        if p:
            self.input_var.set(p)
            self.out_var.set(os.path.dirname(p))

    def _lock_ui(self, running: bool):
        state = "disabled" if running else "normal"
        self.btn_run.configure(state=state)
        self.btn_cancel.configure(state="normal" if running else "disabled")

    def _run_scanner(self):
        if self.runner.is_busy():
            return

        self._sync_ratio_entries()

        inp = self.input_var.get().strip()
        if not inp:
            messagebox.showerror("Error", "Selecciona un PDF.")
            return
        if not os.path.isfile(inp):
            messagebox.showerror("Error", "Ruta de entrada no valida.")
            return

        out_dir = os.path.dirname(inp)
        self.out_var.set(out_dir)

        if not (self.out_excel_var.get() or self.out_txt_var.get()):
            messagebox.showwarning("Aviso", "Selecciona al menos una salida (Excel/TXT).")
            return

        # persist config
        self.cfg["last"]["pdf_in"] = inp
        self.cfg["last"]["out_dir"] = out_dir

        if self.selection.crop_rect is None:
            l = r = t = b = None
        else:
            l, r, t, b = self.selection.crop_rect
        self.cfg["last"]["cut_left_ratio"] = l
        self.cfg["last"]["cut_right_ratio"] = r
        self.cfg["last"]["cut_top_ratio"] = t
        self.cfg["last"]["cut_bottom_ratio"] = b
        self.cfg["last"]["cut_upper_ratio"] = t
        self.cfg["last"]["cut_lower_ratio"] = b

        self.cfg["last"]["scanner_out_excel"] = bool(self.out_excel_var.get())
        self.cfg["last"]["scanner_out_txt"] = bool(self.out_txt_var.get())
        save_config(self.cfg)

        def job(cancel_event, log):
            from . import core
            core.run_gcloud_scanner(
                input_path=inp,
                out_dir=out_dir,
                make_excel=bool(self.out_excel_var.get()),
                make_txt=bool(self.out_txt_var.get()),
                crop_rect=self.selection.crop_rect,
                crop_rect_by_page=self.selection.crop_rect_by_page,
                skip_polys_by_page=self.selection.skip_polys_by_page,
                skip_circles_by_page=self.selection.skip_circles_by_page,
                suppress_pages=sorted(self.selection.suppress_pages),
                vertical_cuts=self.selection.vertical_cuts,
                vertical_cuts_by_page=self.selection.vertical_cuts_by_page,
                log=log,
                cancel_event=cancel_event,
            )

        def done(ok, err):
            self._lock_ui(False)
            if not ok:
                if err and "Cancelado" in str(err):
                    return
                messagebox.showerror("Scanner", str(err) if err else "Error")
                return

        self._lock_ui(True)
        self.runner.start("Scanner: GCV -> Excel/TXT", job, on_done=done)
