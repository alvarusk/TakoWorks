from __future__ import annotations

import os
import re
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import cv2
import numpy as np
import pandas as pd

# ============================================================
# Scanner core (headless) for TakoWorks
# - Pipeline: crop (upper/lower ratios) -> render to image -> slice columns
#           -> YomiToku OCR -> post-edit MD -> parse Actor/Text -> Excel/ASS/TXT
# - Supports reuse of intermediates: *_images and *_yomi_md
# - Unicode-safe image writing on Windows
# ============================================================

LogFn = Callable[[str], None]


# =========================
#  Ejecutable YomiToku (blindado contra PATH)
# =========================
def _default_yomitoku_exe() -> str:
    # Prefer the yomitoku.exe next to the running python (venv), fallback to PATH.
    p = Path(sys.executable).with_name("yomitoku.exe")
    return str(p) if p.exists() else "yomitoku"


# =========================
#  OCR / extracción
# =========================
OCR_DPI = 250

# Reduce solape entre columnas (sube si sigues viendo duplicados)
INNER_MARGIN_PX = 10

# Filtro de "columna vacía"
INK_NONWHITE_RATIO = 0.0045
INK_MIN_NONWHITE_PIXELS = 900

# Detección de separadores como líneas verticales largas (global)
SEP_MIN_HEIGHT_RATIO = 0.65   # sube a 0.70 si detecta caracteres como separadores
SEP_MAX_WIDTH_PX = 12
SEP_EDGE_MARGIN_PX = 20
SEP_MIN_GAP_MM = 6.0
SEP_KERNEL_HEIGHT_RATIO = 0.35


# =========================
#  Limpieza / parsing
# =========================
JP_ENDING_CHARS = "。！？!?…』」"
JP_PUNCT_END_EXTRA = "、，,．.｡。！？!?…」』）)"  # útil para merge de líneas MD

REPLACE_MAP = {
    ")S": ") ",
    ")E": ")",
}

# OJO: aquí mantenemos tokens de "ruido" comunes, pero el pos-edit "post_edit_yomitoku_md"
# se aplica antes y ya elimina algunos símbolos.
DELETE_TOKENS = [
    "(M)×", "(M)", "M)", "(M",
    "(OFF)", "(OF", "FF)",
    "()", "（）",
    "N)!", "N)",
    "(ON)60", "(ON)",
    "(O!", "(O",
    "(こぼし)", "（こぼし）",
    "(セリフこぼし)", "（セリフこぼし）",
    "(E)", "（E）",
    "←", "↓", "↑",
]

PUNCT_ONLY_RE = re.compile(
    r"^[!?！？]+$|^[。、。．\.…]+$|^[」』]+$|^[!?！？]+[」』]+$|^[」』]+[!?！？]+$"
)


def _check_cancel(cancel_event) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Cancelado")


def safe_name(name: str) -> str:
    """Nombre seguro para carpeta/archivo (evita #, espacios, etc.)."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("_")


# =========================
#  OpenCV + rutas Unicode (Windows)
# =========================
def cv_imwrite_u(path: Path, img: np.ndarray) -> None:
    """
    cv2.imwrite puede fallar con rutas no-ASCII en Windows (p.ej. japonés) sin lanzar excepción.
    Workaround: imencode + escribir bytes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = (path.suffix or ".png").lower()
    if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
        ext = ".png"
        path = path.with_suffix(ext)

    ok = False
    try:
        ok = bool(cv2.imwrite(str(path), img))
    except Exception:
        ok = False

    if ok:
        return

    success, buf = cv2.imencode(ext, img)
    if not success:
        raise RuntimeError(f"cv2.imencode falló para {path.name}")
    path.write_bytes(buf.tobytes())


def has_visible_ink_image(img_bgr: np.ndarray) -> bool:
    if img_bgr is None or img_bgr.size == 0:
        return False
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray < 250)
    return (float(np.mean(mask)) >= INK_NONWHITE_RATIO) and (int(np.sum(mask)) >= INK_MIN_NONWHITE_PIXELS)


def detect_vertical_separators_long_global(bgr: np.ndarray, dpi: int) -> List[int]:
    """Detecta separadores como líneas verticales largas SIN depender de una altura de barrido."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel_len = max(30, int(h * SEP_KERNEL_HEIGHT_RATIO))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    temp = cv2.morphologyEx(thr, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    temp = cv2.dilate(temp, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    contours, _ = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_h = int(h * SEP_MIN_HEIGHT_RATIO)
    xs: List[int] = []
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        if hh < min_h:
            continue
        if ww > SEP_MAX_WIDTH_PX:
            continue
        if x <= SEP_EDGE_MARGIN_PX or x >= (w - SEP_EDGE_MARGIN_PX):
            continue
        xs.append(x + ww // 2)

    xs = sorted(set(xs))
    px_per_mm = dpi / 25.4
    min_gap_px = int(SEP_MIN_GAP_MM * px_per_mm)
    kept: List[int] = []
    for x in xs:
        if not kept or (x - kept[-1]) >= min_gap_px:
            kept.append(x)
    return kept


def slice_columns_from_image(bgr: np.ndarray, lines_x: List[int]) -> List[Tuple[int, np.ndarray]]:
    """Corta en columnas usando separadores (orden derecha→izquierda)."""
    h, w = bgr.shape[:2]
    edges = [0] + sorted(lines_x) + [w - 1]
    pairs = list(zip(edges[:-1], edges[1:]))
    pairs = list(reversed(pairs))

    out: List[Tuple[int, np.ndarray]] = []
    sec = 1
    for (xl, xr) in pairs:
        x0 = xl + INNER_MARGIN_PX
        x1 = xr - INNER_MARGIN_PX
        if x1 <= x0:
            continue
        if (x1 - x0) < 35:
            continue
        col = bgr[:, x0:x1].copy()
        if not has_visible_ink_image(col):
            continue
        out.append((sec, col))
        sec += 1
    return out


# =========================
#  NUEVO: Post-edición del MD de YomiToku (orden exacto)
# =========================
def post_edit_yomitoku_md(md_text: str) -> str:
    """
    Aplica la limpieza EXACTA que pediste, justo después de que YomiToku genere el MD.
    Orden:
      1) Borra "|-"
      2) Borra "|"
      3) Borra "~"
      4) Borra "←"
      5) Borra "□"
      6) Reemplaza "\." por "."
      7) Reemplaza "\(" por "("
      8) Reemplaza "\)" por ")"
      9) Reemplaza "\!" por "!"
     10) Borra líneas vacías (o con solo espacios)
     11) Reemplaza "...。" por "..."
     12) Reemplaza "..。" por "..."
     13) Reemplaza ".。" por "。"
    """
    if md_text is None:
        return ""

    s = md_text

    # 1..5
    s = s.replace("|-", "")
    s = s.replace("|", "")
    s = s.replace("~", "")
    s = s.replace("←", "")
    s = s.replace("□", "")

    # 6..9 (des-escape)
    s = s.replace(r"\.", ".")
    s = s.replace(r"\(", "(")
    s = s.replace(r"\)", ")")
    s = s.replace(r"\!", "!")

    # 10 (remove empty lines)
    lines = [ln for ln in s.splitlines() if ln.strip() != ""]
    s = "\n".join(lines)

    # 11..13 (punct fixes)
    s = s.replace("...。", "...")
    s = s.replace("..。", "...")
    s = s.replace(".。", "。")

    return s


def merge_jp_lines(raw_text: str) -> List[str]:
    """
    Une líneas japonesas del MD:
    - ignora líneas vacías
    - quita encabezados Markdown (#, ##…)
    - si una línea NO termina en 」 ni puntuación/fin de frase japonés,
      se concatena con la siguiente.
    """
    lines = raw_text.splitlines()
    merged: List[str] = []
    buf = ""

    for raw in lines:
        s = raw.strip()
        if not s:
            continue

        if s.startswith("#"):
            s = re.sub(r"^#+\s*", "", s).strip()
            if not s:
                continue

        buf = (buf + s) if buf else s

        # final de frase
        if s[-1] in JP_ENDING_CHARS or s[-1] in JP_PUNCT_END_EXTRA:
            merged.append(buf)
            buf = ""

    if buf:
        merged.append(buf)
    return merged


# =========================
#  NUEVO: Encadenado por comillas japonesas incompletas 「」
# =========================
def merge_unbalanced_jp_quotes(lines: List[str]) -> List[str]:
    """
    Regla:
      - Si contiene 「 pero no 」 => encadena con la siguiente(s) hasta cerrar.
      - Si contiene 」 pero no 「 => encadena a la anterior.
    Implementación por balance de comillas.
    """
    out: List[str] = []
    buf = ""
    balance = 0  # nº de 「 - nº de 」

    for raw in lines:
        s = (raw or "").strip()
        if not s:
            continue

        opens = s.count("「")
        closes = s.count("」")

        # Caso: cierra sin abrir -> al anterior
        if opens == 0 and closes > 0 and balance == 0 and out:
            out[-1] = (out[-1] + s).strip()
            continue

        if balance > 0:
            buf = (buf + s).strip()
        else:
            buf = s

        balance += opens - closes

        if balance <= 0:
            out.append(buf.strip())
            buf = ""
            balance = 0

    if buf.strip():
        out.append(buf.strip())

    return out


def clean_text(s: str) -> str:
    """Limpieza para bloques de texto (MD o PyMuPDF text). Conserva contenido entre paréntesis."""
    if not s:
        return ""

    for a, b in REPLACE_MAP.items():
        s = s.replace(a, b)

    # Des-escapar paréntesis del MD para conservar contenido
    s = s.replace(r"\(", "(").replace(r"\)", ")")
    s = s.replace(r"\（", "（").replace(r"\）", "）")

    # Des-escapar signos típicos
    s = s.replace(r"\!", "!").replace(r"\！", "！")
    s = s.replace(r"\~", "~").replace(r"\～", "～")

    # Borrado tokens concretos (ruido típico)
    for t in DELETE_TOKENS:
        s = s.replace(t, "")

    # Eliminar paréntesis vacíos
    s = re.sub(r"\(\s*\)", "", s)
    s = re.sub(r"（\s*）", "", s)

    # Eliminar "SE" en mayúsculas en cualquier posición (incluye fullwidth)
    s = s.replace("SE", "").replace("ＳＥ", "").replace("SＥ", "").replace("ＳE", "")

    s = s.replace("　", " ")
    s = re.sub(r"[ \t]+", " ", s).strip()

    # Si 」 va seguido de más texto, insertar salto
    s = re.sub(r"」(?!\s*$|\n)", "」\n", s)
    return s


def looks_like_actor(line: str) -> bool:
    """
    Actor típico en MD ya poseditado:
      - 1..6 caracteres (tu regla)
      - sin comillas japonesas
      - sin puntuación ni símbolos típicos
    """
    if not line:
        return False
    if "「" in line or "」" in line:
        return False
    if re.search(r"[。、「」()（）…！？!?.,，:：;；\-—→←↓↑『』]", line):
        return False
    return 1 <= len(line) <= 6


def parse_dialogues_from_text(block: str) -> List[dict]:
    """
    Convierte un bloque (ya con saltos) en filas actor/texto.
    Soporta:
      - Actor en línea separada + línea(s) de texto (a menudo con 「」)
      - Actor「Texto」 en una sola línea
    """
    block = clean_text(block)
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]

    # Encadenar comillas japonesas rotas a nivel de línea (por si el merge anterior no las cerró)
    lines = merge_unbalanced_jp_quotes(lines)

    out: List[dict] = []
    current_actor = ""

    for ln in lines:
        if "(SE)" in ln or "（SE）" in ln:
            continue

        if looks_like_actor(ln):
            current_actor = ln
            continue

        # Línea con comillas completas
        if "「" in ln and "」" in ln and ln.find("「") < ln.rfind("」"):
            actor = ln.split("「", 1)[0].strip() or current_actor
            inner = ln.split("「", 1)[1]
            text = inner.rsplit("」", 1)[0].strip()
            out.append({"actor": actor, "texto": text})
        else:
            # Narración / texto sin comillas
            out.append({"actor": current_actor, "texto": ln})

    # Quitar comillas SOLO aquí
    cleaned: List[dict] = []
    for d in out:
        a = (d.get("actor") or "").replace("「", "").replace("」", "").strip()
        t = (d.get("texto") or "").replace("「", "").replace("」", "").strip()
        if a or t:
            cleaned.append({"actor": a, "texto": t})
    return cleaned


# =========================
#  Postprocesado DF
# =========================
def infer_actor_prefix(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    counts = df["actor"].fillna("").astype(str).str.strip()
    counts = counts[counts != ""].value_counts()
    actors = [a for a, c in counts.items() if c >= 2]
    actors = sorted(set(actors), key=len, reverse=True)
    if not actors:
        return df

    def split_row(row):
        a = (row.get("actor") or "").strip()
        t = (row.get("texto") or "").strip()
        if a or not t:
            return row
        for cand in actors:
            if t.startswith(cand) and len(t) > len(cand):
                rest = t[len(cand):].lstrip()
                if len(rest) >= 2:
                    row["actor"] = cand
                    row["texto"] = rest
                    return row
        return row

    return df.apply(split_row, axis=1)


def fill_actor_with_carry(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["actor"] = df["actor"].fillna("").astype(str)
    df = df.sort_values(["pagina", "seccion"], kind="stable").reset_index(drop=True)

    last_actor_prev_page = ""
    for p in sorted(df["pagina"].unique()):
        mask = df["pagina"] == p
        idxs = df.index[mask].tolist()
        if not idxs:
            continue

        first_i = idxs[0]
        if df.at[first_i, "actor"].strip() == "" and last_actor_prev_page.strip():
            df.at[first_i, "actor"] = last_actor_prev_page

        df.loc[mask, "actor"] = df.loc[mask, "actor"].replace("", np.nan).ffill().fillna("")

        non_empty = df.loc[mask, "actor"].replace("", np.nan).dropna()
        if len(non_empty) > 0:
            last_actor_prev_page = str(non_empty.iloc[-1])

    return df


def merge_punctuation_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df = df.sort_values(["pagina", "seccion"], kind="stable").reset_index(drop=True)

    drop_idx = []
    for i in range(1, len(df)):
        a = str(df.at[i, "actor"] or "").strip()
        t = str(df.at[i, "texto"] or "").strip()
        if not t or not PUNCT_ONLY_RE.match(t):
            continue

        a_prev = str(df.at[i - 1, "actor"] or "").strip()
        if a != a_prev:
            continue
        if int(df.at[i, "pagina"]) != int(df.at[i - 1, "pagina"]):
            continue
        if int(df.at[i, "seccion"]) != int(df.at[i - 1, "seccion"]):
            continue

        df.at[i - 1, "texto"] = str(df.at[i - 1, "texto"] or "").rstrip() + t
        drop_idx.append(i)

    if drop_idx:
        df = df.drop(index=drop_idx).reset_index(drop=True)
    return df


def drop_overlap_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.drop_duplicates(subset=["pagina", "actor", "texto"]).copy()
    df = df.sort_values(["pagina", "seccion"], kind="stable").reset_index(drop=True)

    rows = df.to_dict("records")
    keep = [True] * len(rows)

    for i in range(len(rows) - 1):
        a1 = (rows[i].get("actor") or "").strip()
        t1 = (rows[i].get("texto") or "").strip()
        a2 = (rows[i + 1].get("actor") or "").strip()
        t2 = (rows[i + 1].get("texto") or "").strip()
        p1 = rows[i].get("pagina")
        p2 = rows[i + 1].get("pagina")

        if p1 != p2 or not t1 or not t2:
            continue
        if a1 != a2:
            continue
        if t1 in t2 and len(t2) >= len(t1) + 6:
            keep[i] = False
        elif t2 in t1 and len(t1) >= len(t2) + 6:
            keep[i + 1] = False

    out = [r for i, r in enumerate(rows) if keep[i]]
    return pd.DataFrame(out)


def finalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["actor"] = df["actor"].fillna("").astype(str)
    df["texto"] = df["texto"].fillna("").astype(str)

    df = infer_actor_prefix(df)
    df = fill_actor_with_carry(df)
    df = merge_punctuation_rows(df)
    df = drop_overlap_duplicates(df)
    return df


# =========================
#  OCR: YomiToku → filas
# =========================
def _index_md_by_page_section(md_dir: Path) -> Dict[Tuple[int, int], Path]:
    idx: Dict[Tuple[int, int], Path] = {}
    for p in md_dir.rglob("*.md"):
        m = re.search(r"_p(\d{3})s(\d{3})", p.name)
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2)))
        if key not in idx or p.stat().st_mtime > idx[key].stat().st_mtime:
            idx[key] = p
    return idx


def _post_edit_and_persist_md(md_path: Path, log: LogFn) -> str:
    """
    Lee MD, aplica post_edit_yomitoku_md y lo escribe de vuelta si cambia.
    Devuelve el texto (ya poseditado).
    """
    raw = md_path.read_text(encoding="utf-8", errors="ignore")
    edited = post_edit_yomitoku_md(raw)
    if edited != raw:
        try:
            md_path.write_text(edited, encoding="utf-8", errors="replace")
        except Exception as e:
            log(f"[WARN] No pude reescribir MD poseditado ({md_path.name}): {e}")
    return edited


def _df_from_md_files(md_files: List[Path], log: LogFn = lambda s: None) -> pd.DataFrame:
    rows: List[dict] = []
    for md_path in sorted(md_files):
        m = re.search(r"_p(\d{3})s(\d{3})", md_path.name)
        if not m:
            continue
        page_num = int(m.group(1))
        sec = int(m.group(2))

        md = _post_edit_and_persist_md(md_path, log)
        merged = merge_jp_lines(md)
        merged = merge_unbalanced_jp_quotes(merged)

        block = "\n".join(merged)
        dialogues = parse_dialogues_from_text(block)
        for d in dialogues:
            rows.append({
                "pagina": page_num,
                "seccion": sec,
                "actor": (d.get("actor") or "").strip(),
                "texto": (d.get("texto") or "").strip(),
            })
    return finalize_df(pd.DataFrame(rows)) if rows else pd.DataFrame(columns=["pagina", "seccion", "actor", "texto"])


def _ocr_one_column_to_rows(
    yomitoku_exe: str,
    col_img_path: Path,
    page_num: int,
    sec: int,
    md_dir: Path,
    log: LogFn,
) -> List[dict]:
    if not col_img_path.exists():
        log(f"[WARN] Imagen no encontrada: {col_img_path}")
        return []

    outdir_col = md_dir / safe_name(col_img_path.stem)
    outdir_col.mkdir(parents=True, exist_ok=True)

    cmd = [
        yomitoku_exe,
        str(col_img_path),
        "-f", "md",
        "-o", str(outdir_col),
        "-d", "cpu",
        "--ignore_line_break",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        if err:
            log(err)
        log(f"[WARN] YomiToku error ({proc.returncode}) en {col_img_path.name}")
        return []

    candidates = list(outdir_col.rglob("*.md"))
    if not candidates:
        # fallback: buscar por p###s### en todo md_dir
        candidates = [p for p in md_dir.rglob("*.md") if re.search(rf"_p{page_num:03d}s{sec:03d}", p.name)]

    if not candidates:
        log(f"[WARN] No MD para {col_img_path.stem}")
        return []

    md_path = max(candidates, key=lambda p: p.stat().st_mtime)
    md = _post_edit_and_persist_md(md_path, log)

    merged = merge_jp_lines(md)
    merged = merge_unbalanced_jp_quotes(merged)

    block = "\n".join(merged)
    dialogues = parse_dialogues_from_text(block)

    out_rows: List[dict] = []
    for d in dialogues:
        out_rows.append({
            "pagina": page_num,
            "seccion": sec,
            "actor": (d.get("actor") or "").strip(),
            "texto": (d.get("texto") or "").strip(),
        })
    return out_rows


# =========================
#  Outputs
# =========================
def write_txt(df: pd.DataFrame, out_txt: Path) -> None:
    lines: List[str] = []
    for r in df.itertuples(index=False):
        a = str(getattr(r, "actor", "") or "").strip()
        t = str(getattr(r, "texto", "") or "").strip()
        lines.append(f"{a}|{t}" if a else t)
    out_txt.write_text("\n".join(lines), encoding="utf-8")


def df_to_ass_with_styles(
    df: pd.DataFrame,
    out_ass: Path,
    style_name: str,
    log: LogFn,
    cancel_event=None,
) -> None:
    """
    Crea ASS directamente desde el DF, pero inyecta estilos usando Stylizer
    para mantener consistencia con TakoWorks.

    Nota: El Actor se escribe en el campo "Name" del evento ASS (Aegisub: Actor).
    """
    _check_cancel(cancel_event)

    lines: List[str] = []
    lines += ["[Script Info]", "Title: TakoWorks Scanner", "ScriptType: v4.00+", "WrapStyle: 0", "ScaledBorderAndShadow: yes", ""]
    # placeholder styles (Stylizer los reemplaza)
    lines += [
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,0,2,20,20,20,1",
        "",
    ]
    lines += ["[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]

    for r in df.itertuples(index=False):
        _check_cancel(cancel_event)
        actor = str(getattr(r, "actor", "") or "").strip().replace(",", " ")
        text = str(getattr(r, "texto", "") or "").strip()
        if not text and not actor:
            continue
        text = text.replace("\n", r"\N")
        lines.append(f"Dialogue: 0,0:00:00.00,0:00:00.00,{style_name},{actor},0,0,0,,{text}")

    from ..stylizer import core as styl_core  # local import
    adapter = _dummy_console(log)
    lines = styl_core.replace_styles_section(lines, console=adapter)

    out_ass.write_text("\n".join(lines), encoding="utf-8-sig", errors="replace")


def _dummy_console(log: LogFn):
    class C:
        def insert(self, *_args):
            txt = str(_args[-1])
            for line in txt.splitlines():
                if line.strip():
                    log(line)
        def see(self, *_args): pass
        def update_idletasks(self): pass
        def config(self, **_kw): pass
        def configure(self, **_kw): pass
    return C()


def excel_to_ass_with_styles(
    xlsx_path: str,
    out_dir: str,
    style_name: str = "Gen_Main",
    log: LogFn = lambda s: None,
    cancel_event=None,
) -> str:
    """
    Mantener compatibilidad con el botón Excel→ASS del panel.
    """
    xlsx_p = Path(xlsx_path)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(xlsx_p)
    actor_col = _pick_col(df, "actor", "Actor", "NAME", "Name")
    text_col = _pick_col(df, "texto", "Texto", "text", "Text")
    if not text_col:
        raise RuntimeError("No encuentro columna Texto/text en el Excel.")

    out_ass = out_dir_p / (xlsx_p.stem + "_GenMain.ass")

    lines: List[str] = []
    lines += ["[Script Info]", "Title: TakoWorks Scanner", "ScriptType: v4.00+", "WrapStyle: 0", "ScaledBorderAndShadow: yes", ""]
    lines += [
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,0,2,20,20,20,1",
        "",
    ]
    lines += ["[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]

    for _, r in df.iterrows():
        _check_cancel(cancel_event)
        actor = (str(r[actor_col]).strip() if actor_col else "").replace(",", " ")
        text = str(r[text_col]).strip()
        if not text and not actor:
            continue
        text = text.replace("\n", r"\N")
        lines.append(f"Dialogue: 0,0:00:00.00,0:00:00.00,{style_name},{actor},0,0,0,,{text}")

    from ..stylizer import core as styl_core
    adapter = _dummy_console(log)
    lines = styl_core.replace_styles_section(lines, console=adapter)

    out_ass.write_text("\n".join(lines), encoding="utf-8-sig", errors="replace")
    return str(out_ass)


def _pick_col(df: pd.DataFrame, *names: str) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in cols:
            return cols[n.lower()]
    return None


# =========================
#  Main scanner entry points
# =========================
@dataclass
class ScanOutputs:
    xlsx: Optional[str] = None
    ass: Optional[str] = None
    txt: Optional[str] = None


def run_scanner(
    input_path: str,
    out_dir: str,
    batch: bool,
    upper_ratio: Optional[float],
    lower_ratio: Optional[float],
    reuse_images: bool = True,
    reuse_md: bool = True,
    make_excel: bool = True,
    make_ass: bool = True,
    make_txt: bool = False,
    cleanup: bool = False,
    yomitoku_exe: Optional[str] = None,
    log: LogFn = lambda s: None,
    cancel_event=None,
) -> Dict[str, ScanOutputs]:
    """
    Procesa un PDF o una carpeta (batch). Devuelve un dict: {pdf_path: ScanOutputs}
    """
    in_path = Path(input_path)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    exe = yomitoku_exe.strip() if (yomitoku_exe and yomitoku_exe.strip()) else _default_yomitoku_exe()

    if batch:
        pdfs = sorted([p for p in in_path.glob("*.pdf")]) if in_path.is_dir() else [in_path]
    else:
        pdfs = [in_path]

    pdfs = [p for p in pdfs if p.is_file() and p.suffix.lower() == ".pdf"]
    if not pdfs:
        raise RuntimeError("No encontré PDFs para procesar.")

    results: Dict[str, ScanOutputs] = {}

    for idx, pdf_path in enumerate(pdfs, start=1):
        _check_cancel(cancel_event)
        log(f"={'='*70}")
        log(f"[{idx}/{len(pdfs)}] {pdf_path.name}")
        outputs = _process_one_pdf(
            pdf_path=pdf_path,
            out_dir=out_dir_p,
            upper_ratio=upper_ratio,
            lower_ratio=lower_ratio,
            reuse_images=reuse_images,
            reuse_md=reuse_md,
            make_excel=make_excel,
            make_ass=make_ass,
            make_txt=make_txt,
            cleanup=cleanup,
            yomitoku_exe=exe,
            log=log,
            cancel_event=cancel_event,
        )
        results[str(pdf_path)] = outputs

    return results


def _process_one_pdf(
    pdf_path: Path,
    out_dir: Path,
    upper_ratio: Optional[float],
    lower_ratio: Optional[float],
    reuse_images: bool,
    reuse_md: bool,
    make_excel: bool,
    make_ass: bool,
    make_txt: bool,
    cleanup: bool,
    yomitoku_exe: str,
    log: LogFn,
    cancel_event=None,
) -> ScanOutputs:
    stem = pdf_path.stem

    # Intermedios se guardan junto al PDF por compatibilidad con tus carpetas existentes
    base_dir = pdf_path.parent

    cropped_pdf_path = base_dir / f"{stem}_CROPPED.pdf"
    images_dir = base_dir / f"{stem}_images"
    md_dir = base_dir / f"{stem}_yomi_md"
    images_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    # A) Reusar MD
    if reuse_md:
        md_files = [p for p in md_dir.rglob("*.md") if re.search(r"_p\d{3}s\d{3}", p.name)]
        if md_files:
            log(f"Reuse MD: {len(md_files)} MD encontrados -> reconstruyendo sin OCR.")
            df = _df_from_md_files(md_files, log=log)
            return _write_outputs(pdf_path, out_dir, df, make_excel, make_ass, make_txt, log, cancel_event)

    # B) Reusar imágenes
    if reuse_images:
        col_images = sorted(images_dir.glob(f"{stem}_p???s???.png"))
        if col_images:
            log(f"Reuse imágenes: {len(col_images)} columnas encontradas.")
            md_index = _index_md_by_page_section(md_dir)
            rows: List[dict] = []
            for img_path in col_images:
                _check_cancel(cancel_event)
                m = re.search(r"_p(\d{3})s(\d{3})", img_path.name)
                if not m:
                    continue
                page_num = int(m.group(1))
                sec = int(m.group(2))

                md_path = md_index.get((page_num, sec))
                if md_path:
                    md = _post_edit_and_persist_md(md_path, log)
                    merged = merge_jp_lines(md)
                    merged = merge_unbalanced_jp_quotes(merged)
                    block = "\n".join(merged)
                    dialogues = parse_dialogues_from_text(block)
                    for d in dialogues:
                        rows.append({"pagina": page_num, "seccion": sec,
                                     "actor": (d.get("actor") or "").strip(),
                                     "texto": (d.get("texto") or "").strip()})
                else:
                    rows.extend(_ocr_one_column_to_rows(yomitoku_exe, img_path, page_num, sec, md_dir, log))
            df = finalize_df(pd.DataFrame(rows)) if rows else pd.DataFrame(columns=["pagina", "seccion", "actor", "texto"])
            return _write_outputs(pdf_path, out_dir, df, make_excel, make_ass, make_txt, log, cancel_event)

    # C) Pipeline completo
    if upper_ratio is None or lower_ratio is None:
        raise RuntimeError("Faltan cortes upper/lower. Usa la vista previa o activa reuse.")

    doc = fitz.open(str(pdf_path))
    try:
        crop_doc = fitz.open()
        crop_clips: List[Tuple[float, float, float, float]] = []
        for pi in range(len(doc)):
            _check_cancel(cancel_event)
            page = doc.load_page(pi)
            rect = page.rect
            y0 = max(0.0, min(rect.height, rect.height * float(upper_ratio)))
            y1 = max(0.0, min(rect.height, rect.height * float(lower_ratio)))
            if y1 <= y0 + 1:
                y0, y1 = rect.height * 0.4, rect.height * 0.9
            clip = fitz.Rect(rect.x0, y0, rect.x1, y1)
            crop_clips.append((clip.x0, clip.y0, clip.x1, clip.y1))

            new_page = crop_doc.new_page(width=clip.width, height=clip.height)
            new_page.show_pdf_page(new_page.rect, doc, pi, clip=clip)

        crop_doc.save(str(cropped_pdf_path))
        crop_doc.close()
        log(f"PDF recortado: {cropped_pdf_path.name}")

        rows: List[dict] = []
        for pi in range(len(doc)):
            _check_cancel(cancel_event)
            page_num = pi + 1
            page = doc.load_page(pi)
            clip = fitz.Rect(*crop_clips[pi])

            zoom = OCR_DPI / 72.0
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            elif pix.n == 3:
                bgr = arr
            else:
                bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

            # Guardar recorte página (debug/cache)
            crop_img_path = images_dir / f"{stem}_p{page_num:03d}_crop.png"
            cv_imwrite_u(crop_img_path, bgr)

            lines_x = detect_vertical_separators_long_global(bgr, dpi=OCR_DPI)
            cols = slice_columns_from_image(bgr, lines_x)
            if not cols and has_visible_ink_image(bgr):
                cols = [(1, bgr)]

            for sec, col_img in cols:
                col_img_path = images_dir / f"{stem}_p{page_num:03d}s{sec:03d}.png"
                cv_imwrite_u(col_img_path, col_img)
                rows.extend(_ocr_one_column_to_rows(yomitoku_exe, col_img_path, page_num, sec, md_dir, log))

        df = finalize_df(pd.DataFrame(rows)) if rows else pd.DataFrame(columns=["pagina", "seccion", "actor", "texto"])
        outputs = _write_outputs(pdf_path, out_dir, df, make_excel, make_ass, make_txt, log, cancel_event)

        if cleanup:
            # Por defecto, NO borro _images/_yomi_md porque sirven de cache.
            # Solo borro el CROPPED para evitar confusión.
            try:
                if cropped_pdf_path.exists():
                    cropped_pdf_path.unlink()
            except Exception:
                pass

        return outputs

    finally:
        doc.close()


def _write_outputs(
    pdf_path: Path,
    out_dir: Path,
    df: pd.DataFrame,
    make_excel: bool,
    make_ass: bool,
    make_txt: bool,
    log: LogFn,
    cancel_event=None,
) -> ScanOutputs:
    stem = pdf_path.stem
    out = ScanOutputs()

    if df is None or df.empty:
        log("Sin resultados -> no se generan salidas.")
        return out

    df = df.sort_values(["pagina", "seccion"], kind="stable").reset_index(drop=True)

    # Limpieza final común
    for col in ("actor", "texto"):
        if col in df.columns:
            # Quitar todas las instancias de \F\)
            df[col] = df[col].astype(str).str.replace(r"\F\)", "", regex=False)

    if make_excel:
        out_xlsx = out_dir / f"{stem}_scanner.xlsx"
        try:
            df.to_excel(out_xlsx, index=False)
            out.xlsx = str(out_xlsx)
            log(f"Excel: {out_xlsx}")
        except Exception as e:
            log(f"ERROR Excel (¿openpyxl instalado?): {e}")

    if make_txt:
        out_txt = out_dir / f"{stem}_scanner.txt"
        write_txt(df, out_txt)
        out.txt = str(out_txt)
        log(f"TXT: {out_txt}")

    if make_ass:
        out_ass = out_dir / f"{stem}_scanner.ass"
        df_to_ass_with_styles(df, out_ass, style_name="Gen_Main", log=log, cancel_event=cancel_event)
        out.ass = str(out_ass)
        log(f"ASS: {out_ass}")

    return out


# =========================
#  CLI compat (para panel.py)
# =========================
def main(argv: Optional[List[str]] = None) -> None:
    """
    Entry-point pensado para ser llamado desde el panel (sys.argv) o CLI.
    """
    import argparse

    p = argparse.ArgumentParser(prog="scanner", add_help=True)
    p.add_argument("input", help="PDF o carpeta con PDFs (si --batch).")
    p.add_argument("--out-dir", default="", help="Directorio de salida (por defecto, junto al PDF).")
    p.add_argument("--batch", action="store_true", help="Procesar todos los PDFs de una carpeta.")
    p.add_argument("--upper", type=float, default=None, help="Ratio superior (0..1) del recorte.")
    p.add_argument("--lower", type=float, default=None, help="Ratio inferior (0..1) del recorte.")
    p.add_argument("--no-reuse-images", action="store_true")
    p.add_argument("--no-reuse-md", action="store_true")
    p.add_argument("--no-excel", action="store_true")
    p.add_argument("--no-ass", action="store_true")
    p.add_argument("--txt", action="store_true")
    p.add_argument("--cleanup", action="store_true")
    p.add_argument("--yomitoku-exe", default="", help="Ruta a yomitoku(.exe).")

    args = p.parse_args(argv)

    input_path = args.input
    out_dir = args.out_dir.strip() or str(Path(input_path).parent)

    def _log(s: str):
        print(s)

    run_scanner(
        input_path=input_path,
        out_dir=out_dir,
        batch=bool(args.batch),
        upper_ratio=args.upper,
        lower_ratio=args.lower,
        reuse_images=not bool(args.no_reuse_images),
        reuse_md=not bool(args.no_reuse_md),
        make_excel=not bool(args.no_excel),
        make_ass=not bool(args.no_ass),
        make_txt=bool(args.txt),
        cleanup=bool(args.cleanup),
        yomitoku_exe=(args.yomitoku_exe or "").strip() or None,
        log=_log,
    )
