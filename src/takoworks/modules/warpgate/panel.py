from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from ...config import save_config

def _configure_vlc_runtime() -> Optional[str]:
    candidates: List[Path] = []
    python_bits = int(struct.calcsize("P") * 8)
    found_x86_only = False

    env_lib = os.environ.get("PYTHON_VLC_LIB_PATH", "").strip().strip('"')
    if env_lib:
        env_lib_path = Path(env_lib)
        if env_lib_path.name.lower() == "libvlc.dll":
            candidates.append(env_lib_path.parent)
        else:
            candidates.append(env_lib_path)

    env_dir = os.environ.get("TAKOWORKS_VLC_DIR", "").strip().strip('"')
    if env_dir:
        candidates.append(Path(env_dir))

    for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(env_name, "").strip()
        if base:
            candidates.append(Path(base) / "VideoLAN" / "VLC")

    local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
    if local_appdata:
        candidates.append(Path(local_appdata) / "Programs" / "VideoLAN" / "VLC")

    seen = set()
    for folder in candidates:
        key = str(folder).lower()
        if not key or key in seen:
            continue
        seen.add(key)

        lib_dll = folder / "libvlc.dll"
        if not lib_dll.is_file():
            continue
        if python_bits == 64 and "program files (x86)" in key:
            found_x86_only = True
            continue

        os.environ["PYTHON_VLC_LIB_PATH"] = str(lib_dll)
        plugins = folder / "plugins"
        if plugins.is_dir():
            os.environ["PYTHON_VLC_MODULE_PATH"] = str(plugins)

        try:
            os.add_dll_directory(str(folder))
        except Exception:
            pass

        path = os.environ.get("PATH", "")
        entries = {p.strip().lower() for p in path.split(os.pathsep) if p.strip()}
        if str(folder).lower() not in entries:
            os.environ["PATH"] = f"{folder}{os.pathsep}{path}" if path else str(folder)
        return None

    if found_x86_only:
        return "Se detecto VLC de 32 bits en Program Files (x86), pero TakoWorks usa Python de 64 bits. Instala VLC de 64 bits."
    return None


_vlc_runtime_hint = _configure_vlc_runtime()
_vlc_import_error: Optional[str] = _vlc_runtime_hint

try:
    import vlc  # type: ignore
except Exception as exc:
    vlc = None
    if _vlc_runtime_hint:
        _vlc_import_error = f"{_vlc_runtime_hint}\n{exc}"
    else:
        _vlc_import_error = str(exc)


DEFAULT_FORMAT_FIELDS = [
    "Layer",
    "Start",
    "End",
    "Style",
    "Name",
    "MarginL",
    "MarginR",
    "MarginV",
    "Effect",
    "Text",
]


def ass_time_to_seconds(t: str) -> Optional[float]:
    t = (t or "").strip()
    if not t:
        return None
    parts = t.split(":")
    if len(parts) != 3:
        return None
    try:
        h = int(parts[0])
        m = int(parts[1])
        s_part = parts[2]
        if "." in s_part:
            s_str, cs_str = s_part.split(".", 1)
            s = int(s_str)
            cs = int(cs_str)
        else:
            s = int(s_part)
            cs = 0
        return h * 3600.0 + m * 60.0 + s + (cs / 100.0)
    except Exception:
        return None


def seconds_to_hms(t: float) -> str:
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int(round((t - int(t)) * 100.0))
    return f"{h:02d}:{m:02d}:{s:02d}.{cs:02d}"


def _ensure_len(parts: List[str], n: int) -> None:
    if len(parts) < n:
        parts.extend([""] * (n - len(parts)))


def _parse_format_fields(line: str) -> List[str]:
    raw = line.split(":", 1)[1] if ":" in line else ""
    return [x.strip() for x in raw.split(",") if x.strip()]


def _format_line_from_fields(fields: List[str]) -> str:
    return "Format: " + ",".join(fields)


def _indices_from_fields(fields: List[str]) -> Dict[str, int]:
    def idx_of(name: str) -> int:
        for i, f in enumerate(fields):
            if f.strip().lower() == name.lower():
                return i
        return -1

    out = {
        "Start": idx_of("Start"),
        "End": idx_of("End"),
        "Name": idx_of("Name"),
        "Text": idx_of("Text"),
        "Style": idx_of("Style"),
    }
    if out["Start"] < 0:
        out["Start"] = 1
    if out["End"] < 0:
        out["End"] = 2
    if out["Name"] < 0:
        out["Name"] = 4
    if out["Text"] < 0:
        out["Text"] = len(fields) - 1 if fields else 9
    return out


@dataclass
class AssEvent:
    kind: str
    fields: List[str]
    start_s: Optional[float] = None
    end_s: Optional[float] = None
    warped: bool = False

    def get(self, idx: int) -> str:
        if idx < 0 or idx >= len(self.fields):
            return ""
        return self.fields[idx] or ""

    def set(self, idx: int, value: str) -> None:
        if idx < 0:
            return
        _ensure_len(self.fields, idx + 1)
        self.fields[idx] = value


@dataclass
class AssDocument:
    path: str
    before_lines: List[str]
    after_lines: List[str]
    meta_lines: List[str]
    format_fields: List[str]
    format_line: str
    indices: Dict[str, int]
    events: List[AssEvent]

    @classmethod
    def load(cls, path: str) -> "AssDocument":
        text = ""
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            text = f.read()
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = text.split("\n")

        before: List[str] = []
        after: List[str] = []
        events_lines: List[str] = []
        mode = "before"
        for line in lines:
            stripped = line.strip()
            if stripped.lower() == "[events]":
                mode = "events"
                continue
            if mode == "events" and stripped.startswith("[") and stripped.lower() != "[events]":
                mode = "after"
            if mode == "events":
                events_lines.append(line)
            elif mode == "before":
                before.append(line)
            else:
                after.append(line)

        format_fields: List[str] = []
        format_line = ""
        meta_lines: List[str] = []
        events: List[AssEvent] = []

        for line in events_lines:
            stripped = line.strip()
            low = stripped.lower()
            if low.startswith("format:"):
                format_fields = _parse_format_fields(stripped)
                format_line = stripped
                continue
            if low.startswith("dialogue:") or low.startswith("comment:"):
                kind = "Dialogue" if low.startswith("dialogue:") else "Comment"
                payload = stripped.split(":", 1)[1].lstrip()
                fields_ref = format_fields or DEFAULT_FORMAT_FIELDS
                parts = payload.split(",", maxsplit=len(fields_ref) - 1)
                _ensure_len(parts, len(fields_ref))
                indices = _indices_from_fields(fields_ref)
                start = parts[indices["Start"]].strip() if indices["Start"] >= 0 else ""
                end = parts[indices["End"]].strip() if indices["End"] >= 0 else ""
                ev = AssEvent(
                    kind=kind,
                    fields=parts,
                    start_s=ass_time_to_seconds(start),
                    end_s=ass_time_to_seconds(end),
                )
                events.append(ev)
            else:
                meta_lines.append(line)

        if not format_fields:
            format_fields = list(DEFAULT_FORMAT_FIELDS)
        if not format_line:
            format_line = _format_line_from_fields(format_fields)
        indices = _indices_from_fields(format_fields)

        return cls(
            path=path,
            before_lines=before,
            after_lines=after,
            meta_lines=meta_lines,
            format_fields=format_fields,
            format_line=format_line,
            indices=indices,
            events=events,
        )

    def render_event(self, ev: AssEvent) -> str:
        return f"{ev.kind}: " + ",".join(ev.fields)

    def to_string(self) -> str:
        lines: List[str] = []
        lines.extend(self.before_lines)
        lines.append("[Events]")
        lines.append(self.format_line)
        lines.extend(self.meta_lines)
        for ev in self.events:
            lines.append(self.render_event(ev))
        lines.extend(self.after_lines)
        if lines and lines[-1] != "":
            lines.append("")
        return "\n".join(lines)


class WarpGatePanel(ttk.Frame):
    def __init__(self, parent, runner, cfg: dict):
        super().__init__(parent)
        self.runner = runner
        self.cfg = cfg

        last = cfg.get("last", {})
        self.video_var = tk.StringVar(value=last.get("warpgate_video", ""))
        self.ass_a_var = tk.StringVar(value=last.get("warpgate_ass_a", ""))
        self.ass_b_var = tk.StringVar(value=last.get("warpgate_ass_b", ""))

        self.doc_a: Optional[AssDocument] = None
        self.doc_b: Optional[AssDocument] = None

        self.video_cap: Optional[cv2.VideoCapture] = None
        self.video_fps: float = 0.0
        self.video_frame_count: int = 0
        self.video_duration: float = 0.0
        self.video_playing: bool = False
        self.video_after_id: Optional[str] = None
        self.video_photo: Optional[tk.PhotoImage] = None
        self.video_last_frame = None
        self.vlc_instance = None
        self.vlc_player = None
        self.vlc_media = None
        self.vlc_timer_after_id: Optional[str] = None
        self.vlc_error: str = _vlc_import_error or ""
        self.video_mode: str = "none"
        self.undo_stack: List[dict] = []

        self._scroll_syncing = False
        self._entry_updating = False

        self._build()
        self._bind_shortcuts()

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)

        loaders = ttk.Frame(self)
        loaders.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        loaders.columnconfigure(1, weight=1)

        r0 = ttk.Frame(loaders)
        r0.grid(row=0, column=0, columnspan=3, sticky="ew", pady=2)
        ttk.Label(r0, text="Video").pack(side="left")
        ttk.Entry(r0, textvariable=self.video_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r0, text="Browse", command=self._pick_video).pack(side="left")

        r1 = ttk.Frame(loaders)
        r1.grid(row=1, column=0, columnspan=3, sticky="ew", pady=2)
        ttk.Label(r1, text="File A (timed)").pack(side="left")
        ttk.Entry(r1, textvariable=self.ass_a_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r1, text="Browse", command=self._pick_ass_a).pack(side="left")

        r2 = ttk.Frame(loaders)
        r2.grid(row=2, column=0, columnspan=3, sticky="ew", pady=2)
        ttk.Label(r2, text="File B (untimed)").pack(side="left")
        ttk.Entry(r2, textvariable=self.ass_b_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(r2, text="Browse", command=self._pick_ass_b).pack(side="left")

        ttk.Button(loaders, text="Load", command=self._load_all).grid(row=0, column=3, rowspan=3, padx=(8, 0))

        top = ttk.Frame(self)
        top.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 6))
        top.columnconfigure(0, weight=1)
        top.rowconfigure(0, weight=1)
        top.rowconfigure(1, weight=0)

        video_row = ttk.Frame(top)
        video_row.grid(row=0, column=0, sticky="nsew")
        video_row.columnconfigure(0, weight=1)

        video_box = ttk.Frame(video_row)
        video_box.grid(row=0, column=0, sticky="nsew")
        video_box.columnconfigure(0, weight=1)
        video_box.rowconfigure(0, weight=1)

        self.video_container = tk.Frame(video_box, bg="black")
        self.video_container.grid(row=0, column=0, sticky="nsew")
        self.video_container.bind("<Configure>", lambda _e: self._refresh_video_frame())
        self.video_label = ttk.Label(self.video_container, text="No video loaded", anchor="center")
        self.video_label.place(relx=0, rely=0, relwidth=1, relheight=1)

        controls = ttk.Frame(video_box)
        controls.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        ttk.Button(controls, text="Play/Pause", command=self._toggle_play).pack(side="left")
        ttk.Button(controls, text="-5s", command=lambda: self._seek_by(-5.0)).pack(side="left", padx=4)
        ttk.Button(controls, text="+5s", command=lambda: self._seek_by(5.0)).pack(side="left")
        self.video_time_label = ttk.Label(controls, text="00:00:00.00 / 00:00:00.00")
        self.video_time_label.pack(side="left", padx=8)

        btns = ttk.Frame(video_row)
        btns.grid(row=0, column=1, sticky="ns", padx=(10, 0))

        for text, cmd in [
            ("Merge", self._merge_b),
            ("Cut", self._cut_b),
            ("Delete", self._delete_line),
            ("Copy >", self._copy_a_to_b),
            ("< Copy", self._copy_b_to_a),
            ("New", self._new_line),
            ("Warp", self._warp_transfer),
            ("Export", self._export_a),
        ]:
            ttk.Button(btns, text=text, command=cmd, width=10).pack(side="top", fill="x", pady=2)

        line_box = ttk.Frame(top)
        line_box.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        line_box.columnconfigure(1, weight=1)

        self.a_info_var = tk.StringVar(value="A: ")
        self.b_info_var = tk.StringVar(value="B: ")
        self.a_text_var = tk.StringVar(value="")
        self.b_text_var = tk.StringVar(value="")

        ttk.Label(line_box, textvariable=self.a_info_var).grid(row=0, column=0, sticky="w")
        self.a_entry = ttk.Entry(line_box, textvariable=self.a_text_var, state="readonly")
        self.a_entry.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        ttk.Label(line_box, textvariable=self.b_info_var).grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.b_entry = ttk.Entry(line_box, textvariable=self.b_text_var)
        self.b_entry.grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=(4, 0))
        self.b_entry.bind("<Return>", lambda _e: self._commit_b_text())
        self.b_entry.bind("<FocusOut>", lambda _e: self._commit_b_text())

        bottom = ttk.Frame(self)
        bottom.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
        bottom.columnconfigure(0, weight=1)
        bottom.columnconfigure(1, weight=1)
        bottom.rowconfigure(1, weight=1)

        ttk.Label(bottom, text="File A").grid(row=0, column=0, sticky="w")
        ttk.Label(bottom, text="File B").grid(row=0, column=1, sticky="w")

        self.list_a = tk.Listbox(bottom, selectmode=tk.EXTENDED)
        self.list_b = tk.Listbox(bottom, selectmode=tk.EXTENDED)
        self.list_a.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        self.list_b.grid(row=1, column=1, sticky="nsew", padx=(6, 0))

        self.scrollbar = ttk.Scrollbar(bottom, orient="vertical", command=self._on_scrollbar)
        self.scrollbar.grid(row=1, column=2, sticky="ns")
        self.list_a.configure(yscrollcommand=self._on_list_scroll)
        self.list_b.configure(yscrollcommand=self._on_list_scroll)

        self.list_a.bind("<<ListboxSelect>>", self._on_select_a)
        self.list_b.bind("<<ListboxSelect>>", self._on_select_b)

        self.list_a.bind("<MouseWheel>", self._on_mousewheel)
        self.list_b.bind("<MouseWheel>", self._on_mousewheel)
        self.list_a.bind("<FocusIn>", lambda _e: self._set_focus("a"))
        self.list_b.bind("<FocusIn>", lambda _e: self._set_focus("b"))
        self.b_entry.bind("<FocusIn>", lambda _e: self._set_focus("b"))

        self._focus_target = "a"

    def _bind_shortcuts(self) -> None:
        for widget in (self.list_a, self.list_b, self.b_entry):
            widget.bind("<Up>", self._move_both_up)
            widget.bind("<Down>", self._move_both_down)
            widget.bind("<Control-Up>", self._move_a_up)
            widget.bind("<Control-Down>", self._move_a_down)
            widget.bind("<Alt-Up>", self._move_b_up)
            widget.bind("<Alt-Down>", self._move_b_down)
            widget.bind("<F4>", lambda _e: self._merge_b() or "break")
            widget.bind("<Control-F4>", lambda _e: self._cut_b() or "break")
            widget.bind("<Delete>", lambda _e: self._delete_line() or "break")
            widget.bind("<Control-Right>", lambda _e: self._copy_a_to_b() or "break")
            widget.bind("<Control-Left>", lambda _e: self._copy_b_to_a() or "break")
            widget.bind("<Control-z>", lambda _e: self._undo_last() or "break")
            widget.bind("<Control-Z>", lambda _e: self._undo_last() or "break")

    def _clone_doc(self, doc: Optional[AssDocument]) -> Optional[AssDocument]:
        if doc is None:
            return None
        events = [
            AssEvent(
                kind=ev.kind,
                fields=list(ev.fields),
                start_s=ev.start_s,
                end_s=ev.end_s,
                warped=ev.warped,
            )
            for ev in doc.events
        ]
        return AssDocument(
            path=doc.path,
            before_lines=list(doc.before_lines),
            after_lines=list(doc.after_lines),
            meta_lines=list(doc.meta_lines),
            format_fields=list(doc.format_fields),
            format_line=str(doc.format_line),
            indices=dict(doc.indices),
            events=events,
        )

    def _snapshot_state(self) -> dict:
        return {
            "doc_a": self._clone_doc(self.doc_a),
            "doc_b": self._clone_doc(self.doc_b),
            "sel_a": list(self.list_a.curselection()),
            "sel_b": list(self.list_b.curselection()),
            "focus": self._focus_target,
        }

    def _push_undo(self) -> None:
        self.undo_stack.append(self._snapshot_state())
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def _restore_state(self, state: dict) -> None:
        self.doc_a = state.get("doc_a")
        self.doc_b = state.get("doc_b")
        self._refresh_lists()
        self.list_a.selection_clear(0, tk.END)
        self.list_b.selection_clear(0, tk.END)
        for idx in state.get("sel_a", []):
            if idx < self.list_a.size():
                self.list_a.selection_set(idx)
        for idx in state.get("sel_b", []):
            if idx < self.list_b.size():
                self.list_b.selection_set(idx)
        self._focus_target = state.get("focus", "a")
        self._update_line_display()
        a_idx = self._get_first_selection(self.list_a)
        if a_idx is not None:
            self._seek_from_a(a_idx)

    def _undo_last(self) -> None:
        if not self.undo_stack:
            return
        state = self.undo_stack.pop()
        self._restore_state(state)

    def _set_focus(self, target: str) -> None:
        self._focus_target = target

    def _pick_video(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("Video", "*.mkv *.mp4 *.avi *.mov *.ts"), ("All files", "*.*")])
        if p:
            self.video_var.set(p)

    def _pick_ass_a(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("ASS/SSA", "*.ass *.ssa"), ("All files", "*.*")])
        if p:
            self.ass_a_var.set(p)

    def _pick_ass_b(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("ASS/SSA", "*.ass *.ssa"), ("All files", "*.*")])
        if p:
            self.ass_b_var.set(p)

    def _load_all(self) -> None:
        video_path = self.video_var.get().strip()
        ass_a = self.ass_a_var.get().strip()
        ass_b = self.ass_b_var.get().strip()

        if not video_path or not os.path.isfile(video_path):
            messagebox.showerror("WarpGate", "Select a valid video.")
            return
        if not ass_a or not os.path.isfile(ass_a):
            messagebox.showerror("WarpGate", "Select a valid File A.")
            return
        if not ass_b or not os.path.isfile(ass_b):
            messagebox.showerror("WarpGate", "Select a valid File B.")
            return

        try:
            self.doc_a = AssDocument.load(ass_a)
        except Exception as exc:
            messagebox.showerror("WarpGate", f"Could not read File A: {exc}")
            return
        try:
            self.doc_b = AssDocument.load(ass_b)
        except Exception as exc:
            messagebox.showerror("WarpGate", f"Could not read File B: {exc}")
            return

        if not self.doc_a.events:
            messagebox.showwarning("WarpGate", "File A has no events in [Events].")
        if not self.doc_b.events:
            messagebox.showwarning("WarpGate", "File B has no events in [Events].")

        self._load_video(video_path)
        self._refresh_lists()

        self.cfg.setdefault("last", {})["warpgate_video"] = video_path
        self.cfg.setdefault("last", {})["warpgate_ass_a"] = ass_a
        self.cfg.setdefault("last", {})["warpgate_ass_b"] = ass_b
        save_config(self.cfg)

    def _load_video(self, path: str) -> None:
        self._stop_video()
        if self._load_video_vlc(path):
            return
        detail = self.vlc_error.strip()
        if detail:
            messagebox.showerror("WarpGate", f"Could not start VLC. Preview requires VLC.\n\nDetails: {detail}")
            return
        messagebox.showerror("WarpGate", "Could not start VLC. Preview requires VLC.")

    def _load_video_vlc(self, path: str) -> bool:
        if vlc is None:
            self.vlc_error = _vlc_import_error or "Could not import python-vlc."
            return False
        if not self._init_vlc_player():
            return False
        try:
            self.vlc_media = self.vlc_instance.media_new(path)
            self.vlc_player.set_media(self.vlc_media)
            self.video_mode = "vlc"
            self._show_video_label(False)
            self.vlc_player.play()
            self.video_playing = True
            self._schedule_vlc_timer()
            self.after(250, self._vlc_pause_on_load)
            self.vlc_error = ""
            return True
        except Exception as exc:
            self.vlc_error = str(exc)
            return False

    def _vlc_pause_on_load(self) -> None:
        if not self.vlc_player:
            return
        try:
            self.vlc_player.pause()
        except Exception:
            return
        self.video_playing = False
        self._update_timer()

    def _init_vlc_player(self) -> bool:
        try:
            if self.vlc_instance is None:
                self.vlc_instance = vlc.Instance()  # type: ignore[call-arg]
            if self.vlc_player is None:
                self.vlc_player = self.vlc_instance.media_player_new()
            self.video_container.update_idletasks()
            handle = self.video_container.winfo_id()
            if handle:
                try:
                    self.vlc_player.set_hwnd(handle)
                except Exception as exc:
                    self.vlc_error = str(exc)
                    return False
            return True
        except Exception as exc:
            self.vlc_error = str(exc)
            return False

    def _show_video_label(self, show: bool) -> None:
        if show:
            self.video_label.place(relx=0, rely=0, relwidth=1, relheight=1)
        else:
            self.video_label.place_forget()

    def _schedule_vlc_timer(self) -> None:
        if self.vlc_timer_after_id:
            try:
                self.after_cancel(self.vlc_timer_after_id)
            except Exception:
                pass
        self.vlc_timer_after_id = self.after(200, self._vlc_timer_loop)

    def _vlc_timer_loop(self) -> None:
        self._update_timer()
        if self.video_playing and self.video_mode == "vlc":
            self.vlc_timer_after_id = self.after(200, self._vlc_timer_loop)

    def _stop_video(self) -> None:
        self.video_playing = False
        if self.video_after_id:
            try:
                self.after_cancel(self.video_after_id)
            except Exception:
                pass
            self.video_after_id = None
        if self.vlc_timer_after_id:
            try:
                self.after_cancel(self.vlc_timer_after_id)
            except Exception:
                pass
            self.vlc_timer_after_id = None
        if self.vlc_player:
            try:
                self.vlc_player.stop()
            except Exception:
                pass
        self.video_mode = "none"
        self._show_video_label(True)
        self.video_label.configure(text="No video loaded")

    def _toggle_play(self) -> None:
        if self.video_mode == "vlc":
            if not self.vlc_player:
                return
            if self.video_playing:
                try:
                    self.vlc_player.pause()
                except Exception:
                    pass
                self.video_playing = False
            else:
                try:
                    self.vlc_player.play()
                except Exception:
                    return
                self.video_playing = True
                self._schedule_vlc_timer()
            return

        return

    def _play_step(self) -> None:
        return

    def _seek_by(self, delta_s: float) -> None:
        if self.video_mode == "vlc":
            if not self.vlc_player:
                return
            try:
                cur_ms = float(self.vlc_player.get_time() or 0.0)
            except Exception:
                return
            self._seek_to(max(0.0, (cur_ms / 1000.0) + delta_s))
            return

    def _seek_to(self, seconds: float) -> None:
        if self.video_mode == "vlc":
            if not self.vlc_player:
                return
            try:
                self.vlc_player.set_time(int(max(0.0, seconds * 1000.0)))
            except Exception:
                return
            self._update_timer()
            return

    def _refresh_video_frame(self) -> None:
        if self.video_mode == "vlc":
            return
        if self.video_last_frame is not None:
            self._display_frame(self.video_last_frame)

    def _display_frame(self, frame) -> None:
        if self.video_mode == "vlc":
            return
        if frame is None:
            return
        return

    def _update_timer(self) -> None:
        if self.video_mode == "vlc":
            if not self.vlc_player:
                self.video_time_label.configure(text="00:00:00.00 / 00:00:00.00")
                return
            try:
                cur_ms = float(self.vlc_player.get_time() or 0.0)
                total_ms = float(self.vlc_player.get_length() or 0.0)
            except Exception:
                self.video_time_label.configure(text="00:00:00.00 / 00:00:00.00")
                return
            cur_s = max(0.0, cur_ms / 1000.0)
            total = max(0.0, total_ms / 1000.0)
            self.video_time_label.configure(text=f"{seconds_to_hms(cur_s)} / {seconds_to_hms(total)}")
            return
        if not self.video_cap:
            self.video_time_label.configure(text="00:00:00.00 / 00:00:00.00")
            return
        cur_ms = float(self.video_cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        cur_s = cur_ms / 1000.0
        total = self.video_duration
        self.video_time_label.configure(text=f"{seconds_to_hms(cur_s)} / {seconds_to_hms(total)}")

    def _refresh_lists(self) -> None:
        self._refresh_list(self.list_a, self.doc_a, is_a=True)
        self._refresh_list(self.list_b, self.doc_b, is_a=False)
        self._update_line_display()

    def _refresh_list(self, lb: tk.Listbox, doc: Optional[AssDocument], *, is_a: bool) -> None:
        lb.delete(0, tk.END)
        if not doc:
            return
        idx_text = doc.indices["Text"]
        idx_name = doc.indices["Name"]
        idx_start = doc.indices["Start"]
        idx_end = doc.indices["End"]
        for i, ev in enumerate(doc.events):
            text = ev.get(idx_text).strip()
            actor = ev.get(idx_name).strip()
            if is_a:
                start = ev.get(idx_start).strip()
                end = ev.get(idx_end).strip()
                time_part = f"{start}->{end}" if (start or end) else "--"
                line = f"{i+1:04d} | {time_part} | {actor} | {text}"
            else:
                line = f"{i+1:04d} | {actor} | {text}"
            lb.insert(tk.END, line)
            if ev.warped:
                lb.itemconfig(i, background="#b8f2b8")

    def _update_line_display(self) -> None:
        self._entry_updating = True
        try:
            self.a_info_var.set(self._format_info(self.doc_a, self._get_first_selection(self.list_a), "A"))
            self.b_info_var.set(self._format_info(self.doc_b, self._get_first_selection(self.list_b), "B"))
            self.a_text_var.set(self._get_text_from_doc(self.doc_a, self._get_first_selection(self.list_a)))
            self.b_text_var.set(self._get_text_from_doc(self.doc_b, self._get_first_selection(self.list_b)))
        finally:
            self._entry_updating = False

    def _format_info(self, doc: Optional[AssDocument], idx: Optional[int], label: str) -> str:
        if not doc or idx is None or idx < 0 or idx >= len(doc.events):
            return f"{label}:"
        ev = doc.events[idx]
        name = ev.get(doc.indices["Name"]).strip()
        if label == "A":
            start = ev.get(doc.indices["Start"]).strip()
            end = ev.get(doc.indices["End"]).strip()
            time_part = f"{start}->{end}" if (start or end) else "--"
            return f"{label}: {time_part} | {name}"
        return f"{label}: {name}"

    def _get_text_from_doc(self, doc: Optional[AssDocument], idx: Optional[int]) -> str:
        if not doc or idx is None or idx < 0 or idx >= len(doc.events):
            return ""
        ev = doc.events[idx]
        return ev.get(doc.indices["Text"])

    def _commit_b_text(self) -> None:
        if self._entry_updating:
            return
        doc = self.doc_b
        if not doc:
            return
        idx = self._get_first_selection(self.list_b)
        if idx is None or idx < 0 or idx >= len(doc.events):
            return
        new_text = self.b_text_var.get()
        old_text = doc.events[idx].get(doc.indices["Text"])
        if new_text != old_text:
            self._push_undo()
        doc.events[idx].set(doc.indices["Text"], new_text)
        self._refresh_list(self.list_b, doc, is_a=False)
        self.list_b.selection_set(idx)

    def _get_first_selection(self, lb: tk.Listbox) -> Optional[int]:
        sel = lb.curselection()
        if not sel:
            return None
        return int(sel[0])

    def _on_select_a(self, _evt=None) -> None:
        idx = self._get_first_selection(self.list_a)
        self._update_line_display()
        self._seek_from_a(idx)

    def _on_select_b(self, _evt=None) -> None:
        self._update_line_display()

    def _seek_from_a(self, idx: Optional[int]) -> None:
        if not self.video_cap or not self.doc_a or idx is None:
            return
        if idx < 0 or idx >= len(self.doc_a.events):
            return
        ev = self.doc_a.events[idx]
        if ev.start_s is None:
            return
        self._seek_to(ev.start_s)

    def _on_scrollbar(self, *args) -> None:
        self.list_a.yview(*args)
        self.list_b.yview(*args)

    def _on_list_scroll(self, *args) -> None:
        self.scrollbar.set(*args)
        if self._scroll_syncing:
            return
        self._scroll_syncing = True
        try:
            if args:
                self.list_a.yview_moveto(args[0])
                self.list_b.yview_moveto(args[0])
        finally:
            self._scroll_syncing = False

    def _on_mousewheel(self, event) -> str:
        delta = int(-1 * (event.delta / 120))
        self.list_a.yview_scroll(delta, "units")
        self.list_b.yview_scroll(delta, "units")
        return "break"

    def _move_selection(self, lb: tk.Listbox, direction: int) -> None:
        size = lb.size()
        if size == 0:
            return
        idx = self._get_first_selection(lb)
        if idx is None:
            idx = 0 if direction > 0 else size - 1
        else:
            idx = max(0, min(size - 1, idx + direction))
        lb.selection_clear(0, tk.END)
        lb.selection_set(idx)
        lb.see(idx)

    def _both_selected(self) -> bool:
        return bool(self.list_a.curselection()) and bool(self.list_b.curselection())

    def _move_both_up(self, _evt=None):
        if self._both_selected():
            self._move_selection(self.list_a, -1)
            self._move_selection(self.list_b, -1)
        else:
            target = self.list_a if self._focus_target == "a" else self.list_b
            self._move_selection(target, -1)
        self._update_line_display()
        self._seek_from_a(self._get_first_selection(self.list_a))
        return "break"

    def _move_both_down(self, _evt=None):
        if self._both_selected():
            self._move_selection(self.list_a, 1)
            self._move_selection(self.list_b, 1)
        else:
            target = self.list_a if self._focus_target == "a" else self.list_b
            self._move_selection(target, 1)
        self._update_line_display()
        self._seek_from_a(self._get_first_selection(self.list_a))
        return "break"

    def _move_a_up(self, _evt=None):
        self._move_selection(self.list_a, -1)
        self._update_line_display()
        self._seek_from_a(self._get_first_selection(self.list_a))
        return "break"

    def _move_a_down(self, _evt=None):
        self._move_selection(self.list_a, 1)
        self._update_line_display()
        self._seek_from_a(self._get_first_selection(self.list_a))
        return "break"

    def _move_b_up(self, _evt=None):
        self._move_selection(self.list_b, -1)
        self._update_line_display()
        return "break"

    def _move_b_down(self, _evt=None):
        self._move_selection(self.list_b, 1)
        self._update_line_display()
        return "break"

    def _merge_b(self) -> None:
        doc = self.doc_b
        if not doc:
            return
        sel = [int(x) for x in self.list_b.curselection()]
        if len(sel) < 2:
            return
        self._push_undo()
        sel.sort()
        idx_text = doc.indices["Text"]
        base_idx = sel[0]
        texts = [doc.events[i].get(idx_text) for i in sel]
        merged = " ".join([t for t in texts if t is not None and t != ""]).strip()
        doc.events[base_idx].set(idx_text, merged)
        for i in reversed(sel[1:]):
            doc.events.pop(i)
        self._refresh_list(self.list_b, doc, is_a=False)
        self.list_b.selection_set(base_idx)
        self._update_line_display()

    def _cut_b(self) -> None:
        doc = self.doc_b
        if not doc:
            return
        idx = self._get_first_selection(self.list_b)
        if idx is None or idx < 0 or idx >= len(doc.events):
            return
        self._push_undo()
        text = self.b_text_var.get()
        try:
            cursor = int(self.b_entry.index("insert"))
        except Exception:
            cursor = len(text)
        before = text[:cursor]
        after = text[cursor:]
        doc.events[idx].set(doc.indices["Text"], before)
        new_ev = self._clone_event(doc, doc.events[idx])
        new_ev.set(doc.indices["Text"], after)
        doc.events.insert(idx + 1, new_ev)
        self._refresh_list(self.list_b, doc, is_a=False)
        self.list_b.selection_clear(0, tk.END)
        self.list_b.selection_set(idx + 1)
        self._update_line_display()

    def _delete_line(self) -> None:
        target = self._focus_target
        if target == "b":
            self._delete_in_list(self.list_b, self.doc_b, is_a=False)
        else:
            self._delete_in_list(self.list_a, self.doc_a, is_a=True)

    def _delete_in_list(self, lb: tk.Listbox, doc: Optional[AssDocument], *, is_a: bool) -> None:
        if not doc:
            return
        sel = [int(x) for x in lb.curselection()]
        if not sel:
            return
        self._push_undo()
        for i in reversed(sorted(sel)):
            if 0 <= i < len(doc.events):
                doc.events.pop(i)
        self._refresh_list(lb, doc, is_a=is_a)
        self._update_line_display()

    def _copy_a_to_b(self) -> None:
        if not self.doc_a or not self.doc_b:
            return
        src_idx = self._get_first_selection(self.list_a)
        if src_idx is None or src_idx >= len(self.doc_a.events):
            return
        dst_idx = self._get_first_selection(self.list_b)
        self._push_undo()
        insert_at = (dst_idx + 1) if dst_idx is not None else len(self.doc_b.events)
        src_ev = self.doc_a.events[src_idx]
        new_ev = self._blank_from_template(self.doc_b, self._template_b())
        new_ev.set(self.doc_b.indices["Text"], src_ev.get(self.doc_a.indices["Text"]))
        new_ev.set(self.doc_b.indices["Name"], src_ev.get(self.doc_a.indices["Name"]))
        self.doc_b.events.insert(insert_at, new_ev)
        self._refresh_list(self.list_b, self.doc_b, is_a=False)
        self.list_b.selection_set(insert_at)
        self._update_line_display()

    def _copy_b_to_a(self) -> None:
        if not self.doc_a or not self.doc_b:
            return
        src_idx = self._get_first_selection(self.list_b)
        dst_idx = self._get_first_selection(self.list_a)
        if src_idx is None or dst_idx is None:
            return
        if src_idx >= len(self.doc_b.events) or dst_idx >= len(self.doc_a.events):
            return
        self._push_undo()
        src_ev = self.doc_b.events[src_idx]
        template = self.doc_a.events[dst_idx]
        new_ev = self._clone_event(self.doc_a, template)
        new_ev.set(self.doc_a.indices["Text"], src_ev.get(self.doc_b.indices["Text"]))
        new_ev.set(self.doc_a.indices["Name"], src_ev.get(self.doc_b.indices["Name"]))
        self.doc_a.events.insert(dst_idx + 1, new_ev)
        self._refresh_list(self.list_a, self.doc_a, is_a=True)
        self.list_a.selection_set(dst_idx + 1)
        self._update_line_display()

    def _new_line(self) -> None:
        if self._focus_target == "b":
            self._new_line_in_doc(self.doc_b, self.list_b, is_a=False)
        else:
            self._new_line_in_doc(self.doc_a, self.list_a, is_a=True)

    def _new_line_in_doc(self, doc: Optional[AssDocument], lb: tk.Listbox, *, is_a: bool) -> None:
        if not doc:
            return
        idx = self._get_first_selection(lb)
        self._push_undo()
        insert_at = (idx + 1) if idx is not None else len(doc.events)
        template = doc.events[idx] if idx is not None and idx < len(doc.events) else (doc.events[0] if doc.events else None)
        if template is None:
            fields = [""] * len(doc.format_fields)
            new_ev = AssEvent(kind="Dialogue", fields=fields)
        else:
            new_ev = self._clone_event(doc, template)
        new_ev.set(doc.indices["Text"], "")
        new_ev.set(doc.indices["Name"], "")
        doc.events.insert(insert_at, new_ev)
        self._refresh_list(lb, doc, is_a=is_a)
        lb.selection_set(insert_at)
        self._update_line_display()

    def _warp_transfer(self) -> None:
        if not self.doc_a or not self.doc_b:
            return
        sel_a = [int(x) for x in self.list_a.curselection()]
        sel_b = [int(x) for x in self.list_b.curselection()]
        if not sel_a or not sel_b:
            messagebox.showerror("WarpGate", "Select lines in both files.")
            return
        if len(sel_a) != len(sel_b):
            messagebox.showerror("WarpGate", "Select the same number of lines in A and B.")
            return
        mode = self._ask_warp_mode()
        if mode is None:
            return
        self._push_undo()
        for ia, ib in zip(sorted(sel_a), sorted(sel_b)):
            if ia >= len(self.doc_a.events) or ib >= len(self.doc_b.events):
                continue
            ev_a = self.doc_a.events[ia]
            ev_b = self.doc_b.events[ib]
            if mode in ("text", "both"):
                ev_a.set(self.doc_a.indices["Text"], ev_b.get(self.doc_b.indices["Text"]))
            if mode in ("actor", "both"):
                ev_a.set(self.doc_a.indices["Name"], ev_b.get(self.doc_b.indices["Name"]))
            ev_a.warped = True
            ev_b.warped = True
        self._refresh_lists()

    def _ask_warp_mode(self) -> Optional[str]:
        win = tk.Toplevel(self)
        win.title("Warp")
        win.resizable(False, False)
        win.grab_set()
        choice = tk.StringVar(value="both")

        ttk.Label(win, text="Transfer:").pack(padx=12, pady=(12, 4), anchor="w")
        ttk.Radiobutton(win, text="Text", variable=choice, value="text").pack(anchor="w", padx=12)
        ttk.Radiobutton(win, text="Actor", variable=choice, value="actor").pack(anchor="w", padx=12)
        ttk.Radiobutton(win, text="Text + Actor", variable=choice, value="both").pack(anchor="w", padx=12)

        result: List[Optional[str]] = [None]

        def _ok():
            result[0] = choice.get()
            win.destroy()

        def _cancel():
            result[0] = None
            win.destroy()

        btns = ttk.Frame(win)
        btns.pack(pady=12)
        ttk.Button(btns, text="OK", command=_ok).pack(side="left", padx=6)
        ttk.Button(btns, text="Cancel", command=_cancel).pack(side="left", padx=6)
        win.protocol("WM_DELETE_WINDOW", _cancel)
        win.wait_window()
        return result[0]

    def _export_a(self) -> None:
        if not self.doc_a:
            return
        base = os.path.splitext(os.path.basename(self.doc_a.path))[0]
        if not base.lower().endswith("_full"):
            base = f"{base}_full"
        out_path = os.path.join(os.path.dirname(self.doc_a.path), f"{base}.ass")
        try:
            with open(out_path, "w", encoding="utf-8-sig", errors="replace") as f:
                f.write(self.doc_a.to_string())
        except Exception as exc:
            messagebox.showerror("WarpGate", f"Could not export: {exc}")
            return
        messagebox.showinfo("WarpGate", f"Exported: {out_path}")

    def _clone_event(self, doc: AssDocument, template: AssEvent) -> AssEvent:
        fields = list(template.fields)
        ev = AssEvent(kind=template.kind, fields=fields)
        start = ev.get(doc.indices["Start"])
        end = ev.get(doc.indices["End"])
        ev.start_s = ass_time_to_seconds(start)
        ev.end_s = ass_time_to_seconds(end)
        return ev

    def _blank_from_template(self, doc: AssDocument, template: Optional[AssEvent]) -> AssEvent:
        if template is None:
            fields = [""] * len(doc.format_fields)
            return AssEvent(kind="Dialogue", fields=fields)
        return self._clone_event(doc, template)

    def _template_b(self) -> Optional[AssEvent]:
        if not self.doc_b or not self.doc_b.events:
            return None
        idx = self._get_first_selection(self.list_b)
        if idx is not None and 0 <= idx < len(self.doc_b.events):
            return self.doc_b.events[idx]
        return self.doc_b.events[0]
