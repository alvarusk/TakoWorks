from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass
from typing import Callable, Optional, Any


LogFn = Callable[[str], None]
TaskFn = Callable[[threading.Event, LogFn], Any]
DoneFn = Callable[[bool, Optional[str]], None]  # (ok, error_text)


@dataclass
class TaskHandle:
    cancel_event: threading.Event


class TaskRunner:
    """
    Ejecuta tareas en background y bombea logs a Tk sin congelar la UI.
    """
    def __init__(self, tk_root, console_write: LogFn):
        self._root = tk_root
        self._console_write = console_write
        self._q: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._active = False
        self._cancel_event = threading.Event()
        self._done_cb: Optional[DoneFn] = None

        self._root.after(60, self._pump)

    def is_busy(self) -> bool:
        return self._active

    def cancel(self) -> None:
        self._cancel_event.set()
        self._console_write("[!] Cancel solicitado…")

    def start(self, title: str, fn: TaskFn, on_done: Optional[DoneFn] = None) -> TaskHandle:
        if self._active:
            self._console_write("[!] Ya hay una tarea en ejecución.")
            return TaskHandle(self._cancel_event)

        self._active = True
        self._cancel_event = threading.Event()
        self._done_cb = on_done

        self._console_write(f"\n=== {title} ===")

        def log(msg: str) -> None:
            self._q.put(("log", msg))

        def run():
            try:
                fn(self._cancel_event, log)
                self._q.put(("done", ""))
            except Exception:
                self._q.put(("error", traceback.format_exc()))

        threading.Thread(target=run, daemon=True).start()
        return TaskHandle(self._cancel_event)

    def _pump(self):
        try:
            while True:
                typ, payload = self._q.get_nowait()
                if typ == "log":
                    if payload:
                        self._console_write(payload)
                elif typ == "done":
                    self._active = False
                    if self._done_cb:
                        self._done_cb(True, None)
                elif typ == "error":
                    self._active = False
                    self._console_write("ERROR:\n" + payload)
                    if self._done_cb:
                        self._done_cb(False, payload)
        except queue.Empty:
            pass
        finally:
            self._root.after(60, self._pump)
