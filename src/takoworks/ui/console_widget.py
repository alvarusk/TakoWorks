from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText


class ConsoleFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", padx=6, pady=(6, 0))

        ttk.Label(toolbar, text="Console").pack(side="left")
        ttk.Button(toolbar, text="Clear", command=self.clear).pack(side="right")

        self.text = ScrolledText(self, height=10, wrap="word")
        self.text.pack(fill="both", expand=True, padx=6, pady=6)
        self.text.configure(state="disabled")

    def write(self, msg: str) -> None:
        self.text.configure(state="normal")
        if not msg.endswith("\n"):
            msg += "\n"
        self.text.insert(tk.END, msg)
        self.text.see(tk.END)
        self.text.configure(state="disabled")
        self.text.update_idletasks()

    def clear(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", tk.END)
        self.text.configure(state="disabled")
