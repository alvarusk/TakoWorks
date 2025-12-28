import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

try:
    from docx import Document  # para .docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False



import re

_CJK_RE = re.compile(r"[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")

def _split_cjk_spaces(text: str):
    """Si el texto contiene CJK y también espacios, lo parte en varias líneas.

    Esto evita destrozar idiomas con espacios (ES/EN), pero permite el caso típico
    de scripts JP/CJK donde un espacio significa 'siguiente línea' dentro del Word.
    """
    t = (text or "").strip()
    if not t:
        return []
    if "\\N" in t or "\\n" in t:
        return [t]
    if not _CJK_RE.search(t):
        return [t]
    if (" " not in t) and ("　" not in t):  # espacio normal o fullwidth
        return [t]
    parts = [p.strip() for p in re.split(r"[ \t　]+", t) if p.strip()]
    return parts or [t]

def log(msg, console):
    console.insert(tk.END, msg + "\n")
    console.see(tk.END)
    console.update_idletasks()


def parse_line(raw: str):
    """
    Recibe una línea tipo:
    - 卓异：这怎么可能
    - 卓异: 这怎么可能
    Devuelve (actor, texto).
    Si no encuentra dos puntos, devuelve ("", línea).
    """
    line = raw.strip()
    if not line:
        return None, None  # línea vacía

    # Priorizar el dos puntos chino
    if "：" in line:
        actor_part, text_part = line.split("：", 1)
    elif ":" in line:
        actor_part, text_part = line.split(":", 1)
    else:
        # Sin separador claro: lo tratamos todo como texto
        return "", line

    actor = actor_part.strip()
    text = text_part.strip()

    # Limpiar posibles corchetes o paréntesis alrededor del actor
    for ch in "[]（）()":
        actor = actor.replace(ch, "")
    actor = actor.strip()

    return actor, text


def read_lines_from_doc(path: str, console):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".docx":
        if not HAS_DOCX:
            raise RuntimeError(
                "Falta la librería 'python-docx'. Instálala con: pip install python-docx"
            )
        log("Leyendo .docx…", console)
        doc = Document(path)
        raw_lines = []
        for p in doc.paragraphs:
            if p.text:
                # Puede haber saltos de línea dentro del mismo párrafo
                for sub_line in p.text.splitlines():
                    raw_lines.append(sub_line)
    else:
        log("Leyendo archivo de texto…", console)
        with open(path, "r", encoding="utf-8") as f:
            raw_lines = f.read().splitlines()

    # Parsear cada línea
    parsed = []
    for raw in raw_lines:
        actor, text = parse_line(raw)
        if actor is None and text is None:
            continue  # vacía
        if text == "":
            continue  # sin texto, la saltamos
        # Reemplazar saltos de línea internos por \N de ASS
        text = text.replace("\n", r"\N")
        for chunk in _split_cjk_spaces(text):
            parsed.append((actor, chunk))

    return parsed


def build_ass_content(lines):
    """
    lines: lista de tuplas (actor, text)
    Devuelve el contenido completo del archivo ASS como cadena.
    """

    # Cabecera ASS básica
    script_info = (
        "[Script Info]\n"
        "; Generado automáticamente desde Word\n"
        "ScriptType: v4.00+\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: TV.601\n"
        "\n"
    )

    styles = (
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial,40,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        "-1,0,0,0,100,100,0,0,1,3,0,2,10,10,10,1\n"
        "\n"
    )

    events_header = (
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    # Todos los diálogos con el MISMO tiempo
    start = "0:00:00.00"
    end = "0:00:01.00"

    dialogue_lines = []
    for actor, text in lines:
        dialogue = (
            f"Dialogue: 0,{start},{end},Default,{actor},0000,0000,0000,,{text}"
        )
        dialogue_lines.append(dialogue)

    content = script_info + styles + events_header + "\n".join(dialogue_lines) + "\n"
    return content


def convert_word_to_ass(word_path: str, console):
    if not word_path:
        messagebox.showerror("Error", "Selecciona primero un archivo de entrada.")
        return

    if not os.path.isfile(word_path):
        messagebox.showerror("Error", "La ruta del archivo no es válida.")
        return

    base, _ = os.path.splitext(word_path)
    ass_path = base + ".ass"

    try:
        log(f"Archivo de entrada: {word_path}", console)
        lines = read_lines_from_doc(word_path, console)

        if not lines:
            messagebox.showwarning(
                "Aviso",
                "No se han encontrado líneas válidas (Actor: Texto)."
            )
            return

        log(f"Líneas encontradas: {len(lines)}", console)

        ass_content = build_ass_content(lines)

        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(ass_content)

        log(f"ASS generado: {ass_path}", console)
        messagebox.showinfo("Listo", f"ASS generado:\n{ass_path}")

    except Exception as e:
        log(f"ERROR: {e}", console)
        messagebox.showerror("Error", str(e))


def select_word_file(entry_var, console):
    file_path = filedialog.askopenfilename(
        title="Seleccionar archivo Word o TXT",
        filetypes=(
            ("Archivos de Word", "*.docx"),
            ("Archivos de texto", "*.txt"),
            ("Todos los archivos", "*.*"),
        ),
    )
    if file_path:
        entry_var.set(file_path)
        log(f"Seleccionado: {file_path}", console)


def main():
    root = tk.Tk()
    root.title("Word → ASS (Actor → Name)")

    # Marco principal
    frame = tk.Frame(root, padx=10, pady=10)
    frame.pack(fill=tk.BOTH, expand=True)

    # Selector de archivo Word
    word_var = tk.StringVar()

    lbl_word = tk.Label(frame, text="Archivo Word / TXT:")
    lbl_word.grid(row=0, column=0, sticky="w")

    entry_word = tk.Entry(frame, textvariable=word_var, width=60)
    entry_word.grid(row=1, column=0, padx=(0, 5), pady=5, sticky="we")

    btn_word = tk.Button(
        frame,
        text="Buscar…",
        command=lambda: select_word_file(word_var, console),
    )
    btn_word.grid(row=1, column=1, pady=5, sticky="e")

    # Botón convertir
    btn_convert = tk.Button(
        frame,
        text="Convertir a ASS",
        command=lambda: convert_word_to_ass(word_var.get(), console),
    )
    btn_convert.grid(row=2, column=0, columnspan=2, pady=(5, 10))

    # Consola de mensajes
    lbl_console = tk.Label(frame, text="Console:")
    lbl_console.grid(row=3, column=0, sticky="w")

    console = scrolledtext.ScrolledText(frame, width=80, height=15)
    console.grid(row=4, column=0, columnspan=2, pady=(0, 5), sticky="nsew")

    # Que el frame crezca con la ventana
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(4, weight=1)

    log("Listo. Selecciona un Word o TXT y pulsa 'Convertir a ASS'.", console)

    root.mainloop()


if __name__ == "__main__":
    main()
