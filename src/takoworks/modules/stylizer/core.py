import os
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import scrolledtext


# ============================================================
#  Remove linebreaks (\\N) en diálogos
# ============================================================

import re

_CJK_RE = re.compile(r"[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")


def _remove_linebreaks(text: str) -> str:
    """Elimina \\N/\n en diálogos.

    Conserva el salto si es diálogo de 2 personajes: '- ...\\N- ...'.
    """
    t = text or ""
    # 2 personajes: empieza con '-' y hay otro '-' justo tras \\N/\n (ignorando tags/espacios)
    if re.search(r"^\s*-.*(?:\\N|\\n)\s*-", t):
        return t

    # Reemplazo inteligente: en CJK no metemos espacio; en texto latino sí
    def repl(m):
        left = m.group(1)
        right = m.group(2)
        if _CJK_RE.search(left) or _CJK_RE.search(right):
            return left + right
        return left + " " + right

    # Caso más común: ...X\\NY...
    t = re.sub(r"(.)(?:\\N|\\n)(.)", repl, t)
    # Cualquier \\N restante
    t = t.replace(r"\\N", "").replace(r"\\n", "")
    # Colapsar espacios
    t = re.sub(r"\s{2,}", " ", t)
    return t


STYLES_BLOCK = """[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Gen_Main,Trebuchet MS,24,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,18,1
Style: Gen_Italics,Trebuchet MS,24,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,-1,0,0,100,100,0,0,1,2,1,2,10,10,18,1
Style: Gen_Main_Up,Trebuchet MS,24,&H00FFFFFF,&H000000FF,&H00000000,&H00090909,0,0,0,0,100,100,0,0,1,2,1,8,10,10,18,1
Style: Gen_Italics_top,Trebuchet MS,24,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,-1,0,0,100,100,0,0,1,2,1,8,10,10,18,1
Style: Gen_Nota,Trebuchet MS,20,&H00FFFFFF,&H000000FF,&H00262626,&H00505050,0,-1,0,0,100,100,0,0,1,2,1,8,10,10,18,1
Style: Cart_A_Tre,Trebuchet MS,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,8,20,20,20,1
Style: Cart_B_Tre,Trebuchet MS,24,&H00323232,&H000000FF,&H00D9D9D9,&H00D9D9D9,-1,0,0,0,100,100,0,0,1,2,1,8,20,20,20,1
Style: Cart_C_Tre,Trebuchet MS,20,&H00000000,&H000000FF,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,100,100,0,0,1,2,0,8,20,20,20,1
Style: Cart_A_Tim,Times New Roman,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,8,20,20,20,1
Style: Cart_B_Tim,Times New Roman,24,&H00323232,&H000000FF,&H00D9D9D9,&H00D9D9D9,-1,0,0,0,100,100,0,0,1,2,1,8,20,20,20,1
Style: Cart_C_Tim,Times New Roman,20,&H00000000,&H000000FF,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,100,100,0,0,1,2,0,8,20,20,20,1
Style: Edit_Margin,Trebuchet MS,24,&H00FFFFFF,&H000000FF,&H0063110A,&H00000000,0,0,0,0,100,100,0,0,1,2,0,8,10,10,10,1
Style: Cart_A_Verd,Verdana,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,110,0,0,1,2,0,8,20,20,20,1
Style: Cart_C_Verd,Verdana,20,&H00000000,&H000000FF,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,100,100,0,0,1,2,0,8,20,20,20,1
Style: Cart_A_Tre - Block,Trebuchet MS,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,3,2,0,8,20,20,20,1
Style: Cart_A_Verd_Names,Verdana,20,&H00DBCB59,&H000000FF,&H00F7F5EE,&H00000000,-1,-1,0,0,100,110,0,0,1,1.5,0,2,20,20,20,1
"""

ALLOWED_STYLES = {
    "Gen_Main",
    "Gen_Italics",
    "Gen_Main_Up",
    "Gen_Italics_top",
    "Gen_Nota",
    "Cart_A_Tre",
    "Cart_B_Tre",
    "Cart_C_Tre",
    "Cart_A_Tim",
    "Cart_B_Tim",
    "Cart_C_Tim",
    "Edit_Margin",
    "Cart_A_Verd",
    "Cart_C_Verd",
    "Cart_A_Tre - Block",
    "Cart_A_Verd_Names",
}

def log_console(console, message):
    if console is None:
        print(message)
        return
    console.configure(state="normal")
    console.insert(tk.END, message + "\n")
    console.see(tk.END)
    console.configure(state="disabled")
    console.update_idletasks()

def replace_styles_section(lines, console=None):
    if not lines:
        return lines
    styles_replaced = False
    new_lines = []
    i = 0
    n = len(lines)

    styles_header_lower = "[v4+ styles]"

    # Pre-split STYLES_BLOCK en líneas con salto
    styles_block_lines = [l + "\n" for l in STYLES_BLOCK.strip().splitlines()]

    while i < n:
        line = lines[i]
        stripped = line.strip()
        if stripped.lower() == styles_header_lower:
            log_console(console, "-> Encontrada sección [V4+ Styles]. Reemplazando estilos...")
            # Insertar bloque completo
            new_lines.extend(styles_block_lines)
            styles_replaced = True
            i += 1
            # Saltar hasta la siguiente sección (línea que empieza por '[')
            while i < n and not lines[i].lstrip().startswith("["):
                i += 1
            continue
        new_lines.append(line)
        i += 1

    if not styles_replaced:
        log_console(console, "-> No se encontró sección [V4+ Styles]. Insertando sección nueva al inicio de [Events]...")
        # Intentar insertar antes de [Events]
        inserted = False
        for idx, line in enumerate(new_lines):
            if line.strip().lower() == "[events]":
                new_lines = new_lines[:idx] + styles_block_lines + ["\n"] + new_lines[idx:]
                inserted = True
                break
        if not inserted:
            log_console(console, "-> Tampoco se encontró sección [Events]. Añadiendo sección de estilos al final del archivo.")
            new_lines.extend(["\n"] + styles_block_lines)

    return new_lines

def process_events(
    lines,
    clean_carteles=False,
    clean_comments=False,
    transform_styles=False,
    clean_text=False,
    remove_linebreaks=False,
    console=None,
):
    new_lines = []
    in_events = False
    format_fields = None
    style_idx = None
    name_idx = None
    text_idx = None
    unknown_styles = set()

    for line in lines:
        stripped = line.lstrip()
        # Limpiar comentarios crudos
        if clean_comments and stripped.startswith("Comment:"):
            continue

        if stripped.startswith("["):
            section = stripped.strip()
            in_events = section.lower() == "[events]"
            format_fields = None
            style_idx = None
            name_idx = None
            text_idx = None
            new_lines.append(line)
            continue

        if in_events and stripped.startswith("Format:"):
            # Parse Format
            fmt = stripped[len("Format:"):].strip()
            fields = [f.strip() for f in fmt.split(",")]
            format_fields = fields
            style_idx = fields.index("Style") if "Style" in fields else None
            name_idx = fields.index("Name") if "Name" in fields else None
            text_idx = fields.index("Text") if "Text" in fields else None
            if style_idx is None or name_idx is None or text_idx is None:
                log_console(
                    console,
                    "ADVERTENCIA: No se encontraron columnas 'Style', 'Name' y/o 'Text' en la línea Format de [Events]."
                )
            new_lines.append(line)
            continue

        if in_events and (stripped.startswith("Dialogue:") or stripped.startswith("Comment:")) and format_fields and style_idx is not None:
            # Procesar eventos
            leading = line[: len(line) - len(stripped)]
            line_ending = "\n" if line.endswith("\n") else ""
            prefix, rest = stripped.split(":", 1)
            rest = rest.lstrip()
            # Split solo en las primeras N-1 comas
            parts = rest.split(",", maxsplit=len(format_fields) - 1)
            # Asegurar longitud
            if len(parts) < len(format_fields):
                parts += [""] * (len(format_fields) - len(parts))

            style_val = parts[style_idx].strip()
            name_val = parts[name_idx].strip() if name_idx is not None else ""

            # 1) Transformar estilos (primero)
            if transform_styles and prefix.lower().startswith("dialogue"):
                new_style = style_val

                s_low = style_val.lower()
                n_low = name_val.lower()

                # Orden importante: primero "top italics"
                if "top italics" in s_low:
                    new_style = "Gen_Italics_top"
                elif "top" in s_low:
                    new_style = "Gen_Main_Up"
                elif "italics" in s_low:
                    new_style = "Gen_Italics"
                elif "default" in s_low:
                    new_style = "Gen_Main"
                elif "flashback" in s_low:
                    new_style = "Gen_Main"

                # Reglas por Name (prioridad para carteles)
                if "text" in n_low or "sign" in n_low:
                    new_style = "Cart_A_Tre"
                # Narrador
                if "narr" in n_low or "narrator" in n_low:
                    new_style = "Gen_Italics"

                parts[style_idx] = new_style
                style_val = new_style  # para chequeos posteriores

                if style_val not in ALLOWED_STYLES:
                    unknown_styles.add(style_val)

            # 2) Limpiar texto (vaciar columna Text)
            if clean_text and text_idx is not None:
                parts[text_idx] = ""

            # 2.5) Remove linebreaks (\\N) en diálogos
            if remove_linebreaks and prefix.lower().startswith("dialogue") and text_idx is not None:
                parts[text_idx] = _remove_linebreaks(parts[text_idx])

            # 3) Limpiar carteles (DESPUÉS de transformar estilos)
            if clean_carteles and prefix.lower().startswith("dialogue"):
                style_lower = style_val.lower()
                name_lower = name_val.lower()
                if ("cart" in style_lower) or ("cartel" in style_lower) or ("cart" in name_lower) or ("cartel" in name_lower):
                    # Línea descartada
                    continue

            # Reconstruir línea
            new_rest = ",".join(parts)
            new_line = f"{leading}{prefix}: {new_rest}{line_ending}"
            new_lines.append(new_line)
            continue

        # Cualquier otra línea
        new_lines.append(line)

    # Informar estilos desconocidos
    if transform_styles and unknown_styles:
        log_console(console, "Estilos encontrados que NO están en la lista del punto 5:")
        for st in sorted(unknown_styles):
            log_console(console, f"  - {st}")

    return new_lines

def read_file(path):
    # Intentar UTF-8 con BOM, luego cp1252
    for enc in ("utf-8-sig", "cp1252"):
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.readlines()
        except UnicodeError:
            continue
        except Exception:
            break
    # Último intento, reemplazando caracteres problemáticos
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        return f.readlines()

def write_file(path, lines):
    with open(path, "w", encoding="utf-8-sig", errors="replace") as f:
        f.writelines(lines)

class ASSGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Limpiador .ASS")
        self.geometry("800x600")

        self.input_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()

        self.var_clean_carteles = tk.BooleanVar(value=False)
        self.var_clean_comments = tk.BooleanVar(value=False)
        self.var_add_styles = tk.BooleanVar(value=False)
        self.var_transform_styles = tk.BooleanVar(value=False)
        self.var_clean_text = tk.BooleanVar(value=False)

        self._build_widgets()

    def _build_widgets(self):
        # Frame de entrada
        frame_top = tk.Frame(self)
        frame_top.pack(fill="x", padx=10, pady=10)

        tk.Label(frame_top, text="Archivo .ASS (input):").grid(row=0, column=0, sticky="w")
        entry_input = tk.Entry(frame_top, textvariable=self.input_path_var, width=60)
        entry_input.grid(row=0, column=1, padx=5)
        tk.Button(frame_top, text="Examinar...", command=self.browse_input).grid(row=0, column=2, padx=5)

        tk.Label(frame_top, text="Carpeta de salida:").grid(row=1, column=0, sticky="w", pady=(5, 0))
        entry_output = tk.Entry(frame_top, textvariable=self.output_dir_var, width=60)
        entry_output.grid(row=1, column=1, padx=5, pady=(5, 0))
        tk.Button(frame_top, text="Examinar...", command=self.browse_output).grid(row=1, column=2, padx=5, pady=(5, 0))

        # Frame de opciones
        frame_opts = tk.LabelFrame(self, text="Opciones")
        frame_opts.pack(fill="x", padx=10, pady=10)

        tk.Checkbutton(frame_opts, text="Limpiar carteles", variable=self.var_clean_carteles).grid(row=0, column=0, sticky="w", padx=5, pady=2)
        tk.Checkbutton(frame_opts, text="Limpiar comentarios", variable=self.var_clean_comments).grid(row=1, column=0, sticky="w", padx=5, pady=2)
        tk.Checkbutton(frame_opts, text="Añadir estilos (reemplaza [V4+ Styles])", variable=self.var_add_styles).grid(row=0, column=1, sticky="w", padx=5, pady=2)
        tk.Checkbutton(frame_opts, text="Transformar estilos (diálogos)", variable=self.var_transform_styles).grid(row=1, column=1, sticky="w", padx=5, pady=2)
        tk.Checkbutton(frame_opts, text="Limpiar texto (vaciar columna Text)", variable=self.var_clean_text).grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=2)

        # Botones de acción
        frame_actions = tk.Frame(self)
        frame_actions.pack(fill="x", padx=10, pady=(0, 10))

        tk.Button(frame_actions, text="Procesar", command=self.process).pack(side="left")
        tk.Button(frame_actions, text="Salir", command=self.destroy).pack(side="right")

        # Consola
        frame_console = tk.LabelFrame(self, text="Console")
        frame_console.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.console = scrolledtext.ScrolledText(frame_console, state="disabled", wrap="word")
        self.console.pack(fill="both", expand=True)

    def browse_input(self):
        path = filedialog.askopenfilename(
            title="Seleccionar archivo .ass",
            filetypes=[("Archivos ASS", "*.ass"), ("Todos los archivos", "*.*")]
        )
        if path:
            self.input_path_var.set(path)
            # Por defecto, la carpeta de salida es la del input
            if not self.output_dir_var.get():
                self.output_dir_var.set(os.path.dirname(path))

    def browse_output(self):
        folder = filedialog.askdirectory(title="Seleccionar carpeta de salida")
        if folder:
            self.output_dir_var.set(folder)

    def process(self):
        input_path = self.input_path_var.get().strip()
        output_dir = self.output_dir_var.get().strip()

        if not input_path:
            messagebox.showerror("Error", "Debe seleccionar un archivo .ASS de entrada.")
            return

        if not os.path.isfile(input_path):
            messagebox.showerror("Error", "El archivo de entrada no existe.")
            return

        if not output_dir:
            output_dir = os.path.dirname(input_path)
            self.output_dir_var.set(output_dir)

        if not os.path.isdir(output_dir):
            messagebox.showerror("Error", "La carpeta de salida no es válida.")
            return

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_OUT.ass")

        # Limpiar consola
        self.console.configure(state="normal")
        self.console.delete("1.0", tk.END)
        self.console.configure(state="disabled")

        log_console(self.console, f"Procesando: {input_path}")
        log_console(self.console, f"Salida: {output_path}")

        try:
            lines = read_file(input_path)
            log_console(self.console, f"Líneas leídas: {len(lines)}")

            # Añadir estilos
            if self.var_add_styles.get():
                lines = replace_styles_section(lines, console=self.console)

            # Procesar eventos
            lines = process_events(
                lines,
                clean_carteles=self.var_clean_carteles.get(),
                clean_comments=self.var_clean_comments.get(),
                transform_styles=self.var_transform_styles.get(),
                clean_text=self.var_clean_text.get(),
                console=self.console,
            )

            write_file(output_path, lines)
            log_console(self.console, "Proceso completado.")
            messagebox.showinfo("Listo", f"Archivo procesado y guardado como:\n{output_path}")
        except Exception as e:
            log_console(self.console, "ERROR durante el procesado:")
            log_console(self.console, str(e))
            log_console(self.console, traceback.format_exc())
            messagebox.showerror("Error", f"Ocurrió un error:\n{e}")

def main():
    app = ASSGui()
    app.mainloop()

if __name__ == "__main__":
    main()
