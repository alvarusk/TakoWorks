
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Actualiza TakoWorks Corrector v0.473 a v0.474."""

from __future__ import annotations

import ast
import sys
from datetime import datetime
from pathlib import Path


NEW_EXTRACT = r'''
def _is_repetitive_cartel_text(text: str) -> bool:
    words = re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)
    if len(words) < 8:
        return False
    most_common = max(words.count(w) for w in set(words))
    return most_common >= 8 and most_common / len(words) >= 0.75


def extract_dialogue_lines(events: List[Event]) -> Tuple[List[str], List[bool]]:
    out_lines: List[str] = []
    flags: List[bool] = []
    skipped = 0

    for ev in events:
        if ev.kind != "Dialogue":
            continue

        cleaned = clean_ass_text(ev.text)
        if not cleaned:
            continue

        if ev.is_cartel and _is_repetitive_cartel_text(cleaned):
            skipped += 1
            continue

        out_lines.append(cleaned)
        flags.append(bool(ev.is_cartel))

    if skipped:
        print(f"[i] CARTEL repetitivos omitidos de LT: {skipped}")
        print("[i] Los eventos originales no se modifican.")

    return out_lines, flags
'''


OLD_HTTP = '''    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)'''

NEW_HTTP = '''    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"LanguageTool HTTP {exc.code}: {detail[:650]}"
        ) from exc
    return json.loads(raw)'''


RECOVERY = r'''
def lt_check_resilient(
    chunk_lines: List[str],
    settings: Settings,
    skipped: List[str],
) -> List[Tuple[List[str], Dict[str, Any]]]:
    if not chunk_lines:
        return []

    try:
        response = lt_check("\n".join(chunk_lines), settings)
        return [(chunk_lines, response)]
    except Exception as exc:
        message = str(exc).casefold()

        error_rate = (
            "too many errors" in message
            or "error rate" in message
            or "demasiados errores" in message
        )
        size_error = (
            "http 413" in message
            or "text exceeds" in message
            or "too long" in message
        )

        if not (error_rate or size_error):
            raise

        if len(chunk_lines) == 1:
            if size_error:
                raise

            skipped.append(chunk_lines[0])
            print(
                "[WARN] LanguageTool rechaza una línea por "
                "exceso de errores. Se omite únicamente esa línea."
            )
            print(f"       {chunk_lines[0][:120]}")
            return []

        midpoint = len(chunk_lines) // 2
        print(
            "[i] LanguageTool rechazó un bloque. "
            "Dividiendo para aislar el problema..."
        )

        left = lt_check_resilient(
            chunk_lines[:midpoint], settings, skipped
        )
        right = lt_check_resilient(
            chunk_lines[midpoint:], settings, skipped
        )
        return left + right

'''


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"Se esperaban 1 coincidencia y aparecieron {count}: "
            f"{old[:90]!r}"
        )
    return source.replace(old, new, 1)


def replace_function(source: str, name: str, new_code: str) -> str:
    tree = ast.parse(source)
    found = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(found) != 1:
        raise RuntimeError(f"No se encontró la función {name}")

    node = found[0]
    lines = source.splitlines(keepends=True)
    replacement = new_code.strip("\n") + "\n"
    return (
        "".join(lines[:node.lineno - 1])
        + replacement
        + "".join(lines[node.end_lineno:])
    )


def patch(source: str) -> str:
    if "def _is_repetitive_cartel_text(" in source:
        raise RuntimeError("La corrección ya está instalada.")

    if "v0.473 (repack)" not in source:
        raise RuntimeError(
            "Versión de corrector.py no reconocida. "
            "No se modifica el archivo."
        )

    source = replace_function(
        source, "extract_dialogue_lines", NEW_EXTRACT
    )

    begin = source.index("def lt_check(")
    end = source.index("\ndef lt_add_word(", begin)
    block = source[begin:end]
    block = replace_once(block, OLD_HTTP, NEW_HTTP)
    source = source[:begin] + block + source[end:]

    source = replace_once(
        source,
        "\ndef lt_add_word(",
        "\n" + RECOVERY.strip("\n") + "\n\n\ndef lt_add_word(",
    )

    source = replace_once(
        source,
        'ap.add_argument("--max-chars", type=int, default=45000,',
        'ap.add_argument("--max-chars", type=int, default=4000,',
    )

    source = replace_once(
        source,
        '    lines, flags = extract_dialogue_lines(events_reordered)\n',
        '''    lines, flags = extract_dialogue_lines(events_reordered)
    skipped_repetitions = sum(
        1 for ev in events_reordered
        if ev.kind == "Dialogue" and ev.is_cartel
        and _is_repetitive_cartel_text(clean_ass_text(ev.text))
    )
''',
    )

    source = replace_once(
        source,
        '    chunks = split_into_chunks(merged, max_chars=args.max_chars)\n',
        '''    chunks = split_into_chunks(merged, max_chars=args.max_chars)
    skipped_lt: List[str] = []
''',
    )

    source = replace_once(
        source,
        '''        chunk_text = "\\n".join(chunk_lines)
        try:
            resp = lt_check(chunk_text, settings)
''',
        '''        try:
            results = lt_check_resilient(
                chunk_lines, settings, skipped_lt
            )
''',
    )

    source = replace_once(
        source,
        '''        raw_responses.append(resp)
        rows = map_matches_to_rows(chunk_lines, resp)
        if not personal_word_set and local_word_set:
            personal_word_set = set(local_word_set)
        if personal_word_set:
            rows = filter_rows_by_personal_dict(rows, personal_word_set)
        rows_all.extend(rows)
        print(f"    - Chunk {ci}/{len(chunks)}: {len(rows)} errores")
''',
        '''        chunk_errors = 0
        for checked_lines, resp in results:
            raw_responses.append(resp)
            rows = map_matches_to_rows(checked_lines, resp)
            if not personal_word_set and local_word_set:
                personal_word_set = set(local_word_set)
            if personal_word_set:
                rows = filter_rows_by_personal_dict(
                    rows, personal_word_set
                )
            rows_all.extend(rows)
            chunk_errors += len(rows)

        print(
            f"    - Chunk {ci}/{len(chunks)}: "
            f"{chunk_errors} errores"
        )
''',
    )

    source = replace_once(
        source,
        '        "Líneas (merged)": str(len(merged)),\n',
        '''        "CARTEL repetitivos omitidos de LT": str(skipped_repetitions),
        "Fragmentos rechazados por LT": str(len(skipped_lt)),
        "Líneas (merged)": str(len(merged)),
''',
    )

    source = replace_once(
        source,
        "v0.473 (repack)",
        "v0.474 (CARTEL/LT fix)",
    )

    ast.parse(source)
    compile(source, "corrector.py", "exec")
    return source


def main() -> int:
    if len(sys.argv) > 1:
        target = Path(sys.argv[1]).resolve()
    else:
        target = Path(__file__).resolve().with_name("corrector.py")

    if not target.is_file():
        print(f"[ERROR] No se encuentra: {target}")
        return 2

    try:
        original = target.read_text(encoding="utf-8-sig")
        corrected = patch(original)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 3

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = target.with_name(target.name + ".bak_" + stamp)

    try:
        backup.write_bytes(target.read_bytes())
        temp = target.with_name(target.name + ".tmp")
        temp.write_text(corrected, encoding="utf-8")
        temp.replace(target)
    except Exception as exc:
        print(f"[ERROR] No se pudo instalar: {exc}")
        return 4

    print("[OK] Corrector actualizado a v0.474")
    print(f"[OK] Archivo: {target}")
    print(f"[OK] Copia de seguridad: {backup}")
    print("[i] La DLL y los CMD permanecen sin cambios.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
