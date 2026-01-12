# TakoWorks (v1.39.0)

Toolkit para transcribir, romanizar y traducir guiones/ASS de japonés y chino. Genera ASS por modelo, HTML resumen y registra costes (opcional Supabase).

## Requisitos
- Python 3.x
- FFmpeg en `PATH`
- Opcional: diccionarios Yomitan (`jpdict_dir` / `cndict_dir`), `zhpr` + `transformers` para puntuación zh.

## Configuración
- Archivo principal: `config.json` (en la raíz del repo o junto al exe).
- Secretos/no versionado: `%APPDATA%/TakoWorks/config.local.json` (Windows) o `~/.config/TakoWorks/config.local.json`:
```json
{
  "api_keys": {
    "openai": "sk-...",
    "anthropic": "sk-ant-...",
    "gemini": "AIza...",
    "deepseek": "sk-..."
  },
  "supabase": {
    "url": "https://xxxxx.supabase.co",
    "service_key": "sbp_..._service_role",
    "anon_key": "sbp_..._anon_optional"
  }
}
```
- Variables de entorno (tienen prioridad): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`/`SUPABASE_SERVICE_ROLE_KEY`/`SUPABASE_ANON_KEY`, `SUPABASE_COST_TABLE` (por defecto `voicex_api_costs`).
- Rutas opcionales en `config.json`: `jpdict_dir`, `cndict_dir`, `ffmpeg_dir`, `yomitoku_dir`, costes por 1K tokens en `cost_per_1k`.

## Versionado y releases
- Versión fuente: `src/takoworks/__init__.py` (el encabezado de este README se actualiza con el bump).
- Script de bump: `python bin/bump_version.py --mode auto|minor|patch` (auto: `minor` en main, `patch` en ramas).
- Releases: publica tag/release en GitHub con la versión (ej. v1.7.1) para que Actions genere `TakoWorks_win64.zip`.
- MS Store: usa versión semver sin sufijos; al empaquetar MSIX fija la versión del manifest (ej. 1.7.1.0) alineada con `__version__`.

## Uso rápido
- Transcribir + traducir:
```
python -m takoworks.modules.transcriber.core input.ass input.mp4 --lang ja|zh \
  --models gpt,claude,gemini,deepseek [--do-roman-morph] [--html]
```
- Añadir solo romaji/pinyin + diccionario a un ASS existente:
```
python -m takoworks.modules.transcriber.core input.ass --skip-asr --do-roman-morph
```
- Flags útiles: `--out-dir`, `--base-name`, `--pad-ms`, `--source-type`, `--series`.

## Salidas
- ASS por modelo: línea original, `{romaji/pinyin}`, `{glosas}` y traducción (no oculta líneas de origen en traducciones).
- HTML opcional con columnas: original, romaji/pinyin, glosas, GPT, Claude, Gemini, DeepSeek.

## Modelos y manejo de errores
- Modelos soportados: GPT-5 (OpenAI), Claude, Gemini 2.5 Flash, DeepSeek.
- Si falta clave o falla la inicialización de un modelo, se omite y el pipeline sigue con los demás; se informa en consola y no se genera su archivo de salida.
- Costes: se calculan si el SDK devuelve uso; si Supabase está configurado, se guardan los costes en la tabla indicada.

## Supabase (costes)
- Define `SUPABASE_URL` + `SUPABASE_SERVICE_KEY` (o en `config.local.json`) para registrar costes.
- Tabla por defecto: `voicex_api_costs` (configurable con `SUPABASE_COST_TABLE`).

## Tests
- Requiere `pytest`. Ejecuta: `pytest -q`
- Cobertura actual: helpers ASS y `parse_json_translations`.

## Notas
- El formato de líneas con romanización/glosas usa llaves `{}` y sanitización para no romper ASS (helpers en `ass_utils.py`).
- La puntuación libre en zh requiere `zhpr`; si falta, el pipeline continúa con texto original.
