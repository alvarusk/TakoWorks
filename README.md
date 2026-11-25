# TakoWorks (v1.118.0)

Toolkit for transcribing, romanizing, translating, and reviewing ASS subtitle files for Japanese and Chinese. Generates per-model ASS files, summary HTML, and optionally logs costs to Supabase.

## Requirements
- Python 3.x
- FFmpeg in `PATH`
- Optional: `zhpr` + `transformers` for Chinese punctuation restoration

## Configuration
- Main config file: `config.json` (in the repo root or next to the executable)
- Secrets / non-versioned config: `%APPDATA%/TakoWorks/config.local.json` on Windows or `~/.config/TakoWorks/config.local.json`
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
- Environment variables take priority: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` / `SUPABASE_SERVICE_ROLE_KEY` / `SUPABASE_ANON_KEY`, `SUPABASE_COST_TABLE` (default: `voicex_api_costs`)
- Optional `config.json` paths: `ffmpeg_dir`, `yomitoku_dir`, and per-1K token costs in `cost_per_1k`

## Versioning and Releases
- Source version: `src/takoworks/__init__.py` (the README header is updated when bumping)
- Version bump script: `python bin/bump_version.py --mode auto|minor|patch` (`auto`: `minor` on main, `patch` on branches)
- Releases: publish a GitHub tag/release with the version number, e.g. `v1.7.1`, so Actions can generate `TakoWorks_win64.zip`
- MS Store: use a semver version without suffixes; when packaging MSIX, align the manifest version, e.g. `1.7.1.0`, with `__version__`

## Quick Start
- Transcribe + translate:
```bash
python -m takoworks.modules.transcriber.core input.ass input.mp4 --lang ja|zh \
  --models gpt,claude,gemini,deepseek [--do-roman-morph] [--html]
```
- Add only romaji/pinyin via DeepSeek to an existing ASS file:
```bash
python -m takoworks.modules.transcriber.core input.ass --skip-asr --do-roman-morph
```
- Useful transcriber flags: `--out-dir`, `--base-name`, `--pad-ms`, `--source-type`, `--series`

### Timed ASS transcription and episode PDF verification

In Transcriber, uncheck **Ignore transcription**, select the episode video/audio,
choose `purfview`, and browse to `faster-whisper-xxl.exe` if needed (standard
Subtitle Edit installations are detected automatically). This uses Purfview Faster Whisper XXL with
`large-v3-turbo` and the selected `ja` / `zh` language. FFmpeg cuts one audio clip
per non-comment event whose style contains the literal `Gen_`. The clips run in
one batch, retaining the input ASS timings, styles, actors, and other events.
The original ASS is never overwritten. `legacy` keeps the previous ASR behavior;
the CLI still defaults to that engine for compatibility.

Enable **Verify episode PDF** and select the episode PDF to run OpenAI followed
by Gemini before romaji/pinyin, contextual notes, translations, HTML, and Final
Pass. Both verification passes use the same model constants and API keys as
translations, independently of the selected translation checkboxes. PDF
verification also works with **Ignore transcription** for an already transcribed
ASS. It only changes non-comment `Gen_` events. Both providers receive the actual
PDF, including scanned pages, rather than only extracted text. PDFs must be
smaller than 50 MiB and the configured models must support PDF/image input.

```bash
python -m takoworks.modules.transcriber.core input.ass episode.mkv --lang ja \
  --series "Series" --source-type Manga --asr-engine purfview \
  --purfview-exe "C:/Tools/Faster-Whisper-XXL/faster-whisper-xxl.exe" \
  --reference-pdf episode.pdf --do-roman-morph --html --final-pass
```

Each run creates `<base>_initial_<run-id>/` containing `transcribed.ass`,
`verification_gpt.ass`, `verification_gemini.ass`, and `annotated.ass`.
`verification.json` records every returned attempt, corrections, cited PDF pages,
uncertainties, token usage, and estimated USD costs. Partial ASS files preserve
completed verification blocks if a later block fails. Invalid, blocked, or
truncated responses are retried up to three times; an exhausted block stops the
workflow before annotations/translations. Uncertain lines retain their current
text. Page citations are model-provided evidence to review, not a guarantee of
correctness. Temporary provider PDF uploads are deleted after each pass.

Verification costs use the existing `cost_per_1k.gpt` / `cost_per_1k.gemini`
rates, include usage from rejected responses and Gemini thinking tokens, and
appear separately as `verification_gpt` / `verification_gemini` in the total and
Supabase. These charges are persisted even if verification fails or is cancelled.
Local Purfview transcription has no API charge. Costs remain estimates based on
configured rates; no real provider requests are needed to run the unit tests.

## Outputs
- Per-model ASS files: original line, `{romaji/pinyin}`, `{glosses}`, and translation
- Optional HTML summary with columns for original, romaji/pinyin, glosses, GPT, Claude, Gemini, and DeepSeek

## Models and Error Handling
- Supported models: GPT-5.5 (OpenAI), Claude Opus 4.7, Gemini 3 Flash, and DeepSeek V4 Flash. Romaji/pinyin uses DeepSeek V4 Flash. Context notes use Claude Sonnet 4.6.
- If a key is missing or a model fails to initialize, the pipeline skips it and continues with the rest. The user is informed in the console and no output file is generated for that model.
- Costs are calculated when the SDK returns usage data. If Supabase is configured, costs are written to the configured table.

## Supabase Costs
- Define `SUPABASE_URL` + `SUPABASE_SERVICE_KEY` (or use `config.local.json`) to record costs
- Default table: `voicex_api_costs` (configurable with `SUPABASE_COST_TABLE`)

## Tests
- Requires `pytest`. Run: `pytest -q`
- Current coverage: ASS helpers and `parse_json_translations`

## Notes
- The line format for romanization/glosses uses braces `{}` and sanitization to avoid breaking ASS (helpers in `ass_utils.py`)
- Free Chinese punctuation restoration requires `zhpr`; if it is missing, the pipeline continues with the original text
