# TakoWorks Agent Notes

## Project
- Name: TakoWorks
- Layout: source code in `src/takoworks`, CLI helpers at the repo root, tests in `tests`
- Primary focus in this repo: subtitle tooling for `.ass` files, transcription, romanization, correction, and review workflows

## Working Rules
- Prefer reusing existing helpers in `src/takoworks/modules/transcriber` and `src/takoworks/shared` before adding new parsing logic
- Keep changes non-destructive. Do not revert user changes unless explicitly asked
- Use `apply_patch` for file edits
- Prefer ASCII comments and keep code comments concise unless they clarify non-obvious logic
- Keep CLI and GUI behavior backward compatible unless the task explicitly requests a change

## Useful Commands
- Run tests: `pytest -q`
- Run the main app: `python run_takoworks.py`

## Notes
- `config.json` is shared by the app and CLI tools
- GUI tabs are wired through `src/takoworks/ui/main_window.py`
