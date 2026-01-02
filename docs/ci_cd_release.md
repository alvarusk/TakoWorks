## Flujo rapido (VS Code -> GitHub -> release)
- Trabaja en rama `main` o feature, haz commit/push desde VS Code.
- En cada push a `main` o `workflow_dispatch` se ejecuta `pytest`, se construye el exe con PyInstaller y se sube un zip como artifact.
- Al publicar una Release en GitHub, el mismo workflow adjunta el zip `TakoWorks_win64.zip` a la Release para que el updater lo descargue.

## Requisitos previos
- Ajusta `requirements-lock.txt` si anades nuevos deps (el workflow instala todo desde ahi).
- Instala las secrets de API en tu maquina (no van al workflow). `GITHUB_TOKEN` lo inyecta GitHub Actions para publicar el asset.
- Opcional: edita `bin/update_takoworks.ps1` y pon tu repo en `$Repo` (formato `owner/TakoWorks`).

## Workflow (`.github/workflows/release.yml`)
- Dispara en `push` a `main`, `workflow_dispatch` y `release: published`.
- Job `tests` (ubuntu): instala deps y ejecuta `pytest -q`.
- Job `build-windows` (Windows): PyInstaller con `TakoWorks.spec`, comprime `dist/TakoWorks` a `TakoWorks_win64.zip`, sube artifact y lo adjunta a la Release si la ejecucion viene de `release`.

## Como crear una Release rapida
1) Asegura que `main` tiene los cambios listos.
2) Crea un tag y Release en GitHub (desde la UI). El workflow construye y adjunta el zip.
3) El asset de la Release se llama `TakoWorks_win64.zip` (contiene los binarios PyInstaller).

## Updater en la maquina instalada
- Script: `bin/update_takoworks.ps1`
- Uso basico:
  ```powershell
  # Ajusta owner/repositorio y la ruta donde esta instalado
  .\bin\update_takoworks.ps1 -Repo "owner/TakoWorks" -InstallDir "C:\Program Files\TakoWorks"
  ```
- Opciones:
  - `-AllowPrerelease` para permitir pre-releases.
  - Usa env var `GITHUB_TOKEN` si el repo es privado o para evitar rate limit.
- Que hace:
  - Busca el ultimo release (no draft; sin prerelease salvo que pases el flag).
  - Descarga el asset `TakoWorks_win64.zip`.
  - Descomprime en temp, hace backup de la instalacion actual como `<InstallDir>.bak.<timestamp>` y copia los nuevos binarios.

## Verificacion rapida
- Local: `python -m pip install -r requirements-lock.txt && pytest -q`
- En GitHub Actions: revisa la pestana Actions -> workflow `ci-release` para ver el artefacto o el asset de la Release.
