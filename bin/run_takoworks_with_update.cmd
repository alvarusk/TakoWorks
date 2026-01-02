@echo off
REM Actualiza desde la ultima release y lanza TakoWorks.
REM Ajusta las rutas si instalas en otro sitio.

set REPO=alvarusk/takoworks
REM Instala en carpeta de usuario para evitar permisos de admin.
set INSTALL_DIR=%LOCALAPPDATA%\TakoWorks
set SCRIPT_DIR=%~dp0

REM Lanza el updater (usa GITHUB_TOKEN si el repo es privado)
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%update_takoworks.ps1" -Repo "%REPO%" -InstallDir "%INSTALL_DIR%"

REM Inicia la app
start "" "%INSTALL_DIR%\TakoWorks.exe"
