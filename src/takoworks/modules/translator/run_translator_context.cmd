@echo off
setlocal
set "ASS=%~1"
if "%ASS%"=="" (
  echo [ERROR] No input file.
  exit /b 2
)
py -3 "%~dp0translator_context.py" "%ASS%"
