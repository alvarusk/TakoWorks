@echo off
setlocal
set "ASS=%~1"
if "%ASS%"=="" (
  echo [ERROR] No se recibio ningun archivo.
  pause
  exit /b 2
)

pushd "%~dp0"

rem Log (opcional) para diagnostico
echo [%date% %time%] "%ASS%" >> "%~dp0launch_log.txt"

py -3 "%~dp0corrector.py" "%ASS%"
set "EC=%ERRORLEVEL%"

popd

if not "%EC%"=="0" (
  echo.
  echo [ERROR] Exit code: %EC%
  pause
)
exit /b %EC%
