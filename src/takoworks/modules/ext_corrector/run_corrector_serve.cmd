@echo off
setlocal
set "ASS=%~1"
if "%ASS%"=="" (
  echo [ERROR] No se ha recibido ningún archivo.
  pause
  exit /b 2
)

pushd "%~dp0"

echo [%date% %time%] "%ASS%" --serve >> "%~dp0launch_log.txt"

py -3 "%~dp0corrector.py" "%ASS%" --serve --serve-idle 45
set "EC=%ERRORLEVEL%"

popd

if not "%EC%"=="0" (
  echo.
  echo [ERROR] Exit code: %EC%
  pause
)
exit /b %EC%
