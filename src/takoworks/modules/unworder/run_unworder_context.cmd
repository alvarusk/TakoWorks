@echo off
setlocal
if "%~1"=="" (
  echo [ERROR] No se ha recibido ningún archivo.
  exit /b 2
)
pushd "%~dp0"
python "%~dp0unworder_context.py" %*
set "EC=%ERRORLEVEL%"
popd
if not "%EC%"=="0" (
  echo.
  echo [ERROR] Exit code: %EC%
  pause
)
exit /b %EC%
