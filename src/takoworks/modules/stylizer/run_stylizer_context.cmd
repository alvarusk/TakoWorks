@echo off
setlocal
if "%~1"=="" (
  echo [ERROR] No se recibio ningun archivo.
  exit /b 2
)
pushd "%~dp0"
py -3 "%~dp0stylizer_context.py" %*
set "EC=%ERRORLEVEL%"
popd
if not "%EC%"=="0" (
  echo.
  echo [ERROR] Exit code: %EC%
  pause
)
exit /b %EC%
