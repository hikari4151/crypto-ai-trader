@echo off
setlocal enabledelayedexpansion
title Crypto AI Trader - One-Click Start

echo ========================================
echo   Crypto AI Trader - One-Click Start
echo ========================================
echo.

REM --- Jump to project root ---
cd /d "%~dp0"

REM --- Check virtual environment ---
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual env ".venv" not found.
    echo   Run:  python -m venv .venv
    echo         .venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM --- Default port ---
set PORT=8000
if not "%1"=="" set PORT=%1

REM --- Find first free port in 8000-8010 (port in use → try next) ---
for /L %%P in (!PORT!,1,8010) do (
    "%.venv\Scripts\python.exe" -c "import socket,sys; s=socket.socket(); s.settimeout(0.4); r=s.connect_ex(('127.0.0.1',int(sys.argv[1]))); s.close(); sys.exit(0 if r==0 else 1)" %%P
    if !errorlevel!==1 (
        set PORT=%%P
        goto :port_found
    )
)
echo [ERROR] Ports !PORT!-8010 all in use. Close other instances and retry.
pause
exit /b 1

:port_found
if not "!PORT!"=="8000" echo [INFO] Port 8000 in use, using port !PORT! instead.

echo [1/2] Starting web server on port !PORT! ...
echo       URL: http://localhost:!PORT!
echo.

REM --- Open browser after 3s (non-blocking) ---
start "" /b cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:!PORT!"

REM --- Run server in foreground ---
".venv\Scripts\python.exe" run.py web --port !PORT!

echo.
echo Server stopped.
pause
endlocal
