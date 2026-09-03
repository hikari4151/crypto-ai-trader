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

REM --- Find first free port in 8000-8010 (in use - try next) ---
for /L %%P in (!PORT!,1,8010) do (
    ".venv\Scripts\python.exe" scripts\check_port.py %%P
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

REM --- Start server in background (detached, output to log file) ---
start "CryptoAI-Trader" /B ".venv\Scripts\python.exe" run.py web --port !PORT! > data\logs\server.log 2>&1
set SERVER_PID=!ERRORLEVEL!

REM --- Wait for server to be ready (probe health endpoint, max 60s) ---
echo [2/2] Waiting for server to be ready...
REM Use the bundled python probe instead of curl + timeout:
REM   - curl on localhost may resolve to IPv6 ::1 while the server binds
REM     127.0.0.1, so the probe never gets 200 and the old code opened
REM     the browser after a 30s timeout -> "cannot connect" first.
REM   - timeout /t fails instantly in non-interactive batch contexts.
REM   - This file must stay pure ASCII (no CJK comments) so cmd's GBK
REM     parser does not choke and flash-exit the window.
".venv\Scripts\python.exe" scripts\wait_ready.py !PORT!
if !errorlevel!==0 (
    echo       Server ready. Opening browser...
    goto :open_browser
)

echo.
echo [ERROR] Server did not respond within timeout.
echo   Check the log for details:
if exist data\logs\server.log (
    echo   --- server.log last 20 lines ---
    powershell -NoProfile -Command "Get-Content -Tail 20 'data\logs\server.log'"
    echo   -------------------------------
)
echo.
echo Press any key to exit.
pause >nul
exit /b 1

:open_browser
start http://localhost:!PORT!

echo.
echo Server is running in background (PID: !SERVER_PID!).
echo Close this window or press Ctrl+C to stop the server.
echo.

REM --- Keep window open, wait for user to close ---
pause >nul
echo Stopping server...
taskkill /FI "WINDOWTITLE eq CryptoAI-Trader" /F 2>nul
echo Server stopped.
endlocal
