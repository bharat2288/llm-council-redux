@echo off
title Council Server

echo ============================================================
echo                    COUNCIL SERVER
echo ============================================================
echo.
echo Starting server at http://localhost:5000
echo Press Ctrl+C to stop the server
echo ============================================================
echo.

:: Change to script directory
cd /d "%~dp0"

:: Start browser after a short delay (in background)
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:5000"

:: Run the server (this blocks until Ctrl+C)
python council_server.py

pause
