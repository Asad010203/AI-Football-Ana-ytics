@echo off
set "WORKER_DIR=%~dp0"
start "" "%WORKER_DIR%football-worker.exe"
timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:8000"
