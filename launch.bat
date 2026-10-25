@echo off
title Safe Station EV Dashboard
echo =======================================================
echo   Safe Station EV RL Environment Dashboard Launcher
echo =======================================================
echo Starting Safe Station server on http://localhost:8000 ...
cd /d "%~dp0"
python run_frontend.py
pause
