@echo off
title Safe Station Streamlit Dashboard
echo =======================================================
echo   Safe Station Streamlit Interactive Dashboard
echo =======================================================
echo Launching Streamlit on http://localhost:8501 ...
cd /d "%~dp0"
streamlit run streamlit_app.py
pause
