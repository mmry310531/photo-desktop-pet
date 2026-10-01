@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (echo 請先執行 1-安裝.bat & pause & exit /b 1)
start "" ".venv\Scripts\pythonw.exe" "app\pet.py"
