@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 桌面寵物 - 安裝
echo ============================================
echo   桌面寵物 安裝程式（只需要執行一次）
echo ============================================
echo.

set "PY="
for %%V in (3.12 3.11 3.13 3.10) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
  )
)
if not defined PY (
  echo 找不到合適的 Python，正在用 winget 安裝 Python 3.12 ...
  winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
  if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe""
  ) else (
    echo.
    echo [失敗] 無法自動安裝 Python。請到 https://www.python.org/downloads/ 安裝 Python 3.12，
    echo        安裝時勾選 "Add python.exe to PATH"，然後重新執行這個檔案。
    pause
    exit /b 1
  )
)
echo 使用 Python：%PY%

if not exist ".venv\Scripts\python.exe" (
  echo 建立獨立環境 .venv ...
  %PY% -m venv .venv || (echo [失敗] 建立環境失敗 & pause & exit /b 1)
)
echo 下載並安裝套件（約 250MB，多條連線同時下載）...
".venv\Scripts\python.exe" app\fastdl.py install || (echo [失敗] 套件安裝失敗，請檢查網路後重新執行本檔案（已下載的部分不會重抓） & pause & exit /b 1)
".venv\Scripts\python.exe" app\fastdl.py models

echo 在桌面建立捷徑 ...
".venv\Scripts\python.exe" app\make_shortcuts.py

echo.
echo ============================================
echo   安裝完成！桌面上有「寵物訓練器」和「桌面寵物」兩個捷徑。
echo   現在幫你打開訓練器，把寵物照片拖進去吧。
echo ============================================
start "" ".venv\Scripts\pythonw.exe" "app\trainer.py"
pause
