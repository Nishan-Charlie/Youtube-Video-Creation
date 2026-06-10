@echo off
title YouTube Voiceover Studio - Setup
color 0A

echo ============================================
echo   YouTube Voiceover Studio - Setup
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install from https://python.org
    pause
    exit /b 1
)
echo [OK] Python found

:: Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found. Install from https://nodejs.org
    pause
    exit /b 1
)
echo [OK] Node.js found

echo.
echo [1/3] Installing Python dependencies...
echo       (torch may take a while on first install)
echo.
pip install -r backend\requirements.txt
if errorlevel 1 (
    echo [ERROR] Python dependencies failed.
    pause
    exit /b 1
)

echo.
echo [2/3] Installing Electron...
call npm install
if errorlevel 1 (
    echo [ERROR] npm install failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Creating output folder...
if not exist "output" mkdir output

echo.
echo ============================================
echo   Setup complete! Run start.bat to launch.
echo ============================================
echo.
pause
