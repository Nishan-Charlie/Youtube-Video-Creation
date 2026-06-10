@echo off
title YouTube Voiceover Studio - Build Backend
color 0B

echo ============================================
echo   YouTube Voiceover Studio - Build Backend
echo ============================================
echo.
echo This will bundle backend\server.py into a
echo standalone backend_server.exe using PyInstaller.
echo.
echo WARNING: First build may take 10-20 minutes.
echo Output size will be several GB (due to PyTorch).
echo.
pause

:: Check PyInstaller is installed
pyinstaller --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] PyInstaller not found. Installing...
    pip install pyinstaller
    if errorlevel 1 (
        echo [ERROR] Failed to install PyInstaller.
        pause
        exit /b 1
    )
)
echo [OK] PyInstaller found.

:: Clean previous build artifacts
echo.
echo [1/3] Cleaning previous build...
if exist "dist_backend" rmdir /s /q dist_backend
if exist "build_pyinstaller" rmdir /s /q build_pyinstaller
if exist "backend_server.spec" del /f backend_server.spec

:: Run PyInstaller
echo.
echo [2/3] Compiling backend (this will take a while)...
echo.

pyinstaller ^
    --onefile ^
    --name backend_server ^
    --distpath dist_backend ^
    --workpath build_pyinstaller ^
    --specpath . ^
    --hidden-import flask ^
    --hidden-import flask_cors ^
    --hidden-import google.genai ^
    --hidden-import google.genai.types ^
    --hidden-import chatterbox ^
    --hidden-import chatterbox.tts ^
    --hidden-import torch ^
    --hidden-import torchaudio ^
    --hidden-import torchaudio.transforms ^
    --hidden-import numpy ^
    --hidden-import librosa ^
    --hidden-import soundfile ^
    --hidden-import kokoro ^
    --hidden-import TTS ^
    --hidden-import TTS.api ^
    --hidden-import openai ^
    --hidden-import tkinter ^
    --hidden-import tkinter.filedialog ^
    --hidden-import safetensors ^
    --hidden-import safetensors.torch ^
    --hidden-import perth ^
    --add-data "backend\.env;." ^
    --add-data "backend\requirements.txt;." ^
    --collect-all torch ^
    --collect-all torchaudio ^
    --collect-submodules torch ^
    backend\server.py

if errorlevel 1 (
    echo.
    echo [ERROR] PyInstaller build failed. See output above.
    pause
    exit /b 1
)

:: Verify output
echo.
echo [3/3] Verifying output...
if exist "dist_backend\backend_server.exe" (
    echo.
    echo ============================================
    echo   Build complete!
    echo   Output: dist_backend\backend_server.exe
    echo ============================================
    echo.
    echo Next step: run  npm run build:win
    echo.
) else (
    echo [ERROR] backend_server.exe not found after build.
    exit /b 1
)

pause
