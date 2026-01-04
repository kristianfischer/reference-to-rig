@echo off
REM Reference-to-Rig Setup Script for Windows
REM Run this from the project root directory

echo ========================================
echo Reference-to-Rig Setup
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+
    exit /b 1
)
echo [OK] Python found

REM Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js not found. Please install Node.js 18+
    exit /b 1
)
echo [OK] Node.js found

REM Setup Python environment
echo.
echo Setting up Python engine...
cd engine
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
cd ..
echo [OK] Python engine setup complete

REM Setup UI
echo.
echo Setting up UI...
cd ui
call npm install
cd ..
echo [OK] UI setup complete

REM Build index
echo.
echo Building capture library index...
cd engine
call venv\Scripts\python -m scripts.build_index
cd ..
echo [OK] Index built

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To start the application:
echo   1. Open a terminal and run: cd engine ^&^& venv\Scripts\activate ^&^& uvicorn app.main:app --reload --port 8000
echo   2. Open another terminal and run: cd ui ^&^& npm run dev
echo.
echo Then open http://localhost:1420 in your browser.
echo.


