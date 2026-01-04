@echo off
setlocal enabledelayedexpansion

REM =====================================================
REM JUCE Plugin Setup Script for Reference to Rig
REM This script sets up the JUCE framework and builds the plugin.
REM =====================================================

echo.
echo ========================================
echo   Reference to Rig - JUCE Setup
echo ========================================
echo.

REM Store the plugin directory path
set "PLUGIN_DIR=%~dp0"
cd /d "%PLUGIN_DIR%"

echo Working directory: %CD%
echo.

REM Check if JUCE is already present
if exist "JUCE\CMakeLists.txt" (
    echo [OK] JUCE framework already present
) else (
    echo [INFO] Downloading JUCE framework...
    
    REM Clone JUCE
    git clone --depth 1 --branch 7.0.9 https://github.com/juce-framework/JUCE.git JUCE
    
    if errorlevel 1 (
        echo [ERROR] Failed to clone JUCE. Please check your internet connection.
        pause
        exit /b 1
    )
    
    echo [OK] JUCE downloaded successfully
)

echo.

REM Check for CMake
where cmake >nul 2>nul
if errorlevel 1 (
    echo [ERROR] CMake not found in PATH.
    echo Please install CMake from https://cmake.org/download/
    echo Make sure to add it to your system PATH during installation.
    pause
    exit /b 1
)

echo [OK] CMake found

REM Check for Visual Studio
set "VS_FOUND=0"

REM Check for VS 2022
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" set "VS_FOUND=2022"
if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" set "VS_FOUND=2022"
if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" set "VS_FOUND=2022"

REM Check for VS 2019
if "%VS_FOUND%"=="0" (
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" set "VS_FOUND=2019"
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat" set "VS_FOUND=2019"
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat" set "VS_FOUND=2019"
)

if "%VS_FOUND%"=="0" (
    echo [ERROR] Visual Studio 2019 or 2022 not found.
    echo Please install Visual Studio with C++ workload.
    pause
    exit /b 1
)

echo [OK] Visual Studio %VS_FOUND% found

REM Set the generator based on VS version
if "%VS_FOUND%"=="2022" (
    set "CMAKE_GENERATOR=Visual Studio 17 2022"
) else (
    set "CMAKE_GENERATOR=Visual Studio 16 2019"
)

echo.
echo Using CMake generator: %CMAKE_GENERATOR%
echo.

REM Create and configure build directory
if not exist "build" mkdir build
cd build

echo [INFO] Configuring CMake project...
cmake .. -G "%CMAKE_GENERATOR%" -A x64

if errorlevel 1 (
    echo [ERROR] CMake configuration failed.
    pause
    exit /b 1
)

echo.
echo [OK] CMake configuration complete
echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo To build the plugin:
echo   1. Open build\ReferenceToRig.sln in Visual Studio
echo   2. Select Release configuration
echo   3. Build the solution (Ctrl+Shift+B)
echo.
echo Or build from command line:
echo   cmake --build build --config Release
echo.

pause

