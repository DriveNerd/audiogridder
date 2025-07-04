@echo off
echo ========================================
echo Visual Studio Environment Setup Helper
echo ========================================
echo.

echo This script will help you set up the Visual Studio environment for building.
echo.

:: Try to find Visual Studio installation
set "VS_INSTALL_PATH="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"

if exist "%VSWHERE%" (
    echo Found Visual Studio Installer...
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        set "VS_INSTALL_PATH=%%i"
    )
)

if defined VS_INSTALL_PATH (
    echo Visual Studio found at: %VS_INSTALL_PATH%
    echo.
    echo Setting up Visual Studio environment...
    
    :: Set up the Visual Studio environment
    call "%VS_INSTALL_PATH%\VC\Auxiliary\Build\vcvars64.bat"
    
    if %errorlevel% equ 0 (
        echo ✓ Visual Studio environment configured successfully!
        echo.
        echo You can now run: BUILD_WINDOWS_GPU_SERVER.bat
        echo.
        echo Press any key to start the build process automatically...
        pause >nul
        
        echo Starting build...
        call BUILD_WINDOWS_GPU_SERVER.bat
    ) else (
        echo ✗ Failed to set up Visual Studio environment
        echo Please check your Visual Studio installation
        pause
    )
) else (
    echo ✗ Visual Studio 2022 with C++ tools not found!
    echo.
    echo SOLUTION OPTIONS:
    echo.
    echo Option 1: Install Visual Studio 2022 Community (Recommended)
    echo   1. Download from: https://visualstudio.microsoft.com/vs/community/
    echo   2. During installation, select "Desktop development with C++" workload
    echo   3. Make sure "MSVC v143 - VS 2022 C++ x64/x86 build tools" is checked
    echo   4. Install and restart your computer
    echo.
    echo Option 2: Use Visual Studio Developer Command Prompt
    echo   1. Start Menu → Visual Studio 2022 → Developer Command Prompt for VS 2022
    echo   2. Navigate to this directory: cd /d "%~dp0"
    echo   3. Run: BUILD_WINDOWS_GPU_SERVER.bat
    echo.
    echo Option 3: Use Build Tools for Visual Studio 2022 (Minimal)
    echo   1. Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
    echo   2. Install with C++ build tools
    echo.
    pause
)
