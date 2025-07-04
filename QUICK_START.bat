@echo off
echo ========================================
echo AudioGridder GPU Server Quick Start
echo ========================================
echo.
echo This script will help you build the AudioGridder GPU Server.
echo.
echo Prerequisites check:
echo - Make sure you're running this from a Visual Studio Developer Command Prompt
echo - Ensure you have Visual Studio 2022 with C++ workload installed
echo - Install CMake and add it to PATH
echo - Install CUDA Toolkit (optional, for GPU acceleration)
echo.
echo Press any key to start the build process...
pause >nul

echo.
echo Starting build...
call BUILD_WINDOWS_GPU_SERVER.bat

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Build failed! Check the output above for details.
    pause
    exit /b 1
)

echo.
echo Build process completed successfully!
echo Check the install-windows-gpu directory for your executable.
pause
