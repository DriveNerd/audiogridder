@echo off
echo ========================================
echo vcpkg Dependencies Fix for AudioGridder
echo ========================================
echo.

echo This script will fix common vcpkg dependency issues
echo.

:: Check if we're in the right directory
if not exist "CMakeLists.txt" (
    echo ERROR: Please run this from the AudioGridder-GPU root directory
    pause
    exit /b 1
)

:: Clean previous vcpkg builds
echo Step 1: Cleaning previous vcpkg builds...
if exist "build-windows-gpu-x64" (
    echo Removing old build directory...
    rmdir /s /q "build-windows-gpu-x64"
)

:: Update vcpkg
echo.
echo Step 2: Updating vcpkg...
cd /d "C:\vcpkg"
git pull
call bootstrap-vcpkg.bat

:: Clear vcpkg cache
echo.
echo Step 3: Clearing vcpkg cache...
vcpkg remove --outdated --recurse

:: Install dependencies individually with fallback options
echo.
echo Step 4: Installing dependencies with fallback options...
cd /d "%~dp0"

:: Try minimal dependency set first
echo Installing minimal dependencies...
C:\vcpkg\vcpkg install ffmpeg[core]:x64-windows --recurse

if %errorlevel% neq 0 (
    echo FFmpeg installation failed, trying alternative approach...
    
    :: Try without boost dependencies
    echo.
    echo Step 5: Building without problematic dependencies...
    mkdir build-windows-gpu-x64-minimal
    cd build-windows-gpu-x64-minimal
    
    cmake .. ^
        -G "Visual Studio 17 2022" ^
        -A x64 ^
        -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
        -DAG_ENABLE_CUDA=ON ^
        -DAG_WITH_SERVER=ON ^
        -DAG_WITH_PLUGIN=OFF ^
        -DAG_WITH_TESTS=OFF ^
        -DAG_ENABLE_CODE_SIGNING=OFF ^
        -DAG_WITH_TRACEREADER=OFF ^
        -DCMAKE_INSTALL_PREFIX=../install-windows-gpu ^
        -DCMAKE_CXX_STANDARD=17 ^
        -DCMAKE_CXX_STANDARD_REQUIRED=ON ^
        -Wno-dev
    
    if %errorlevel% equ 0 (
        echo ✓ CMake configuration successful without vcpkg
        echo.
        echo Building...
        cmake --build . --config RelWithDebInfo --target AudioGridderServer
        
        if %errorlevel% equ 0 (
            echo ✓ Build successful!
            echo.
            echo Installing...
            cmake --build . --config RelWithDebInfo --target install
            
            if %errorlevel% equ 0 (
                echo ✓ Installation successful!
                echo.
                echo Your executable is at: install-windows-gpu\bin\AudioGridderServer.exe
            )
        )
    ) else (
        echo ✗ Build failed even without vcpkg dependencies
        echo.
        echo TROUBLESHOOTING OPTIONS:
        echo 1. Make sure you have all Visual Studio C++ components installed
        echo 2. Try installing dependencies manually
        echo 3. Check the CMake error messages above
    )
    
    cd ..
) else (
    echo ✓ Dependencies installed successfully
    echo.
    echo Now running the main build script...
    call BUILD_WINDOWS_GPU_SERVER.bat
)

echo.
echo Script completed. Check the output above for results.
pause
