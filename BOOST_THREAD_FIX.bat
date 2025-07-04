@echo off
echo ========================================
echo Boost Thread Build Fix for AudioGridder
echo ========================================
echo.

echo This script fixes the boost-thread build failure by using only required Boost components
echo.

:: Check if we're in the right directory
if not exist "CMakeLists.txt" (
    echo ERROR: Please run this from the AudioGridder-GPU root directory
    pause
    exit /b 1
)

:: Clean previous builds
echo Step 1: Cleaning previous build directories...
if exist "build-windows-gpu-x64" (
    echo Removing old build directory...
    rmdir /s /q "build-windows-gpu-x64"
)

:: Update vcpkg if available
echo.
echo Step 2: Updating vcpkg (if available)...
if exist "C:\vcpkg" (
    cd /d "C:\vcpkg"
    git pull
    call bootstrap-vcpkg.bat
    
    :: Clear problematic boost packages
    echo Removing problematic boost packages...
    vcpkg remove boost:x64-windows --recurse 2>nul
    vcpkg remove boost-thread:x64-windows --recurse 2>nul
    
    cd /d "%~dp0"
) else (
    echo vcpkg not found at C:\vcpkg - will try build without vcpkg
)

:: Create build directory
echo.
echo Step 3: Creating build directory...
mkdir build-windows-gpu-x64
cd build-windows-gpu-x64

:: Configure with vcpkg (if available)
echo.
echo Step 4: Configuring build...
if exist "C:\vcpkg" (
    echo Using vcpkg with minimal dependencies...
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
        -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake ^
        -DVCPKG_TARGET_TRIPLET=x64-windows ^
        -DCMAKE_CXX_STANDARD=17 ^
        -DCMAKE_CXX_STANDARD_REQUIRED=ON ^
        -Wno-dev
) else (
    echo Building without vcpkg...
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
        -DCMAKE_CXX_STANDARD=17 ^
        -DCMAKE_CXX_STANDARD_REQUIRED=ON ^
        -Wno-dev
)

if %errorlevel% neq 0 (
    echo ✗ CMake configuration failed
    echo.
    echo TROUBLESHOOTING:
    echo 1. Make sure Visual Studio 2022 is installed with C++ components
    echo 2. Install CUDA Toolkit if GPU acceleration is needed
    echo 3. Check the error messages above for specific issues
    echo.
    pause
    exit /b 1
)

echo ✓ CMake configuration successful
echo.

:: Build the project
echo Step 5: Building AudioGridder Server...
cmake --build . --config RelWithDebInfo --target AudioGridderServer

if %errorlevel% neq 0 (
    echo ✗ Build failed
    echo.
    echo Check the build output above for specific errors
    pause
    exit /b 1
)

echo ✓ Build successful!
echo.

:: Check if executable was created
if exist "bin\RelWithDebInfo\AudioGridderServer.exe" (
    echo ✓ AudioGridderServer.exe created successfully
    echo Location: %CD%\bin\RelWithDebInfo\AudioGridderServer.exe
    echo.
    echo You can now run the server with:
    echo bin\RelWithDebInfo\AudioGridderServer.exe
) else (
    echo ⚠ Warning: AudioGridderServer.exe not found in expected location
    echo Check the bin directory for the executable
)

echo.
echo Build process completed!
pause
