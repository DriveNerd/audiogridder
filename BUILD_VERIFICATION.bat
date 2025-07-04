@echo off
echo ========================================
echo AudioGridder GPU Server Build Verification
echo ========================================

echo.
echo Step 1: Checking JUCE submodule...
if not exist "JUCE\CMakeLists.txt" (
    echo ERROR: JUCE submodule not initialized!
    echo Running: git submodule init
    git submodule init
    echo Running: git submodule update --recursive
    git submodule update --recursive
    
    if not exist "JUCE\CMakeLists.txt" (
        echo CRITICAL ERROR: JUCE submodule failed to initialize
        echo Please run manually: git submodule update --init --recursive --force
        pause
        exit /b 1
    )
    echo SUCCESS: JUCE submodule initialized
) else (
    echo SUCCESS: JUCE submodule already present
)

echo.
echo Step 2: Checking dependencies...

echo Checking Boost...
if exist "C:\local\boost_1_87_0" (
    echo SUCCESS: Boost found at C:\local\boost_1_87_0
    set BOOST_ROOT=C:\local\boost_1_87_0
) else (
    echo WARNING: Boost not found at expected location
    echo Please install Boost from: https://sourceforge.net/projects/boost/files/boost-binaries/
)

echo Checking FFmpeg...
ffmpeg -version >nul 2>&1
if %errorlevel% equ 0 (
    echo SUCCESS: FFmpeg is available in PATH
) else (
    echo WARNING: FFmpeg not found in PATH
    echo Installing via Chocolatey...
    choco install ffmpeg --params "/InstallDir:C:\ffmpeg" -y
    set FFMPEG_ROOT=C:\ffmpeg
)

echo Checking CUDA...
nvcc --version >nul 2>&1
if %errorlevel% equ 0 (
    echo SUCCESS: CUDA toolkit is available
) else (
    echo WARNING: CUDA toolkit not found
    echo Please install CUDA Toolkit 11.8+ from NVIDIA
)

echo.
echo Step 3: Cleaning build directory...
if exist "build-windows-x64" (
    rmdir /s /q build-windows-x64
    echo Removed existing build directory
)
mkdir build-windows-x64
cd build-windows-x64

echo.
echo Step 4: Configuring CMake...
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
  -DBOOST_ROOT=%BOOST_ROOT% ^
  -DFFMPEG_ROOT=%FFMPEG_ROOT% ^
  -Wno-dev

if %errorlevel% neq 0 (
    echo ERROR: CMake configuration failed!
    echo Check the error messages above
    pause
    exit /b 1
)

echo SUCCESS: CMake configuration completed

echo.
echo Step 5: Building AudioGridder Server...
cmake --build . --config RelWithDebInfo --target AudioGridderServer

if %errorlevel% neq 0 (
    echo ERROR: Build failed!
    echo Check the error messages above
    pause
    exit /b 1
)

echo.
echo Step 6: Verifying build output...
if exist "bin\RelWithDebInfo\AudioGridderServer.exe" (
    echo SUCCESS: AudioGridderServer.exe built successfully!
    echo Location: %CD%\bin\RelWithDebInfo\AudioGridderServer.exe
    
    echo.
    echo Step 7: Testing GPU server functionality...
    cd bin\RelWithDebInfo
    
    echo Testing help command...
    AudioGridderServer.exe --help
    
    echo.
    echo Testing GPU detection...
    AudioGridderServer.exe --list-gpu-devices
    
    echo.
    echo ========================================
    echo BUILD SUCCESSFUL!
    echo ========================================
    echo.
    echo Your GPU server executable is ready at:
    echo %CD%\AudioGridderServer.exe
    echo.
    echo To start the server:
    echo AudioGridderServer.exe
    echo.
    
) else (
    echo ERROR: AudioGridderServer.exe not found after build!
    echo Build may have failed silently
    dir /s *.exe
    pause
    exit /b 1
)

echo Build verification completed successfully!
pause
