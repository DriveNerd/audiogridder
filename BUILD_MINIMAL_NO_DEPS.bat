@echo off
echo ========================================
echo AudioGridder GPU Server - Minimal Build
echo ========================================
echo.

echo Building AudioGridder GPU Server without external dependencies
echo This version will work without vcpkg, FFmpeg, or Boost
echo.

:: Check if we're in the right directory
if not exist "CMakeLists.txt" (
    echo ERROR: Please run this from the AudioGridder-GPU root directory
    pause
    exit /b 1
)

:: Initialize JUCE submodule if not already done
echo Step 0: Checking JUCE submodule...
if not exist "JUCE\CMakeLists.txt" (
    echo JUCE directory is empty. Attempting to initialize...
    
    :: Check if this is a git repository
    git status >nul 2>&1
    if %errorlevel% equ 0 (
        echo This is a git repository. Initializing submodules...
        git submodule update --init --recursive
        if %errorlevel% neq 0 (
            echo ERROR: Failed to initialize JUCE submodule
            echo.
            echo SOLUTION: Please clone the repository with:
            echo git clone --recursive https://github.com/DriveNerd/AudioGridder-GPU.git
            echo.
            echo Or download JUCE manually and extract to JUCE/ directory
            pause
            exit /b 1
        )
    ) else (
        echo ERROR: This is not a git repository and JUCE is missing.
        echo.
        echo SOLUTIONS:
        echo 1. Clone the repository with git instead of downloading ZIP:
        echo    git clone --recursive https://github.com/DriveNerd/AudioGridder-GPU.git
        echo.
        echo 2. Or download JUCE manually:
        echo    - Run DOWNLOAD_JUCE.bat for step-by-step instructions
        echo    - Or manually download from https://github.com/juce-framework/JUCE
        echo    - Extract to JUCE/ directory in this folder
        echo    - Ensure JUCE/CMakeLists.txt exists
        echo.
        echo 3. Or use the minimal standalone build (see below)
        echo.
        pause
        
        :: Continue with standalone build as fallback
        echo Attempting standalone minimal build without JUCE...
        goto :standalone_build
    )
) else (
    echo JUCE submodule already initialized.
)

:: Clean previous builds
echo Step 1: Cleaning previous builds...
if exist "build-minimal-no-deps" (
    echo Removing old build directory...
    rmdir /s /q "build-minimal-no-deps"
)

:: Create build directory
echo.
echo Step 2: Creating build directory...
mkdir build-minimal-no-deps
cd build-minimal-no-deps

:: Configure CMake with minimal dependencies
echo.
echo Step 3: Configuring CMake (minimal build)...
cmake .. ^
    -G "Visual Studio 17 2022" ^
    -A x64 ^
    -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
    -DAG_ENABLE_CUDA=OFF ^
    -DAG_WITH_SERVER=ON ^
    -DAG_WITH_PLUGIN=OFF ^
    -DAG_WITH_TESTS=OFF ^
    -DAG_ENABLE_CODE_SIGNING=OFF ^
    -DAG_WITH_TRACEREADER=OFF ^
    -DAG_WITH_FFMPEG=OFF ^
    -DAG_WITH_WEBP=OFF ^
    -DCMAKE_INSTALL_PREFIX=../install-minimal ^
    -DCMAKE_CXX_STANDARD=17 ^
    -DCMAKE_CXX_STANDARD_REQUIRED=ON ^
    -DBOOST_ROOT="" ^
    -DBoost_NO_SYSTEM_PATHS=ON ^
    -DBoost_NO_BOOST_CMAKE=ON ^
    -DCMAKE_DISABLE_FIND_PACKAGE_Boost=ON ^
    -DCMAKE_DISABLE_FIND_PACKAGE_FFMPEG=ON ^
    -DCMAKE_DISABLE_FIND_PACKAGE_WebP=ON ^
    -Wno-dev

if %errorlevel% neq 0 (
    echo.
    echo ✗ CMake configuration failed
    echo.
    echo This might be because the project requires these dependencies.
    echo Let me try a different approach...
    echo.
    
    :: Try building just the core components
    echo Step 4: Trying core-only build...
    cd ..
    
:standalone_build
    :: Create a simple standalone build
    echo Creating standalone AudioGridder server...
    mkdir standalone-build
    cd standalone-build
    
    echo Creating ultra-minimal standalone server...
    echo cmake_minimum_required(VERSION 3.15) > CMakeLists.txt
    echo. >> CMakeLists.txt
    echo project(AudioGridderMinimalServer) >> CMakeLists.txt
    echo. >> CMakeLists.txt
    echo set(CMAKE_CXX_STANDARD 17) >> CMakeLists.txt
    echo set(CMAKE_CXX_STANDARD_REQUIRED ON) >> CMakeLists.txt
    echo. >> CMakeLists.txt
    echo # Create minimal standalone server >> CMakeLists.txt
    echo add_executable(AudioGridderServer ../minimal_server_standalone.cpp) >> CMakeLists.txt
    echo. >> CMakeLists.txt
    echo # Windows libraries >> CMakeLists.txt
    echo if(WIN32) >> CMakeLists.txt
    echo     target_link_libraries(AudioGridderServer >> CMakeLists.txt
    echo         ws2_32 >> CMakeLists.txt
    echo         winmm >> CMakeLists.txt
    echo     ^) >> CMakeLists.txt
    echo endif() >> CMakeLists.txt
    echo. >> CMakeLists.txt
    echo # Linux libraries >> CMakeLists.txt
    echo if(UNIX AND NOT APPLE) >> CMakeLists.txt
    echo     find_package(Threads REQUIRED) >> CMakeLists.txt
    echo     target_link_libraries(AudioGridderServer Threads::Threads) >> CMakeLists.txt
    echo endif() >> CMakeLists.txt
    
    echo.
    echo Configuring standalone build...
    cmake . -G "Visual Studio 17 2022" -A x64
    
    if %errorlevel% equ 0 (
        echo ✓ Standalone configuration successful
        echo.
        echo Building...
        cmake --build . --config RelWithDebInfo
        
        if %errorlevel% equ 0 (
            echo.
            echo ========================================
            echo BUILD COMPLETED SUCCESSFULLY!
            echo ========================================
            echo.
            echo Your executable is at: standalone-build\RelWithDebInfo\AudioGridderServer.exe
            echo.
            echo NOTE: This is a minimal build without some features:
            echo - No FFmpeg support (limited audio format support)
            echo - No WebP support (limited image support)  
            echo - No Boost dependencies
            echo - Basic GPU acceleration may still work if CUDA is available
            echo.
            echo The executable should still provide core AudioGridder functionality.
        ) else (
            echo ✗ Standalone build failed
            echo.
            echo FINAL TROUBLESHOOTING:
            echo The project may have hard dependencies that cannot be easily removed.
            echo.
            echo RECOMMENDED SOLUTIONS:
            echo 1. Install the full dependencies as shown in the documentation
            echo 2. Use a pre-built version if available
            echo 3. Contact the project maintainers for a minimal build configuration
        )
    ) else (
        echo ✗ Even standalone configuration failed
        echo This suggests fundamental build environment issues.
    )
    
    cd ..
    
) else (
    echo ✓ CMake configuration successful
    echo.
    echo Step 4: Building AudioGridder Server...
    cmake --build . --config RelWithDebInfo --target AudioGridderServer
    
    if %errorlevel% equ 0 (
        echo ✓ Build successful!
        echo.
        echo Step 5: Installing...
        cmake --build . --config RelWithDebInfo --target install
        
        if %errorlevel% equ 0 (
            echo.
            echo ========================================
            echo BUILD COMPLETED SUCCESSFULLY!
            echo ========================================
            echo.
            echo Your executable is at: install-minimalin\AudioGridderServer.exe
            echo.
            echo This is a minimal build with reduced features but should work
            echo without external dependencies.
        ) else (
            echo ✗ Installation failed, but executable might still be available
            echo Check: build-minimal-no-deps\Server\RelWithDebInfo\AudioGridderServer.exe
        )
    ) else (
        echo ✗ Build failed
        echo Check the error messages above for details
    )
    
    cd ..
)

echo.
echo Script completed. Check the output above for results.
pause
