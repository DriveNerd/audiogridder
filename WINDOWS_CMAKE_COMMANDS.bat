@echo off
echo ========================================
echo Correct CMake Commands for Windows Build
echo ========================================

echo.
echo Step 1: Clean build directory
if exist "build-windows-x64" (
    rmdir /s /q build-windows-x64
)
mkdir build-windows-x64
cd build-windows-x64

echo.
echo Step 2: Configure with vcpkg (RECOMMENDED)
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
  -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake ^
  -DVCPKG_TARGET_TRIPLET=x64-windows ^
  -DCMAKE_CXX_STANDARD=17 ^
  -DCMAKE_CXX_STANDARD_REQUIRED=ON ^
  -Wno-dev

if %errorlevel% neq 0 (
    echo.
    echo ERROR: CMake configuration failed!
    echo Trying alternative configuration without vcpkg...
    echo.
    
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
    echo.
    echo ERROR: Both configurations failed!
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo Step 3: Build the server
cmake --build . --config RelWithDebInfo --target AudioGridderServer

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Build failed!
    echo Check the error messages above.
    pause
    exit /b 1
)

echo.
echo SUCCESS: Build completed!
echo Executable location: bin\RelWithDebInfo\AudioGridderServer.exe

pause
