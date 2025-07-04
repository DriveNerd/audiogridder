# Boost Configuration Fix for Windows Build

## Current Error
```
CMake Error: Could not find a package configuration file provided by "Boost"
Could not find BoostConfig.cmake or boost-config.cmake
```

## Root Cause
CMake cannot find the Boost installation even though it exists at C:\local\boost_1_87_0.

## Solution 1: Set Boost_DIR explicitly

```cmd
cd I:\AudioGridder\AudioGridder-GPU\build-windows-x64

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
  -DBOOST_ROOT=C:\local\boost_1_87_0 ^
  -DBoost_DIR=C:\local\boost_1_87_0\lib\cmake\Boost-1.87.0 ^
  -DFFMPEG_ROOT=C:\ffmpeg ^
  -DCMAKE_CXX_STANDARD=17 ^
  -DCMAKE_CXX_STANDARD_REQUIRED=ON ^
  -Wno-dev
```

## Solution 2: Use FindBoost module mode

```cmd
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
  -DBOOST_ROOT=C:\local\boost_1_87_0 ^
  -DBoost_USE_STATIC_LIBS=ON ^
  -DBoost_USE_MULTITHREADED=ON ^
  -DBoost_USE_STATIC_RUNTIME=OFF ^
  -DFFMPEG_ROOT=C:\ffmpeg ^
  -DCMAKE_CXX_STANDARD=17 ^
  -DCMAKE_CXX_STANDARD_REQUIRED=ON ^
  -Wno-dev
```

## Solution 3: Install Boost via vcpkg (Recommended)

```cmd
cd C:\vcpkg

# Install only required Boost components
.\vcpkg install boost-program-options:x64-windows
.\vcpkg install boost-system:x64-windows
.\vcpkg install boost-filesystem:x64-windows

# Configure with vcpkg
cd I:\AudioGridder\AudioGridder-GPU\build-windows-x64

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
```

## Solution 4: Verify Boost Installation

Check if Boost is properly installed:

```cmd
# Check if Boost directory exists
dir C:\local\boost_1_87_0

# Look for CMake config files
dir C:\local\boost_1_87_0\lib\cmake /s

# Check for boost libraries
dir C:\local\boost_1_87_0\lib\*.lib
```

## Solution 5: Alternative Boost Installation

If current Boost installation is problematic:

```cmd
# Uninstall current Boost
choco uninstall boost-msvc-14.3

# Install specific version
choco install boost-msvc-14.3 --version=1.82.0

# Or download pre-built binaries from:
# https://sourceforge.net/projects/boost/files/boost-binaries/1.82.0/
```

## Recommended Quick Fix

Try Solution 1 first (set Boost_DIR explicitly):

```cmd
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
  -DBOOST_ROOT=C:\local\boost_1_87_0 ^
  -DBoost_DIR=C:\local\boost_1_87_0\lib\cmake\Boost-1.87.0 ^
  -DCMAKE_CXX_STANDARD=17 ^
  -DCMAKE_CXX_STANDARD_REQUIRED=ON ^
  -Wno-dev

cmake --build . --config RelWithDebInfo --target AudioGridderServer
```

This should resolve the Boost configuration error and allow the build to proceed.
