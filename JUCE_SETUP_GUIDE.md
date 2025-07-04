# JUCE Framework Setup Guide for Windows

## Current Issues:
1. Missing JUCE CMakeLists.txt file
2. FFmpeg and WebP not found
3. Unknown CMake command "juce_add_gui_app"

## Solution 1: Initialize JUCE Submodule

The JUCE directory exists but is empty because it's a git submodule that wasn't initialized.

```cmd
cd I:\AudioGridder\AudioGridder-GPU

# Initialize and update git submodules
git submodule init
git submodule update --recursive

# Verify JUCE is now populated
dir JUCE
dir JUCE\CMakeLists.txt
```

## Solution 2: Manual JUCE Download (if submodule fails)

```cmd
cd I:\AudioGridder\AudioGridder-GPU

# Remove empty JUCE directory
rmdir /s /q JUCE

# Clone JUCE manually
git clone https://github.com/juce-framework/JUCE.git
cd JUCE
git checkout 7.0.12

# Verify CMakeLists.txt exists
dir CMakeLists.txt
```

## Solution 3: Fix FFmpeg and WebP with Chocolatey

```cmd
# Install FFmpeg and WebP via Chocolatey
choco install ffmpeg --params "/InstallDir:C:\ffmpeg"
choco install libwebp

# Set environment variables
set FFMPEG_ROOT=C:\ffmpeg
set WEBP_ROOT=C:\ProgramData\chocolatey\lib\libwebp

# Alternative: Download manually
# FFmpeg: https://www.gyan.dev/ffmpeg/builds/
# WebP: https://developers.google.com/speed/webp/download
```

## Solution 4: Configure with Manual Paths

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
  -DFFMPEG_ROOT=C:\ffmpeg ^
  -DWEBP_ROOT=C:\libwebp ^
  -Wno-dev
```

## Solution 5: Use vcpkg for FFmpeg and WebP

```cmd
cd C:\vcpkg

# Install FFmpeg and WebP via vcpkg
.\vcpkg install ffmpeg:x64-windows
.\vcpkg install libwebp:x64-windows

# Then configure with vcpkg toolchain
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
  -DBOOST_ROOT=C:\local\boost_1_87_0 ^
  -Wno-dev
```

## Complete Step-by-Step Fix

```cmd
# Step 1: Fix JUCE submodule
cd I:\AudioGridder\AudioGridder-GPU
git submodule init
git submodule update --recursive

# Step 2: Install dependencies
choco install ffmpeg --params "/InstallDir:C:\ffmpeg"
# OR use vcpkg:
# cd C:\vcpkg && .\vcpkg install ffmpeg:x64-windows libwebp:x64-windows

# Step 3: Clean and reconfigure
rmdir /s /q build-windows-x64
mkdir build-windows-x64
cd build-windows-x64

# Step 4: Configure with all paths
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
  -DFFMPEG_ROOT=C:\ffmpeg ^
  -Wno-dev

# Step 5: Build
cmake --build . --config RelWithDebInfo --target AudioGridderServer
```

## Verification Commands

```cmd
# Check JUCE is properly initialized
dir I:\AudioGridder\AudioGridder-GPU\JUCE\CMakeLists.txt

# Check FFmpeg installation
ffmpeg -version

# Check if WebP tools are available
cwebp -version

# Verify CMake can find libraries
cmake .. -LAH | findstr -i "ffmpeg\|webp\|boost\|juce"
```

## Troubleshooting

### If git submodule fails:
```cmd
# Check .gitmodules file
type .gitmodules

# Force update
git submodule update --init --recursive --force
```

### If FFmpeg still not found:
```cmd
# Set explicit paths
set CMAKE_PREFIX_PATH=C:\ffmpeg;C:\libwebp;%CMAKE_PREFIX_PATH%
```

### If JUCE commands still unknown:
```cmd
# Verify JUCE CMakeLists.txt contains juce_add_gui_app
findstr "juce_add_gui_app" JUCE\CMakeLists.txt
```

The key issue is that JUCE is a git submodule that needs to be initialized. Start with `git submodule init && git submodule update --recursive`.
