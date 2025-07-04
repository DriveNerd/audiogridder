# AudioGridder Windows Build Fix Guide

## Issues Fixed

### 1. JUCE Submodule Missing
**Problem**: The JUCE directory was empty, causing CMake errors.
**Solution**: Initialize git submodules before building.

### 2. Required Dependencies Marked as REQUIRED but Disabled
**Problem**: CMakeLists.txt marked FFmpeg, WebP, and Boost as REQUIRED but the build script tried to disable them.
**Solution**: Modified CMakeLists.txt to make these dependencies optional for minimal builds.

### 3. Missing cmake_minimum_required in Standalone Build
**Problem**: The generated standalone CMakeLists.txt was missing the required cmake_minimum_required directive.
**Solution**: Fixed the BUILD_MINIMAL_NO_DEPS.bat script to include this at the top.

## Changes Made

### CMakeLists.txt
- Made FFmpeg, WebP, and Boost optional dependencies
- Added conditional compilation flags (AG_FFMPEG, AG_WEBP)
- Added proper error handling for missing dependencies
- Fixed include and link directories to be conditional

### Server/CMakeLists.txt
- Made library linking conditional based on availability
- Added compile definitions for optional features

### BUILD_MINIMAL_NO_DEPS.bat
- Added JUCE submodule initialization step
- Fixed standalone CMakeLists.txt generation
- Improved error handling

## Build Instructions

1. **Initialize Submodules** (if not done):
   ```cmd
   git submodule update --init --recursive
   ```

2. **Run the Fixed Build Script**:
   ```cmd
   BUILD_MINIMAL_NO_DEPS.bat
   ```

3. **Alternative Manual Build**:
   ```cmd
   mkdir build-minimal
   cd build-minimal
   cmake .. -G "Visual Studio 17 2022" -A x64 ^
       -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
       -DAG_ENABLE_CUDA=OFF ^
       -DAG_WITH_SERVER=ON ^
       -DAG_WITH_PLUGIN=OFF ^
       -DAG_WITH_TESTS=OFF ^
       -DAG_ENABLE_CODE_SIGNING=OFF ^
       -DAG_WITH_TRACEREADER=OFF ^
       -DCMAKE_CXX_STANDARD=17 ^
       -DCMAKE_CXX_STANDARD_REQUIRED=ON ^
       -DCMAKE_DISABLE_FIND_PACKAGE_Boost=ON ^
       -DCMAKE_DISABLE_FIND_PACKAGE_FFMPEG=ON ^
       -DCMAKE_DISABLE_FIND_PACKAGE_WebP=ON
   cmake --build . --config RelWithDebInfo --target AudioGridderServer
   ```

## Features Available in Minimal Build

✅ **Working Features**:
- Core AudioGridder server functionality
- JUCE audio processing
- Basic plugin hosting (VST3, LV2)
- Network communication
- Basic GPU acceleration (if CUDA available)

❌ **Disabled Features**:
- FFmpeg audio format support
- WebP image support
- Boost-dependent features
- Advanced tracing (requires Boost)

## Troubleshooting

### If CMake Configuration Still Fails:
1. Ensure Visual Studio 2022 is installed with C++ workload
2. Verify git submodules are initialized: `git submodule status`
3. Check that JUCE/CMakeLists.txt exists
4. Try cleaning and rebuilding: delete build directory and retry

### If Build Fails:
1. Check for missing Windows SDK components
2. Ensure C++ CMake tools are installed in Visual Studio
3. Try building individual targets: `cmake --build . --target AudioGridderServer`

### If Runtime Issues:
1. The minimal build may have reduced functionality
2. Some audio formats may not be supported without FFmpeg
3. Consider installing full dependencies for complete functionality

## Next Steps

For full functionality, consider installing:
- vcpkg for dependency management
- FFmpeg for audio format support
- WebP for image support
- Boost for advanced features

See COMPLETE_BUILD_GUIDE.md for full installation instructions.
