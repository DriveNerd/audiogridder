# AudioGridder GPU Server - Windows Build Guide

## Quick Start

**Run this automated build script:**
```cmd
cd I:\AudioGridder\AudioGridder-GPU
BUILD_VERIFICATION.bat
```

This script will automatically:
1. Initialize JUCE submodule
2. Check and install dependencies
3. Configure CMake with correct paths
4. Build the GPU server executable
5. Test the built executable

## Manual Build Instructions

### Prerequisites
- Visual Studio 2019/2022 with C++ tools
- CUDA Toolkit 11.8+
- Git for Windows
- Chocolatey (optional, for easy dependency installation)

### Step-by-Step Build Process

#### 1. Initialize JUCE Submodule (CRITICAL)
```cmd
cd I:\AudioGridder\AudioGridder-GPU
git submodule init
git submodule update --recursive
```

#### 2. Install Dependencies

**Option A: Using Chocolatey**
```cmd
choco install boost-msvc-14.3
choco install ffmpeg --params "/InstallDir:C:\ffmpeg"
```

**Option B: Manual Installation**
- Boost: Download from https://sourceforge.net/projects/boost/files/boost-binaries/
- FFmpeg: Download from https://www.gyan.dev/ffmpeg/builds/

#### 3. Configure and Build
```cmd
mkdir build-windows-x64
cd build-windows-x64

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

cmake --build . --config RelWithDebInfo --target AudioGridderServer
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: "JUCE CMakeLists.txt not found"
**Solution:** Initialize git submodules
```cmd
git submodule update --init --recursive --force
```

#### Issue: "boost-thread build failed"
**Solution:** Use specific boost components only
```cmd
cd C:\vcpkg
.\vcpkg remove boost:x64-windows-static
.\vcpkg install boost-program-options:x64-windows-static
```

#### Issue: "Could not find FFmpeg"
**Solution:** Set explicit FFmpeg path
```cmd
set FFMPEG_ROOT=C:\ffmpeg
# Or install via: choco install ffmpeg
```

#### Issue: "Unknown CMake command juce_add_gui_app"
**Solution:** This indicates JUCE submodule is not properly initialized
```cmd
dir JUCE\CMakeLists.txt  # Should exist
git submodule status      # Check submodule status
```

## Build Verification

After successful build, verify with:
```cmd
cd bin\RelWithDebInfo

# Test basic functionality
AudioGridderServer.exe --help

# Test GPU detection
AudioGridderServer.exe --list-gpu-devices

# Test GPU benchmark
AudioGridderServer.exe --gpu-benchmark
```

## GPU Features

The built server includes:
- ✅ CUDA-accelerated audio processing
- ✅ GPU memory management with automatic fallback
- ✅ Multi-GPU support and device selection
- ✅ Performance monitoring and statistics
- ✅ Automatic CPU fallback when GPU unavailable

## File Structure After Build

```
build-windows-x64/
├── bin/
│   └── RelWithDebInfo/
│       ├── AudioGridderServer.exe    # Main GPU server executable
│       └── *.dll                     # Required runtime libraries
├── lib/                              # Static libraries
└── Server/                           # Build artifacts
```

## Running the Server

```cmd
cd bin\RelWithDebInfo
AudioGridderServer.exe

# Server will start and listen on default port
# Check logs at: %APPDATA%\AudioGridder\server.log
```

## Additional Documentation

- `WINDOWS_BUILD_TROUBLESHOOTING.md` - Detailed troubleshooting guide
- `JUCE_SETUP_GUIDE.md` - JUCE framework setup instructions
- `BUILD_VERIFICATION.bat` - Automated build script

## Success Indicators

✅ CMake configuration completes without errors
✅ AudioGridderServer.exe builds successfully  
✅ CUDA kernels compile (look for .cu.obj files)
✅ Server starts and detects GPU devices
✅ GPU benchmark runs without errors

## Support

If you encounter issues:
1. Run `BUILD_VERIFICATION.bat` for automated diagnosis
2. Check the troubleshooting guides in this repository
3. Verify all prerequisites are installed correctly
4. Ensure NVIDIA GPU drivers are up to date
