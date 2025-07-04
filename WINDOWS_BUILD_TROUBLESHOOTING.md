# Windows Build Troubleshooting Guide

## Current Issue: boost-thread Build Failure

### Error Description
```
error: building boost-thread:x64-windows-static failed with: BUILD_FAILED
```

### Root Cause Analysis
The project only requires `boost-program-options` but vcpkg is trying to build the entire boost package including boost-thread which has known compilation issues on Windows.

## Solution 1: Install Specific Boost Components Only

```cmd
cd C:\vcpkg

# Remove problematic full boost package
.\vcpkg remove boost:x64-windows-static

# Install only required components
.\vcpkg install boost-program-options:x64-windows-static
.\vcpkg install libwebp:x64-windows-static
.\vcpkg install ffmpeg:x64-windows-static

# Verify installation
.\vcpkg list | findstr boost
```

## Solution 2: Modified vcpkg.json

Replace the current vcpkg.json with:

```json
{
  "name" : "audiogridder",
  "version-string" : "1.0.0",
  "builtin-baseline" : "64ca152891d6ab135c6c27881e7eb0ac2fa15bba",
  "dependencies" : [
    {
      "name" : "boost-program-options",
      "version>=" : "1.83.0"
    },
    {
      "name" : "libwebp",
      "version>=" : "1.3.2"
    },
    "ffmpeg"
  ],
  "overrides": [
    {
      "name": "ffmpeg",
      "version": "6.0"
    }
  ]
}
```

## Solution 3: Build Without vcpkg

```cmd
cd I:\AudioGridder\AudioGridder-GPU
rmdir /s /q build-windows-x64
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
  -DAG_WITH_TRACEREADER=OFF
```

## Solution 4: Use Pre-built Libraries

### Download and Install:
1. **Boost**: https://sourceforge.net/projects/boost/files/boost-binaries/
2. **FFmpeg**: https://www.gyan.dev/ffmpeg/builds/
3. **WebP**: https://developers.google.com/speed/webp/download

### Configure with manual paths:
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
  -DBOOST_ROOT=C:\boost_1_82_0 ^
  -DFFMPEG_ROOT=C:\ffmpeg ^
  -DWEBP_ROOT=C:\libwebp
```

## Solution 5: Disable Problematic Features

```cmd
cmake .. ^
  -G "Visual Studio 17 2022" ^
  -A x64 ^
  -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
  -DAG_ENABLE_CUDA=ON ^
  -DAG_WITH_SERVER=ON ^
  -DAG_WITH_PLUGIN=OFF ^
  -DAG_WITH_TESTS=OFF ^
  -DAG_WITH_TRACEREADER=OFF ^
  -DAG_ENABLE_CODE_SIGNING=OFF ^
  -DAG_ENABLE_SENTRY=OFF
```

## Verification Steps

After successful configuration:

```cmd
# Build the server
cmake --build . --config RelWithDebInfo --target AudioGridderServer

# Verify GPU components are built
dir bin\RelWithDebInfo\AudioGridderServer.exe

# Test CUDA functionality
bin\RelWithDebInfo\AudioGridderServer.exe --help
```

## Common Issues and Fixes

### Issue: CUDA not found
```cmd
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
```

### Issue: Missing DLLs
```cmd
# Copy required DLLs to output directory
copy "%CUDA_PATH%\bin\cudart64_*.dll" bin\RelWithDebInfo\
```

### Issue: vcpkg baseline too old
```cmd
cd C:\vcpkg
git pull
.\vcpkg update
```

## Success Indicators

✅ CMake configuration completes without errors
✅ AudioGridderServer.exe builds successfully  
✅ CUDA kernels compile (look for .cu.obj files)
✅ Server starts and detects GPU devices

## Next Steps After Successful Build

1. Test GPU detection: `AudioGridderServer.exe --list-gpu-devices`
2. Run GPU benchmark: `AudioGridderServer.exe --gpu-benchmark`
3. Start server: `AudioGridderServer.exe`
4. Check logs for CUDA initialization messages

## Contact Information

If issues persist, check:
- NVIDIA CUDA documentation
- vcpkg GitHub issues
- AudioGridder project documentation
