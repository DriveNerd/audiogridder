# AudioGridder GPU Server - Windows Build Guide

This guide provides complete instructions for building the AudioGridder GPU Server on Windows with CUDA acceleration support.

## Quick Start

**For users who just want the executable:**
1. Run `BUILD_WINDOWS_GPU_SERVER.bat` from a Visual Studio Developer Command Prompt
2. Find the installer at `install-windows-gpu/AudioGridderServer-GPU-Setup.exe`

## Prerequisites

### Required Software

1. **Visual Studio 2022 Community** (free)
   - Download: https://visualstudio.microsoft.com/vs/community/
   - Install with "Desktop development with C++" workload
   - Ensure MSVC v143 compiler toolset is selected

2. **CMake 3.15 or higher**
   - Download: https://cmake.org/download/
   - Add to PATH during installation

3. **Git for Windows**
   - Download: https://git-scm.com/download/win

### Optional (for GPU acceleration)

4. **CUDA Toolkit 11.0 or higher**
   - Download: https://developer.nvidia.com/cuda-downloads
   - Requires NVIDIA GPU with Compute Capability 3.5+
   - Minimum 2GB GPU memory recommended

5. **vcpkg Package Manager** (recommended for dependencies)
   ```batch
   git clone https://github.com/Microsoft/vcpkg.git C:\vcpkg
   cd C:\vcpkg
   .\bootstrap-vcpkg.bat
   .\vcpkg integrate install
   ```

### Dependencies via vcpkg (recommended)

```batch
cd C:\vcpkg
.\vcpkg install ffmpeg[core]:x64-windows
.\vcpkg install webp:x64-windows
.\vcpkg install boost:x64-windows
```

## Build Instructions

### Method 1: Automated Build (Recommended)

1. **Open Visual Studio Developer Command Prompt**
   - Start Menu → Visual Studio 2022 → Developer Command Prompt for VS 2022

2. **Clone and build**
   ```batch
   git clone <your-repo-url>
   cd AudioGridder-GPU
   git checkout gpu-server-build
   BUILD_WINDOWS_GPU_SERVER.bat
   ```

3. **Find your executable**
   - Executable: `install-windows-gpu\bin\AudioGridderServer.exe`
   - Installer: `install-windows-gpu\AudioGridderServer-GPU-Setup.exe`

### Method 2: Manual Build

1. **Configure**
   ```batch
   mkdir build-windows-gpu-x64
   cd build-windows-gpu-x64
   
   cmake .. ^
     -G "Visual Studio 17 2022" ^
     -A x64 ^
     -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
     -DAG_ENABLE_CUDA=ON ^
     -DAG_WITH_SERVER=ON ^
     -DAG_WITH_PLUGIN=OFF ^
     -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake ^
     -DVCPKG_TARGET_TRIPLET=x64-windows
   ```

2. **Build**
   ```batch
   cmake --build . --config RelWithDebInfo --target AudioGridderServer
   ```

## GPU Features

The AudioGridder GPU Server includes:

- **CUDA-accelerated audio processing** for supported operations
- **Automatic CPU fallback** when GPU is unavailable
- **Multi-GPU support** with device selection
- **Memory-efficient GPU buffer management**
- **Performance monitoring** and statistics

### Supported GPU Operations

- Gain processing and attenuation
- Audio buffer mixing
- Low-pass filtering
- Buffer clearing and initialization
- Parallel multi-channel processing

## Runtime Requirements

### For End Users

1. **Visual C++ Redistributable 2022 x64**
   - Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Required for all builds

2. **NVIDIA GPU Driver** (for GPU acceleration)
   - Version 450.80.02 or higher
   - Download from NVIDIA website

3. **CUDA Runtime** (automatically included in installer)
   - CUDA 11.0+ runtime libraries
   - Bundled with the application

## Testing the Build

### Basic Functionality Test

1. **Run the executable**
   ```batch
   install-windows-gpu\bin\AudioGridderServer.exe
   ```

2. **Check GPU status**
   - Look for "CUDA initialized" in the console output
   - GPU acceleration status shown in server settings

3. **Verify audio processing**
   - Load an audio plugin
   - Process audio to confirm functionality
   - Monitor GPU usage in Task Manager

### Performance Testing

1. **CPU vs GPU comparison**
   - Enable/disable GPU processing in settings
   - Compare processing times for identical operations
   - Monitor memory usage

2. **Stress testing**
   - Load multiple plugins simultaneously
   - Process high sample rate audio
   - Verify stable operation under load

## Troubleshooting

### Build Issues

**"CUDA not found"**
- Install CUDA Toolkit
- Verify CUDA_PATH environment variable
- Restart command prompt after CUDA installation

**"vcpkg dependencies not found"**
- Install dependencies: `vcpkg install ffmpeg webp boost:x64-windows`
- Verify vcpkg integration: `vcpkg integrate install`

**"Visual Studio not found"**
- Use "Developer Command Prompt for VS 2022"
- Verify C++ workload is installed
- Check MSVC v143 toolset is available

### Runtime Issues

**"Application failed to start"**
- Install Visual C++ Redistributable 2022 x64
- Check Windows version compatibility (Windows 10+)

**"GPU acceleration not working"**
- Update NVIDIA drivers
- Verify GPU compute capability (3.5+)
- Check available GPU memory (2GB+ recommended)

**"Audio processing errors"**
- Verify audio drivers are working
- Check sample rate compatibility
- Monitor system resources

## Advanced Configuration

### Custom CUDA Settings

Edit CMakeLists.txt to customize CUDA compilation:

```cmake
# Custom CUDA architectures
set_property(TARGET AudioGridderServer PROPERTY CUDA_ARCHITECTURES 75 80 86)

# CUDA optimization flags
set_property(TARGET AudioGridderServer PROPERTY CUDA_FLAGS "-O3 --use_fast_math")
```

### Performance Tuning

1. **GPU Memory Management**
   - Adjust buffer sizes in GPUAudioBuffer.cpp
   - Monitor memory usage with CUDA profiler

2. **Thread Configuration**
   - Optimize CUDA kernel launch parameters
   - Balance CPU/GPU workload distribution

## Support and Contributing

### Getting Help

1. Check the troubleshooting section above
2. Review build logs for specific error messages
3. Verify all prerequisites are correctly installed

### Contributing

1. Follow CUDA best practices for kernel development
2. Test on multiple GPU architectures
3. Maintain CPU compatibility with fallback paths
4. Document performance characteristics

## License

This GPU-accelerated version maintains the same MIT license as the original AudioGridder project.

---

**Build Date:** Generated automatically  
**CUDA Support:** Enabled  
**Target Platform:** Windows x64  
**Minimum Requirements:** Windows 10, Visual C++ 2022 Runtime
