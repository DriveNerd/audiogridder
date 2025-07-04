# AudioGridder GPU Server - Complete Installation Guide

## Overview

This guide provides step-by-step instructions to build and install the AudioGridder GPU Server with CUDA acceleration on Windows.

## What You Get

After following this guide, you will have:
- ✅ **AudioGridderServer.exe** - The main GPU-accelerated server executable
- ✅ **Installer package** - Professional Windows installer (.exe)
- ✅ **CUDA runtime libraries** - Bundled for GPU acceleration
- ✅ **Complete documentation** - Usage and troubleshooting guides

## Quick Installation (Recommended)

### Prerequisites Check

Before starting, ensure you have:
- Windows 10/11 (64-bit)
- NVIDIA GPU (optional, for GPU acceleration)
- Administrator privileges

### Step 1: Install Required Software

**Download and install in this order:**

1. **Visual Studio 2022 Community** (Free)
   - URL: https://visualstudio.microsoft.com/vs/community/
   - ✅ Select "Desktop development with C++" workload
   - ✅ Ensure "MSVC v143 - VS 2022 C++ x64/x86 build tools" is checked
   - ✅ Ensure "Windows 10/11 SDK" is checked

2. **CMake** (Latest version)
   - URL: https://cmake.org/download/
   - ✅ Choose "Add CMake to system PATH for all users"

3. **Git for Windows**
   - URL: https://git-scm.com/download/win
   - ✅ Use default settings

4. **CUDA Toolkit** (Optional, for GPU acceleration)
   - URL: https://developer.nvidia.com/cuda-downloads
   - ✅ Choose CUDA 11.0 or higher
   - ✅ Requires NVIDIA GPU with driver 450.80.02+

### Step 2: Install vcpkg (Recommended)

Open **Command Prompt as Administrator** and run:

```batch
cd C:git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.ootstrap-vcpkg.bat
.cpkg integrate install
```

Install dependencies:
```batch
.cpkg install ffmpeg[core]:x64-windows
.cpkg install webp:x64-windows  
.cpkg install boost:x64-windows
```

### Step 3: Build AudioGridder GPU Server

1. **Open Developer Command Prompt**
   - Start Menu → Visual Studio 2022 → Developer Command Prompt for VS 2022

2. **Clone and build**
   ```batch
   git clone https://github.com/your-repo/AudioGridder-GPU.git
   cd AudioGridder-GPU
   git checkout gpu-server-build
   BUILD_WINDOWS_GPU_SERVER.bat
   ```

3. **Wait for completion** (5-15 minutes depending on your system)

### Step 4: Find Your Files

After successful build:
- **Executable**: `install-windows-gpuin\AudioGridderServer.exe`
- **Installer**: `install-windows-gpu\AudioGridderServer-GPU-Setup.exe`

## Alternative Build Methods

### Method A: PowerShell Script (Advanced Users)

```powershell
# Run in PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\Build-WindowsGPUServer.ps1
```

Options:
```powershell
# Build without CUDA
.\Build-WindowsGPUServer.ps1 -SkipCUDA

# Build without installer
.\Build-WindowsGPUServer.ps1 -SkipInstaller

# Custom vcpkg path
.\Build-WindowsGPUServer.ps1 -VcpkgPath "D:cpkg"
```

### Method B: Manual CMake Build

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

cmake --build . --config RelWithDebInfo --target AudioGridderServer
```

## Installation for End Users

### Using the Installer (Recommended)

1. **Run the installer**
   ```
   AudioGridderServer-GPU-Setup.exe
   ```

2. **Follow the installation wizard**
   - Choose installation directory
   - Create desktop shortcut (optional)
   - Install Visual C++ Redistributable if prompted

3. **Launch the server**
   - Start Menu → AudioGridder GPU → AudioGridder GPU Server
   - Or run: `C:\Program Files\AudioGridderServer-GPU\AudioGridderServer.exe`

### Manual Installation

1. **Install Visual C++ Redistributable 2022 x64**
   - Download: https://aka.ms/vs/17/release/vc_redist.x64.exe

2. **Copy files to desired location**
   ```
   AudioGridderServer.exe
   *.dll (CUDA runtime libraries)
   ```

3. **Create shortcuts as needed**

## Testing Your Installation

### Basic Functionality Test

1. **Launch AudioGridder Server**
   ```batch
   AudioGridderServer.exe
   ```

2. **Check the console output for:**
   ```
   ✓ AudioGridder Server starting...
   ✓ CUDA initialized successfully (if GPU available)
   ✓ Server listening on port 55055
   ```

3. **Verify GPU acceleration (if available)**
   - Open server settings
   - Look for "GPU Acceleration: Enabled"
   - Check GPU device information

### Performance Test

1. **Load an audio plugin**
2. **Process audio and monitor:**
   - CPU usage in Task Manager
   - GPU usage in Task Manager (Performance tab)
   - Processing latency in AudioGridder

## Troubleshooting

### Build Issues

**❌ "Visual Studio not found"**
```
Solution: Use "Developer Command Prompt for VS 2022"
- Start Menu → Visual Studio 2022 → Developer Command Prompt
```

**❌ "CMake not found"**
```
Solution: Add CMake to PATH
- Reinstall CMake with "Add to PATH" option
- Or manually add C:\Program Files\CMakein to PATH
```

**❌ "CUDA not found"**
```
Solution: Install CUDA Toolkit
- Download from NVIDIA developer website
- Restart command prompt after installation
- Verify with: nvcc --version
```

**❌ "vcpkg dependencies not found"**
```
Solution: Install dependencies
cd C:cpkg
.cpkg install ffmpeg[core]:x64-windows webp:x64-windows boost:x64-windows
```

**❌ "Build failed with errors"**
```
Solution: Check build log
- Look for specific error messages
- Ensure all prerequisites are installed
- Try building individual components
```

### Runtime Issues

**❌ "Application failed to start"**
```
Solution: Install Visual C++ Redistributable
- Download vc_redist.x64.exe from Microsoft
- Restart after installation
```

**❌ "GPU acceleration not working"**
```
Solution: Check GPU requirements
- NVIDIA GPU with Compute Capability 3.5+
- Driver version 450.80.02 or higher
- Minimum 2GB GPU memory
```

**❌ "Audio processing errors"**
```
Solution: Check audio configuration
- Verify audio drivers are working
- Check sample rate compatibility
- Monitor system resources
```

**❌ "Server won't start"**
```
Solution: Check port availability
- Ensure port 55055 is not in use
- Check Windows Firewall settings
- Run as Administrator if needed
```

### Performance Issues

**⚠️ "Slow performance"**
```
Solutions:
- Enable GPU acceleration in settings
- Increase audio buffer size
- Close unnecessary applications
- Monitor GPU memory usage
```

**⚠️ "High CPU usage"**
```
Solutions:
- Verify GPU acceleration is working
- Reduce number of simultaneous plugins
- Check for CPU-intensive plugins
```

## Advanced Configuration

### Custom Build Options

Edit `CMakeLists.txt` for advanced options:

```cmake
# Custom CUDA architectures
set_property(TARGET AudioGridderServer PROPERTY CUDA_ARCHITECTURES 75 80 86)

# Enable debug symbols
set(CMAKE_BUILD_TYPE Debug)

# Custom optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
```

### GPU Memory Optimization

Adjust GPU buffer sizes in `GPUAudioBuffer.cpp`:

```cpp
// Increase buffer size for high sample rates
static constexpr size_t DEFAULT_BUFFER_SIZE = 8192;

// Adjust memory pool size
static constexpr size_t MEMORY_POOL_SIZE = 256 * 1024 * 1024; // 256MB
```

### Performance Monitoring

Enable detailed logging:

```cpp
// In main application
Logger::setLevel(Logger::DEBUG);
CUDAManager::getInstance().enableProfiling(true);
```

## Support and Updates

### Getting Help

1. **Check this guide first** - Most issues are covered here
2. **Review build logs** - Look for specific error messages  
3. **Verify prerequisites** - Ensure all software is correctly installed
4. **Test with minimal configuration** - Disable GPU to isolate issues

### Contributing

To contribute improvements:

1. **Fork the repository**
2. **Create a feature branch**
3. **Test thoroughly on Windows**
4. **Submit a pull request**

### Reporting Issues

When reporting problems, include:

- Windows version and architecture
- Visual Studio version
- CUDA version (if applicable)
- Complete build log
- Steps to reproduce

## License

This GPU-accelerated version maintains the same MIT license as the original AudioGridder project.

---

**Last Updated**: Auto-generated  
**Version**: 1.0.0-GPU  
**Platform**: Windows x64  
**CUDA Support**: Yes
