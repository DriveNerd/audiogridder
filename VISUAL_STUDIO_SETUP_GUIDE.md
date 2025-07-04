# Visual Studio Setup Guide for AudioGridder GPU Server

## 🚨 **SOLVING: "Visual Studio C++ compiler not found"**

This error occurs because the build script cannot find the Visual Studio C++ compiler in your current environment. Here are the solutions:

## 🎯 **IMMEDIATE SOLUTIONS**

### **Solution 1: Use the Setup Helper Script (Easiest)**

I've created a helper script that will automatically find and configure Visual Studio:

```batch
.\SETUP_VISUAL_STUDIO_ENV.bat
```

This script will:
- ✅ Automatically find your Visual Studio installation
- ✅ Set up the compiler environment
- ✅ Run the build process automatically

### **Solution 2: Use Developer Command Prompt (Recommended)**

1. **Open the correct command prompt:**
   - Press `Windows Key + R`
   - Type: `cmd`
   - Press Enter

2. **Navigate to Visual Studio Developer Command Prompt:**
   - Start Menu → **Visual Studio 2022** → **Developer Command Prompt for VS 2022**
   - OR search for "Developer Command Prompt" in Start Menu

3. **Navigate to your project:**
   ```batch
   cd /d "I:\AudioGridder\AudioGridder-GPU"
   ```

4. **Run the build:**
   ```batch
   BUILD_WINDOWS_GPU_SERVER.bat
   ```

### **Solution 3: PowerShell with Manual Environment Setup**

If you prefer PowerShell, you can set up the environment manually:

```powershell
# Find Visual Studio installation
$vsPath = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installerswhere.exe" -latest -property installationPath

# Set up environment
& "$vsPath\VC\Auxiliary\Buildcvars64.bat"

# Run build
.\BUILD_WINDOWS_GPU_SERVER.bat
```

## 🔧 **IF VISUAL STUDIO IS NOT INSTALLED**

### **Option A: Install Visual Studio 2022 Community (Free & Recommended)**

1. **Download Visual Studio 2022 Community:**
   - URL: https://visualstudio.microsoft.com/vs/community/
   - Click "Download Visual Studio" → "Community 2022"

2. **During Installation:**
   - ✅ Select **"Desktop development with C++"** workload
   - ✅ Ensure these components are checked:
     - MSVC v143 - VS 2022 C++ x64/x86 build tools (latest)
     - Windows 10/11 SDK (latest version)
     - CMake tools for Visual Studio
     - Git for Windows (if not already installed)

3. **After Installation:**
   - Restart your computer
   - Use "Developer Command Prompt for VS 2022" for building

### **Option B: Install Build Tools Only (Minimal)**

If you don't want the full Visual Studio IDE:

1. **Download Build Tools:**
   - URL: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
   - Click "Build Tools for Visual Studio 2022"

2. **During Installation:**
   - ✅ Select **"C++ build tools"** workload
   - ✅ Include:
     - MSVC v143 - VS 2022 C++ x64/x86 build tools
     - Windows 10/11 SDK

## 🔍 **VERIFICATION STEPS**

After installing Visual Studio, verify it's working:

### **Test 1: Check Compiler**
```batch
# In Developer Command Prompt
cl
```
Should show: "Microsoft (R) C/C++ Optimizing Compiler"

### **Test 2: Check CMake**
```batch
cmake --version
```
Should show CMake version 3.15 or higher

### **Test 3: Check CUDA (Optional)**
```batch
nvcc --version
```
Should show CUDA compiler version (if CUDA is installed)

## 🎯 **STEP-BY-STEP BUILD PROCESS**

Once Visual Studio is properly set up:

### **Method 1: Automatic (Recommended)**
```batch
.\SETUP_VISUAL_STUDIO_ENV.bat
```

### **Method 2: Manual**
1. **Open Developer Command Prompt for VS 2022**
2. **Navigate to project:**
   ```batch
   cd /d "I:\AudioGridder\AudioGridder-GPU"
   ```
3. **Run build:**
   ```batch
   BUILD_WINDOWS_GPU_SERVER.bat
   ```

### **Method 3: PowerShell with Parameters**
```powershell
# In regular PowerShell (after VS environment is set)
.\Build-WindowsGPUServer.ps1
```

## 🚀 **EXPECTED BUILD OUTPUT**

When successful, you'll see:
```
========================================
AudioGridder GPU Server Windows Builder
========================================

Checking prerequisites...
✓ Visual Studio found: C:\Program Files\Microsoft Visual Studio2\Community
✓ CMake found: cmake version 3.28.1
✓ CUDA found: 12.0 (if installed)

========================================
Starting Build Process
========================================

Step 1: Configuring CMake...
✓ CMake configuration successful

Step 2: Building AudioGridder Server...
✓ Build successful

Step 3: Installing to staging directory...
✓ Installation successful

Step 4: Copying additional files...
✓ Copied AudioGridderServer.exe
✓ CUDA runtime libraries copied

Step 5: Creating installer...
✓ Installer created successfully!

========================================
BUILD COMPLETED SUCCESSFULLY!
========================================

Executable location: install-windows-gpuin\AudioGridderServer.exe
Installer location: install-windows-gpu\AudioGridderServer-GPU-Setup.exe
```

## 🎁 **WHAT YOU'LL GET**

After successful build:
- **📁 install-windows-gpu/bin/AudioGridderServer.exe** - Your GPU-accelerated server
- **📦 install-windows-gpu/AudioGridderServer-GPU-Setup.exe** - Professional installer
- **📄 install-windows-gpu/VCREDIST_REQUIRED.txt** - Runtime requirements info

## ❓ **TROUBLESHOOTING COMMON ISSUES**

### **"CMake not found"**
- Install CMake from https://cmake.org/download/
- Add to PATH during installation

### **"CUDA not found" (Warning)**
- This is optional - build will continue without GPU acceleration
- Install CUDA Toolkit from NVIDIA if you want GPU features

### **"vcpkg dependencies not found"**
- Install vcpkg (optional but recommended):
  ```batch
  git clone https://github.com/Microsoft/vcpkg.git C:cpkg
  cd C:cpkg
  .ootstrap-vcpkg.bat
  .cpkg integrate install
  ```

### **Build fails with linker errors**
- Make sure you're using x64 architecture
- Verify all dependencies are x64 versions
- Try cleaning and rebuilding

## 📞 **STILL HAVING ISSUES?**

If you're still having problems:

1. **Try the automatic setup script:**
   ```batch
   .\SETUP_VISUAL_STUDIO_ENV.bat
   ```

2. **Check your Visual Studio installation:**
   - Open Visual Studio Installer
   - Modify your installation
   - Ensure "Desktop development with C++" is installed

3. **Use the minimal approach:**
   - Install only Build Tools for Visual Studio 2022
   - Use Developer Command Prompt

4. **Verify environment variables:**
   - VSINSTALLDIR should be set
   - PATH should include Visual Studio tools

---

**Remember:** The key is using the **Developer Command Prompt for VS 2022** or running the **SETUP_VISUAL_STUDIO_ENV.bat** script which automatically configures the environment for you.
