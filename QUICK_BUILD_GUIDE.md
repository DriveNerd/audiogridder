# AudioGridder GPU Server - Quick Build Guide

## Problem: Build Failing?

If you're getting build errors, this guide will help you get AudioGridder working quickly.

## Option 1: Recommended - Clone with Git (Best)

If you haven't already, clone the repository properly:

```cmd
git clone --recursive https://github.com/DriveNerd/AudioGridder-GPU.git
cd AudioGridder-GPU
BUILD_MINIMAL_NO_DEPS.bat
```

This ensures all submodules (including JUCE) are downloaded correctly.

## Option 2: Downloaded ZIP? Fix Missing JUCE

If you downloaded the code as a ZIP file, JUCE is missing. Choose one:

### A. Download JUCE Automatically
```cmd
DOWNLOAD_JUCE.bat
```
Follow the instructions, then run:
```cmd
BUILD_MINIMAL_NO_DEPS.bat
```

### B. Manual JUCE Download
1. Go to: https://github.com/juce-framework/JUCE/archive/refs/heads/master.zip
2. Download and extract
3. Rename folder from "JUCE-master" to "JUCE"
4. Place in your AudioGridder directory
5. Run `BUILD_MINIMAL_NO_DEPS.bat`

## Option 3: Minimal Standalone (No JUCE)

If JUCE won't work, the build script will automatically create a minimal standalone server:

```cmd
BUILD_MINIMAL_NO_DEPS.bat
```

This creates a basic server at: `standalone-build\RelWithDebInfo\AudioGridderServer.exe`

## What You Get

### Full Build (with JUCE)
✅ Complete AudioGridder functionality  
✅ Plugin hosting (VST3, LV2)  
✅ Audio processing  
✅ Network communication  
✅ GPU acceleration (if CUDA available)  

### Minimal Build (standalone)
✅ Basic server functionality  
✅ Network communication  
✅ Testing and development  
❌ No plugin hosting  
❌ No advanced audio processing  

## Troubleshooting

### "Not a git repository" Error
- You downloaded ZIP instead of cloning
- Use Option 2 above

### "JUCE not found" Error
- Run `DOWNLOAD_JUCE.bat`
- Or follow Option 2B above

### "CMake configuration failed"
- Make sure Visual Studio 2022 is installed
- Install "Desktop development with C++" workload
- The script will automatically try standalone build

### "Build failed"
- Check you have Windows SDK installed
- Try running from "Developer Command Prompt for VS 2022"
- The minimal standalone should still work

## File Locations

After successful build, find your executable at:

**Full build:**
- `build-minimal-no-deps\Server\RelWithDebInfo\AudioGridderServer.exe`
- Or: `install-minimalin\AudioGridderServer.exe`

**Standalone build:**
- `standalone-build\RelWithDebInfo\AudioGridderServer.exe`

## Next Steps

1. Run the executable to start the server
2. Configure your DAW to connect to the server
3. For full features, consider installing complete dependencies (see COMPLETE_BUILD_GUIDE.md)

## Need Help?

- Check WINDOWS_BUILD_FIX_GUIDE.md for detailed fixes
- See COMPLETE_BUILD_GUIDE.md for full installation
- Check the project issues on GitHub
