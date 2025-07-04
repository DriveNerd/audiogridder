# C++ Standard Library Fix for Windows Build

## Current Issue: chrono_literals Error

### Error Description
```
error C2039: "chrono_literals" ist kein Member von "std"
error C3688: ungültiges Literalsuffix "ms"
```

### Root Cause
The code is using C++14 chrono literals but the compiler is not using the correct C++ standard version.

## Immediate Solution (Try This First)

**Reconfigure CMake with C++17 standard:**

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
  -DCMAKE_CXX_STANDARD=17 ^
  -DCMAKE_CXX_STANDARD_REQUIRED=ON ^
  -Wno-dev

cmake --build . --config RelWithDebInfo --target AudioGridderServer
```

## Alternative Solution: Clean Rebuild

If the above doesn't work, try a complete clean rebuild:

```cmd
cd I:\AudioGridder\AudioGridder-GPU

# Clean everything
rmdir /s /q build-windows-x64
mkdir build-windows-x64
cd build-windows-x64

# Configure with explicit C++ standard
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
  -DCMAKE_CXX_STANDARD=17 ^
  -DCMAKE_CXX_STANDARD_REQUIRED=ON ^
  -DCMAKE_CXX_EXTENSIONS=OFF ^
  -Wno-dev

# Build again
cmake --build . --config RelWithDebInfo --target AudioGridderServer
```

## If C++17 Doesn't Work: Try C++20

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
  -DFFMPEG_ROOT=C:\ffmpeg ^
  -DCMAKE_CXX_STANDARD=20 ^
  -DCMAKE_CXX_STANDARD_REQUIRED=ON ^
  -Wno-dev
```

## Manual Code Fix (Last Resort)

If CMake configuration doesn't fix it, edit the source code:

### Edit Common/Source/Utils.hpp

Replace line 21:
```cpp
using namespace std::chrono_literals;
```

With:
```cpp
// using namespace std::chrono_literals;
```

Replace line 386:
```cpp
cv.wait_for(lock, 500ms, [&] { return done; });
```

With:
```cpp
cv.wait_for(lock, std::chrono::milliseconds(500), [&] { return done; });
```

## Verification

After successful build, you should see:
```
AudioGridderServer.vcxproj -> I:\AudioGridder\AudioGridder-GPU\build-windows-x64\bin\RelWithDebInfo\AudioGridderServer.exe
```

The chrono_literals error should be resolved by ensuring the compiler uses C++17 or higher standard.
