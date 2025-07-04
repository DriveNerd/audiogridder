@echo off
setlocal enabledelayedexpansion

echo ========================================
echo AudioGridder GPU Server Windows Builder
echo ========================================
echo.

:: Set build configuration
set BUILD_TYPE=RelWithDebInfo
set BUILD_DIR=build-windows-gpu-x64
set INSTALL_DIR=install-windows-gpu

:: Check for required tools
echo Checking prerequisites...

:: Check Visual Studio
where cl >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Visual Studio C++ compiler not found!
    echo Please run this from a Visual Studio Developer Command Prompt
    echo or install Visual Studio 2022 with C++ workload
    pause
    exit /b 1
)

:: Check CMake
where cmake >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: CMake not found! Please install CMake 3.15 or higher
    pause
    exit /b 1
)

:: Check CUDA
where nvcc >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: CUDA compiler not found!
    echo GPU acceleration will be disabled
    set CUDA_ENABLED=OFF
) else (
    echo CUDA found: 
    nvcc --version | findstr "release"
    set CUDA_ENABLED=ON
)

:: Check vcpkg (optional but recommended)
if exist "C:\vcpkg\vcpkg.exe" (
    echo vcpkg found at C:\vcpkg
    set VCPKG_TOOLCHAIN=-DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows
) else (
    echo vcpkg not found - will try system dependencies
    set VCPKG_TOOLCHAIN=
)

echo.
echo ========================================
echo Starting Build Process
echo ========================================

:: Clean previous build
if exist "%BUILD_DIR%" (
    echo Cleaning previous build...
    rmdir /s /q "%BUILD_DIR%"
)
mkdir "%BUILD_DIR%"

:: Clean previous install
if exist "%INSTALL_DIR%" (
    rmdir /s /q "%INSTALL_DIR%"
)
mkdir "%INSTALL_DIR%"

cd "%BUILD_DIR%"

echo.
echo Step 1: Configuring CMake...
cmake .. ^
  -G "Visual Studio 17 2022" ^
  -A x64 ^
  -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
  -DAG_ENABLE_CUDA=%CUDA_ENABLED% ^
  -DAG_WITH_SERVER=ON ^
  -DAG_WITH_PLUGIN=OFF ^
  -DAG_WITH_TESTS=OFF ^
  -DAG_ENABLE_CODE_SIGNING=OFF ^
  -DAG_WITH_TRACEREADER=OFF ^
  -DCMAKE_INSTALL_PREFIX=../%INSTALL_DIR% ^
  %VCPKG_TOOLCHAIN% ^
  -DCMAKE_CXX_STANDARD=17 ^
  -DCMAKE_CXX_STANDARD_REQUIRED=ON ^
  -Wno-dev

if %errorlevel% neq 0 (
    echo.
    echo ERROR: CMake configuration failed!
    cd ..
    pause
    exit /b 1
)

echo.
echo Step 2: Building AudioGridder Server...
cmake --build . --config %BUILD_TYPE% --target AudioGridderServer --parallel

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Build failed!
    cd ..
    pause
    exit /b 1
)

echo.
echo Step 3: Installing to staging directory...
cmake --install . --config %BUILD_TYPE%

cd ..

echo.
echo Step 4: Copying additional files...
if not exist "%INSTALL_DIR%\bin" mkdir "%INSTALL_DIR%\bin"

:: Copy main executable
copy "%BUILD_DIR%\bin\%BUILD_TYPE%\AudioGridderServer.exe" "%INSTALL_DIR%\bin\" >nul

:: Copy CUDA runtime if available
if "%CUDA_ENABLED%"=="ON" (
    echo Copying CUDA runtime libraries...
    for %%f in (cudart64_*.dll cublas64_*.dll cublasLt64_*.dll cufft64_*.dll) do (
        if exist "%CUDA_PATH%\bin\%%f" (
            copy "%CUDA_PATH%\bin\%%f" "%INSTALL_DIR%\bin\" >nul 2>&1
        )
    )
)

:: Copy Visual C++ redistributables info
echo Creating redistributable info...
echo Visual C++ Redistributable for Visual Studio 2022 x64 required > "%INSTALL_DIR%\VCREDIST_REQUIRED.txt"
echo Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe >> "%INSTALL_DIR%\VCREDIST_REQUIRED.txt"

echo.
echo Step 5: Creating installer...
call :CreateInstaller

echo.
echo ========================================
echo BUILD COMPLETED SUCCESSFULLY!
echo ========================================
echo.
echo Executable location: %INSTALL_DIR%\bin\AudioGridderServer.exe
echo Installer location: %INSTALL_DIR%\AudioGridderServer-GPU-Setup.exe
echo.
echo To test the server:
echo 1. Install Visual C++ Redistributable (see VCREDIST_REQUIRED.txt)
echo 2. If using GPU features, ensure NVIDIA drivers are installed
echo 3. Run: %INSTALL_DIR%\bin\AudioGridderServer.exe
echo.
pause
exit /b 0

:CreateInstaller
echo Creating Inno Setup installer script...

:: Create temporary installer script
set ISS_FILE=%INSTALL_DIR%\AudioGridderServer-GPU.iss

echo [Setup] > "%ISS_FILE%"
echo AppName=AudioGridder GPU Server >> "%ISS_FILE%"
echo AppVersion=1.0.0-GPU >> "%ISS_FILE%"
echo AppPublisher=AudioGridder GPU Build >> "%ISS_FILE%"
echo AppPublisherURL=https://audiogridder.com >> "%ISS_FILE%"
echo DefaultDirName={commonpf64}\AudioGridderServer-GPU >> "%ISS_FILE%"
echo DefaultGroupName=AudioGridder GPU >> "%ISS_FILE%"
echo OutputBaseFilename=AudioGridderServer-GPU-Setup >> "%ISS_FILE%"
echo OutputDir=. >> "%ISS_FILE%"
echo Compression=lzma >> "%ISS_FILE%"
echo SolidCompression=yes >> "%ISS_FILE%"
echo ArchitecturesAllowed=x64 >> "%ISS_FILE%"
echo ArchitecturesInstallIn64BitMode=x64 >> "%ISS_FILE%"
echo. >> "%ISS_FILE%"
echo [Languages] >> "%ISS_FILE%"
echo Name: "english"; MessagesFile: "compiler:Default.isl" >> "%ISS_FILE%"
echo. >> "%ISS_FILE%"
echo [Files] >> "%ISS_FILE%"
echo Source: "bin\AudioGridderServer.exe"; DestDir: "{app}"; Flags: ignoreversion >> "%ISS_FILE%"
echo Source: "bin\*.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist >> "%ISS_FILE%"
echo Source: "VCREDIST_REQUIRED.txt"; DestDir: "{app}"; Flags: ignoreversion >> "%ISS_FILE%"
echo. >> "%ISS_FILE%"
echo [Icons] >> "%ISS_FILE%"
echo Name: "{autoprograms}\AudioGridder GPU Server"; Filename: "{app}\AudioGridderServer.exe" >> "%ISS_FILE%"
echo Name: "{autodesktop}\AudioGridder GPU Server"; Filename: "{app}\AudioGridderServer.exe"; Tasks: desktopicon >> "%ISS_FILE%"
echo. >> "%ISS_FILE%"
echo [Tasks] >> "%ISS_FILE%"
echo Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked >> "%ISS_FILE%"
echo. >> "%ISS_FILE%"
echo [Run] >> "%ISS_FILE%"
echo Filename: "{app}\AudioGridderServer.exe"; Description: "{cm:LaunchProgram,AudioGridder GPU Server}"; Flags: nowait postinstall skipifsilent >> "%ISS_FILE%"

:: Try to create installer with Inno Setup
where iscc >nul 2>&1
if %errorlevel% equ 0 (
    echo Running Inno Setup compiler...
    cd "%INSTALL_DIR%"
    iscc "AudioGridderServer-GPU.iss"
    if %errorlevel% equ 0 (
        echo Installer created successfully!
    ) else (
        echo Warning: Installer creation failed, but executable is ready
    )
    cd ..
) else (
    echo Inno Setup not found - skipping installer creation
    echo You can manually install Inno Setup and run: iscc "%ISS_FILE%"
)

goto :eof
