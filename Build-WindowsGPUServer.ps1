# AudioGridder GPU Server Windows Builder (PowerShell)
# This script builds the AudioGridder GPU Server with CUDA support on Windows

param(
    [switch]$SkipCUDA = $false,
    [switch]$SkipInstaller = $false,
    [string]$BuildType = "RelWithDebInfo",
    [string]$VcpkgPath = "C:\vcpkg"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "AudioGridder GPU Server Windows Builder" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$BuildDir = "build-windows-gpu-x64"
$InstallDir = "install-windows-gpu"
$CudaEnabled = $true

# Check prerequisites
Write-Host "Checking prerequisites..." -ForegroundColor Yellow

# Check Visual Studio
try {
    $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vsWhere) {
        $vsInstall = & $vsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($vsInstall) {
            Write-Host "✓ Visual Studio found: $vsInstall" -ForegroundColor Green
        } else {
            throw "Visual Studio with C++ tools not found"
        }
    } else {
        throw "Visual Studio installer not found"
    }
} catch {
    Write-Host "✗ ERROR: Visual Studio C++ tools not found!" -ForegroundColor Red
    Write-Host "Please install Visual Studio 2022 with C++ workload" -ForegroundColor Red
    exit 1
}

# Check CMake
try {
    $cmakeVersion = cmake --version 2>$null
    if ($cmakeVersion) {
        Write-Host "✓ CMake found: $($cmakeVersion[0])" -ForegroundColor Green
    } else {
        throw "CMake not found"
    }
} catch {
    Write-Host "✗ ERROR: CMake not found!" -ForegroundColor Red
    Write-Host "Please install CMake 3.15 or higher" -ForegroundColor Red
    exit 1
}

# Check CUDA
if (-not $SkipCUDA) {
    try {
        $nvccVersion = nvcc --version 2>$null
        if ($nvccVersion) {
            $cudaVersion = ($nvccVersion | Select-String "release").ToString().Split(" ")[-1]
            Write-Host "✓ CUDA found: $cudaVersion" -ForegroundColor Green
            $CudaEnabled = $true
        } else {
            throw "CUDA not found"
        }
    } catch {
        Write-Host "⚠ WARNING: CUDA not found - GPU acceleration will be disabled" -ForegroundColor Yellow
        $CudaEnabled = $false
    }
} else {
    Write-Host "⚠ CUDA support skipped by user request" -ForegroundColor Yellow
    $CudaEnabled = $false
}

# Check vcpkg
$VcpkgToolchain = ""
if (Test-Path "$VcpkgPath\vcpkg.exe") {
    Write-Host "✓ vcpkg found at $VcpkgPath" -ForegroundColor Green
    $VcpkgToolchain = "-DCMAKE_TOOLCHAIN_FILE=$VcpkgPath/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows"
} else {
    Write-Host "⚠ vcpkg not found - will try system dependencies" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Build Process" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Clean previous builds
if (Test-Path $BuildDir) {
    Write-Host "Cleaning previous build..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $BuildDir
}
New-Item -ItemType Directory -Path $BuildDir | Out-Null

if (Test-Path $InstallDir) {
    Remove-Item -Recurse -Force $InstallDir
}
New-Item -ItemType Directory -Path $InstallDir | Out-Null

Set-Location $BuildDir

# Configure CMake
Write-Host ""
Write-Host "Step 1: Configuring CMake..." -ForegroundColor Yellow

$cmakeArgs = @(
    ".."
    "-G", "Visual Studio 17 2022"
    "-A", "x64"
    "-DCMAKE_BUILD_TYPE=$BuildType"
    "-DAG_ENABLE_CUDA=$($CudaEnabled.ToString().ToUpper())"
    "-DAG_WITH_SERVER=ON"
    "-DAG_WITH_PLUGIN=OFF"
    "-DAG_WITH_TESTS=OFF"
    "-DAG_ENABLE_CODE_SIGNING=OFF"
    "-DAG_WITH_TRACEREADER=OFF"
    "-DCMAKE_INSTALL_PREFIX=../$InstallDir"
    "-DCMAKE_CXX_STANDARD=17"
    "-DCMAKE_CXX_STANDARD_REQUIRED=ON"
    "-Wno-dev"
)

if ($VcpkgToolchain) {
    $cmakeArgs += $VcpkgToolchain.Split(" ")
}

try {
    & cmake $cmakeArgs
    if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed" }
    Write-Host "✓ CMake configuration successful" -ForegroundColor Green
} catch {
    Write-Host "✗ ERROR: CMake configuration failed!" -ForegroundColor Red
    Set-Location ..
    exit 1
}

# Build
Write-Host ""
Write-Host "Step 2: Building AudioGridder Server..." -ForegroundColor Yellow

try {
    cmake --build . --config $BuildType --target AudioGridderServer --parallel
    if ($LASTEXITCODE -ne 0) { throw "Build failed" }
    Write-Host "✓ Build successful" -ForegroundColor Green
} catch {
    Write-Host "✗ ERROR: Build failed!" -ForegroundColor Red
    Set-Location ..
    exit 1
}

# Install
Write-Host ""
Write-Host "Step 3: Installing to staging directory..." -ForegroundColor Yellow

try {
    cmake --install . --config $BuildType
    Write-Host "✓ Installation successful" -ForegroundColor Green
} catch {
    Write-Host "⚠ WARNING: Installation step failed, copying manually..." -ForegroundColor Yellow
}

Set-Location ..

# Copy additional files
Write-Host ""
Write-Host "Step 4: Copying additional files..." -ForegroundColor Yellow

$binDir = "$InstallDir\bin"
if (-not (Test-Path $binDir)) {
    New-Item -ItemType Directory -Path $binDir | Out-Null
}

# Copy main executable
$exePath = "$BuildDir\bin\$BuildType\AudioGridderServer.exe"
if (Test-Path $exePath) {
    Copy-Item $exePath "$binDir\" -Force
    Write-Host "✓ Copied AudioGridderServer.exe" -ForegroundColor Green
} else {
    Write-Host "✗ ERROR: AudioGridderServer.exe not found!" -ForegroundColor Red
    exit 1
}

# Copy CUDA runtime if available
if ($CudaEnabled -and $env:CUDA_PATH) {
    Write-Host "Copying CUDA runtime libraries..." -ForegroundColor Yellow
    $cudaBinPath = "$env:CUDA_PATH\bin"
    $cudaLibs = @("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll", "cufft64_*.dll")
    
    foreach ($lib in $cudaLibs) {
        $files = Get-ChildItem -Path $cudaBinPath -Name $lib -ErrorAction SilentlyContinue
        foreach ($file in $files) {
            Copy-Item "$cudaBinPath\$file" "$binDir\" -Force -ErrorAction SilentlyContinue
        }
    }
    Write-Host "✓ CUDA runtime libraries copied" -ForegroundColor Green
}

# Create redistributable info
$vcredistInfo = @"
Visual C++ Redistributable for Visual Studio 2022 x64 required
Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

This file contains information about required runtime dependencies.
Install the Visual C++ Redistributable before running AudioGridderServer.exe
"@

$vcredistInfo | Out-File -FilePath "$InstallDir\VCREDIST_REQUIRED.txt" -Encoding UTF8
Write-Host "✓ Created redistributable info" -ForegroundColor Green

# Create installer
if (-not $SkipInstaller) {
    Write-Host ""
    Write-Host "Step 5: Creating installer..." -ForegroundColor Yellow
    
    $issFile = "$InstallDir\AudioGridderServer-GPU.iss"
    $issContent = @"
[Setup]
AppName=AudioGridder GPU Server
AppVersion=1.0.0-GPU
AppPublisher=AudioGridder GPU Build
AppPublisherURL=https://audiogridder.com
DefaultDirName={commonpf64}\AudioGridderServer-GPU
DefaultGroupName=AudioGridder GPU
OutputBaseFilename=AudioGridderServer-GPU-Setup
OutputDir=.
Compression=lzma
SolidCompression=yes
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "bin\AudioGridderServer.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "bin\*.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist
Source: "VCREDIST_REQUIRED.txt"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\AudioGridder GPU Server"; Filename: "{app}\AudioGridderServer.exe"
Name: "{autodesktop}\AudioGridder GPU Server"; Filename: "{app}\AudioGridderServer.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Run]
Filename: "{app}\AudioGridderServer.exe"; Description: "{cm:LaunchProgram,AudioGridder GPU Server}"; Flags: nowait postinstall skipifsilent
"@

    $issContent | Out-File -FilePath $issFile -Encoding UTF8
    
    # Try to create installer with Inno Setup
    $isccPath = Get-Command iscc -ErrorAction SilentlyContinue
    if ($isccPath) {
        try {
            Set-Location $InstallDir
            & iscc "AudioGridderServer-GPU.iss"
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✓ Installer created successfully!" -ForegroundColor Green
            } else {
                Write-Host "⚠ WARNING: Installer creation failed, but executable is ready" -ForegroundColor Yellow
            }
            Set-Location ..
        } catch {
            Write-Host "⚠ WARNING: Installer creation failed, but executable is ready" -ForegroundColor Yellow
            Set-Location ..
        }
    } else {
        Write-Host "⚠ Inno Setup not found - skipping installer creation" -ForegroundColor Yellow
        Write-Host "You can manually install Inno Setup and run: iscc `"$issFile`"" -ForegroundColor Yellow
    }
}

# Success message
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "BUILD COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Executable location: $InstallDir\bin\AudioGridderServer.exe" -ForegroundColor Cyan
if (-not $SkipInstaller) {
    Write-Host "Installer location: $InstallDir\AudioGridderServer-GPU-Setup.exe" -ForegroundColor Cyan
}
Write-Host ""
Write-Host "To test the server:" -ForegroundColor Yellow
Write-Host "1. Install Visual C++ Redistributable (see VCREDIST_REQUIRED.txt)" -ForegroundColor White
Write-Host "2. If using GPU features, ensure NVIDIA drivers are installed" -ForegroundColor White
Write-Host "3. Run: $InstallDir\bin\AudioGridderServer.exe" -ForegroundColor White
Write-Host ""

# Test the executable
Write-Host "Testing executable..." -ForegroundColor Yellow
$exePath = "$InstallDir\bin\AudioGridderServer.exe"
if (Test-Path $exePath) {
    try {
        $fileInfo = Get-Item $exePath
        Write-Host "✓ Executable size: $([math]::Round($fileInfo.Length / 1MB, 2)) MB" -ForegroundColor Green
        
        # Quick dependency check
        $dependencies = & dumpbin /dependents $exePath 2>$null | Select-String "\.dll"
        if ($dependencies) {
            Write-Host "✓ Executable has expected dependencies" -ForegroundColor Green
        }
    } catch {
        Write-Host "⚠ Could not analyze executable" -ForegroundColor Yellow
    }
} else {
    Write-Host "✗ ERROR: Executable not found!" -ForegroundColor Red
}

Write-Host ""
Write-Host "Build completed at $(Get-Date)" -ForegroundColor Cyan
