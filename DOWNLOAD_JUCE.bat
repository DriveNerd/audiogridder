@echo off
echo ========================================
echo JUCE Download Helper
echo ========================================
echo.

echo This script will help you download JUCE manually.
echo.

:: Check if JUCE directory already exists
if exist "JUCE\CMakeLists.txt" (
    echo JUCE is already installed and ready to use.
    pause
    exit /b 0
)

echo JUCE is required for building AudioGridder but is missing.
echo.
echo MANUAL DOWNLOAD INSTRUCTIONS:
echo.
echo 1. Open your web browser and go to:
echo    https://github.com/juce-framework/JUCE/archive/refs/heads/master.zip
echo.
echo 2. Download the ZIP file
echo.
echo 3. Extract the ZIP file
echo.
echo 4. Rename the extracted folder from "JUCE-master" to "JUCE"
echo.
echo 5. Move the "JUCE" folder to this directory:
echo    %CD%
echo.
echo 6. Verify that this file exists:
echo    %CD%\JUCE\CMakeLists.txt
echo.
echo 7. Run BUILD_MINIMAL_NO_DEPS.bat again
echo.

:: Try to open the download URL automatically
echo Attempting to open download page in your browser...
start https://github.com/juce-framework/JUCE/archive/refs/heads/master.zip

echo.
echo After downloading and extracting JUCE, press any key to continue...
pause

:: Check if JUCE was installed
if exist "JUCE\CMakeLists.txt" (
    echo.
    echo ✓ JUCE found! You can now run BUILD_MINIMAL_NO_DEPS.bat
) else (
    echo.
    echo ✗ JUCE still not found. Please follow the instructions above.
    echo.
    echo Make sure the folder structure looks like this:
    echo %CD%\JUCE\CMakeLists.txt
    echo %CD%\JUCE\modules    echo %CD%\JUCE\extras

echo.
pause
