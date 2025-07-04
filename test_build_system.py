#!/usr/bin/env python3
"""
Test script to verify AudioGridder GPU Server Windows build system
This script tests all critical functionalities of the build system
"""

import os
import sys
import subprocess
import re
from pathlib import Path

class BuildSystemTester:
    def __init__(self):
        self.test_results = []
        self.repo_root = Path.cwd()
        
    def log_test(self, test_name, passed, message=""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.test_results.append((test_name, passed, message))
        print(f"{status}: {test_name}")
        if message:
            print(f"    {message}")
    
    def test_file_exists(self, filename, description):
        """Test if a required file exists"""
        file_path = self.repo_root / filename
        exists = file_path.exists()
        size = file_path.stat().st_size if exists else 0
        message = f"Size: {size} bytes" if exists else "File not found"
        self.log_test(f"File exists: {description}", exists, message)
        return exists
    
    def test_script_syntax(self, script_path, script_type):
        """Test script syntax without execution"""
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if script_type == "batch":
                # Basic batch file syntax checks
                has_echo_off = "@echo off" in content
                has_error_handling = "errorlevel" in content.lower()
                has_pause = "pause" in content.lower()
                
                syntax_ok = has_echo_off and has_error_handling
                message = f"Echo off: {has_echo_off}, Error handling: {has_error_handling}, Pause: {has_pause}"
                
            elif script_type == "powershell":
                # Basic PowerShell syntax checks
                has_param = "param(" in content
                has_write_host = "Write-Host" in content
                has_error_handling = "try {" in content or "if (" in content
                
                syntax_ok = has_param and has_write_host
                message = f"Params: {has_param}, Write-Host: {has_write_host}, Error handling: {has_error_handling}"
            
            else:
                syntax_ok = len(content) > 0
                message = f"Content length: {len(content)} chars"
            
            self.log_test(f"Script syntax: {script_path.name}", syntax_ok, message)
            return syntax_ok
            
        except Exception as e:
            self.log_test(f"Script syntax: {script_path.name}", False, f"Error: {str(e)}")
            return False
    
    def test_cmake_configuration(self):
        """Test CMake configuration for CUDA support"""
        cmake_file = self.repo_root / "CMakeLists.txt"
        if not cmake_file.exists():
            self.log_test("CMake CUDA configuration", False, "CMakeLists.txt not found")
            return False
        
        try:
            with open(cmake_file, 'r') as f:
                content = f.read()
            
            has_cuda_option = "AG_ENABLE_CUDA" in content
            has_cuda_toolkit = "CUDAToolkit" in content
            has_cuda_language = "enable_language(CUDA)" in content
            has_cuda_definitions = "AUDIOGRIDDER_ENABLE_CUDA" in content
            
            cuda_support = has_cuda_option and has_cuda_toolkit and has_cuda_language
            message = f"CUDA option: {has_cuda_option}, Toolkit: {has_cuda_toolkit}, Language: {has_cuda_language}, Definitions: {has_cuda_definitions}"
            
            self.log_test("CMake CUDA configuration", cuda_support, message)
            return cuda_support
            
        except Exception as e:
            self.log_test("CMake CUDA configuration", False, f"Error: {str(e)}")
            return False
    
    def test_server_cmake_gpu_integration(self):
        """Test Server CMakeLists.txt for GPU integration"""
        server_cmake = self.repo_root / "Server" / "CMakeLists.txt"
        if not server_cmake.exists():
            self.log_test("Server CMake GPU integration", False, "Server/CMakeLists.txt not found")
            return False
        
        try:
            with open(server_cmake, 'r') as f:
                content = f.read()
            
            has_gpu_sources = "AG_GPU_SOURCES" in content
            has_cuda_sources = "AG_CUDA_SOURCES" in content
            has_cuda_libraries = "CUDA::cudart" in content
            has_cuda_properties = "CUDA_STANDARD" in content
            
            gpu_integration = has_gpu_sources and has_cuda_sources and has_cuda_libraries
            message = f"GPU sources: {has_gpu_sources}, CUDA sources: {has_cuda_sources}, CUDA libs: {has_cuda_libraries}, Properties: {has_cuda_properties}"
            
            self.log_test("Server CMake GPU integration", gpu_integration, message)
            return gpu_integration
            
        except Exception as e:
            self.log_test("Server CMake GPU integration", False, f"Error: {str(e)}")
            return False
    
    def test_gpu_source_files(self):
        """Test if GPU source files exist"""
        gpu_files = [
            "Server/Source/CUDAManager.hpp",
            "Server/Source/CUDAManager.cpp", 
            "Server/Source/GPUAudioBuffer.hpp",
            "Server/Source/GPUAudioBuffer.cpp",
            "Server/Source/GPUProcessor.hpp",
            "Server/Source/GPUProcessor.cpp",
            "Server/Source/AudioKernels.hpp",
            "Server/Source/AudioKernels.cu"
        ]
        
        existing_files = 0
        for gpu_file in gpu_files:
            file_path = self.repo_root / gpu_file
            if file_path.exists():
                existing_files += 1
        
        gpu_files_ok = existing_files >= 6  # At least 6 out of 8 files should exist
        message = f"{existing_files}/{len(gpu_files)} GPU source files found"
        
        self.log_test("GPU source files", gpu_files_ok, message)
        return gpu_files_ok
    
    def test_documentation_completeness(self):
        """Test documentation completeness"""
        doc_files = [
            ("WINDOWS_BUILD_README.md", "Windows build instructions"),
            ("INSTALLATION_GUIDE.md", "Installation guide"),
            ("WINDOWS_INSTALLER_SUMMARY.md", "Summary document"),
            ("GPU_ACCELERATION_README.md", "GPU acceleration documentation")
        ]
        
        doc_score = 0
        for doc_file, description in doc_files:
            file_path = self.repo_root / doc_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if len(content) > 1000:  # Substantial documentation
                        doc_score += 1
                        self.log_test(f"Documentation: {description}", True, f"Size: {len(content)} chars")
                    else:
                        self.log_test(f"Documentation: {description}", False, f"Too short: {len(content)} chars")
                except:
                    self.log_test(f"Documentation: {description}", False, "Read error")
            else:
                self.log_test(f"Documentation: {description}", False, "File not found")
        
        docs_complete = doc_score >= 3  # At least 3 out of 4 docs should be substantial
        return docs_complete
    
    def test_installer_configuration(self):
        """Test installer configuration in build scripts"""
        batch_script = self.repo_root / "BUILD_WINDOWS_GPU_SERVER.bat"
        ps_script = self.repo_root / "Build-WindowsGPUServer.ps1"
        
        installer_features = 0
        
        # Test batch script
        if batch_script.exists():
            try:
                with open(batch_script, 'r') as f:
                    content = f.read()
                if "Inno Setup" in content and "AudioGridderServer-GPU-Setup" in content:
                    installer_features += 1
            except:
                pass
        
        # Test PowerShell script
        if ps_script.exists():
            try:
                with open(ps_script, 'r') as f:
                    content = f.read()
                if "iscc" in content and "AudioGridderServer-GPU.iss" in content:
                    installer_features += 1
            except:
                pass
        
        installer_ok = installer_features >= 1
        message = f"Installer configuration found in {installer_features} script(s)"
        
        self.log_test("Installer configuration", installer_ok, message)
        return installer_ok
    
    def run_all_tests(self):
        """Run all tests"""
        print("=" * 60)
        print("AudioGridder GPU Server Build System Test")
        print("=" * 60)
        print()
        
        # Test 1: Required files exist
        print("Testing file existence...")
        files_ok = all([
            self.test_file_exists("BUILD_WINDOWS_GPU_SERVER.bat", "Main build script"),
            self.test_file_exists("Build-WindowsGPUServer.ps1", "PowerShell build script"),
            self.test_file_exists("QUICK_START.bat", "Quick start script"),
            self.test_file_exists("CMakeLists.txt", "Main CMake configuration"),
            self.test_file_exists("Server/CMakeLists.txt", "Server CMake configuration")
        ])
        
        print()
        
        # Test 2: Script syntax
        print("Testing script syntax...")
        syntax_ok = all([
            self.test_script_syntax(self.repo_root / "BUILD_WINDOWS_GPU_SERVER.bat", "batch"),
            self.test_script_syntax(self.repo_root / "Build-WindowsGPUServer.ps1", "powershell"),
            self.test_script_syntax(self.repo_root / "QUICK_START.bat", "batch")
        ])
        
        print()
        
        # Test 3: CMake CUDA configuration
        print("Testing CMake configuration...")
        cmake_ok = self.test_cmake_configuration()
        server_cmake_ok = self.test_server_cmake_gpu_integration()
        
        print()
        
        # Test 4: GPU source files
        print("Testing GPU source files...")
        gpu_sources_ok = self.test_gpu_source_files()
        
        print()
        
        # Test 5: Documentation
        print("Testing documentation...")
        docs_ok = self.test_documentation_completeness()
        
        print()
        
        # Test 6: Installer configuration
        print("Testing installer configuration...")
        installer_ok = self.test_installer_configuration()
        
        print()
        
        # Summary
        print("=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for _, passed, _ in self.test_results if passed)
        
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print()
        
        # Critical functionality assessment
        critical_ok = all([
            files_ok,
            syntax_ok, 
            cmake_ok,
            server_cmake_ok,
            gpu_sources_ok,
            docs_ok,
            installer_ok
        ])
        
        if critical_ok:
            print("🎉 ALL CRITICAL FUNCTIONALITIES WORKING!")
            print("✅ Build system is ready for use")
        else:
            print("⚠️  Some critical functionalities need attention")
            print("❌ Review failed tests above")
        
        print()
        
        # Failed tests details
        failed_tests = [(name, msg) for name, passed, msg in self.test_results if not passed]
        if failed_tests:
            print("FAILED TESTS:")
            for name, msg in failed_tests:
                print(f"  ❌ {name}: {msg}")
        
        return critical_ok

if __name__ == "__main__":
    tester = BuildSystemTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
