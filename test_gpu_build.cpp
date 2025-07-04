// Test program to verify GPU functionality in AudioGridder
// Compile with: g++ -std=c++17 test_gpu_build.cpp -o test_gpu_build

#include <iostream>
#include <string>

#ifdef AUDIOGRIDDER_ENABLE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

int main() {
    std::cout << "AudioGridder GPU Build Test" << std::endl;
    std::cout << "===========================" << std::endl;
    
#ifdef AUDIOGRIDDER_ENABLE_CUDA
    std::cout << "CUDA support: ENABLED" << std::endl;
    
    // Check CUDA runtime
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error == cudaSuccess) {
        std::cout << "CUDA devices found: " << deviceCount << std::endl;
        
        for (int i = 0; i < deviceCount; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            
            std::cout << "Device " << i << ": " << prop.name << std::endl;
            std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
            std::cout << "  Global memory: " << (prop.totalGlobalMem / 1024 / 1024) << " MB" << std::endl;
            std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        }
        
        // Test basic CUDA operation
        float *d_test;
        error = cudaMalloc(&d_test, sizeof(float) * 1024);
        if (error == cudaSuccess) {
            std::cout << "✓ CUDA memory allocation test: PASSED" << std::endl;
            cudaFree(d_test);
        } else {
            std::cout << "✗ CUDA memory allocation test: FAILED" << std::endl;
        }
        
        // Test cuBLAS
        cublasHandle_t handle;
        cublasStatus_t status = cublasCreate(&handle);
        if (status == CUBLAS_STATUS_SUCCESS) {
            std::cout << "✓ cuBLAS initialization test: PASSED" << std::endl;
            cublasDestroy(handle);
        } else {
            std::cout << "✗ cuBLAS initialization test: FAILED" << std::endl;
        }
        
    } else {
        std::cout << "✗ CUDA runtime error: " << cudaGetErrorString(error) << std::endl;
    }
#else
    std::cout << "CUDA support: DISABLED" << std::endl;
    std::cout << "This build was compiled without CUDA support." << std::endl;
#endif
    
    std::cout << std::endl;
    std::cout << "Build configuration:" << std::endl;
    std::cout << "  C++ Standard: " << __cplusplus << std::endl;
    
#ifdef _WIN32
    std::cout << "  Platform: Windows" << std::endl;
#ifdef _WIN64
    std::cout << "  Architecture: x64" << std::endl;
#else
    std::cout << "  Architecture: x86" << std::endl;
#endif
#endif

#ifdef _MSC_VER
    std::cout << "  Compiler: MSVC " << _MSC_VER << std::endl;
#endif

    std::cout << std::endl;
    std::cout << "Test completed." << std::endl;
    
    return 0;
}
