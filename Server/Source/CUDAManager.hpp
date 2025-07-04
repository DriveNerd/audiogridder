/*
 * Copyright (c) 2024 AudioGridder GPU Extension
 * Licensed under MIT
 *
 * CUDA Manager for GPU-accelerated audio processing
 */

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <mutex>

#ifdef AUDIOGRIDDER_ENABLE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#endif

namespace e47 {

class CUDAManager {
public:
    static CUDAManager& getInstance();
    
    bool initialize();
    void shutdown();
    
    bool isAvailable() const { return m_cudaAvailable; }
    int getDeviceCount() const { return m_deviceCount; }
    std::string getDeviceName(int device) const;
    size_t getDeviceMemory(int device) const;
    
    bool setDevice(int device);
    int getCurrentDevice() const { return m_currentDevice; }
    
    // Memory management
    void* allocateDeviceMemory(size_t size);
    void freeDeviceMemory(void* ptr);
    bool copyToDevice(void* dst, const void* src, size_t size);
    bool copyFromDevice(void* dst, const void* src, size_t size);
    bool copyDeviceToDevice(void* dst, const void* src, size_t size);
    
    // Stream management
    void* createStream();
    void destroyStream(void* stream);
    void synchronizeStream(void* stream);
    void synchronizeDevice();
    
    // cuBLAS operations
    bool initializeCuBLAS();
    void* getCuBLASHandle() { return m_cublasHandle; }
    
    // Error handling
    std::string getLastError() const { return m_lastError; }
    
private:
    CUDAManager() = default;
    ~CUDAManager();
    
    bool m_initialized = false;
    bool m_cudaAvailable = false;
    int m_deviceCount = 0;
    int m_currentDevice = -1;
    void* m_cublasHandle = nullptr;
    std::string m_lastError;
    std::mutex m_mutex;
    
    void setError(const std::string& error);
    bool checkCudaError(const std::string& operation);
};

} // namespace e47
