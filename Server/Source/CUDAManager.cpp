/*
 * Copyright (c) 2024 AudioGridder GPU Extension
 * Licensed under MIT
 *
 * CUDA Manager implementation for GPU-accelerated audio processing
 */

#include "CUDAManager.hpp"
#include <iostream>
#include <sstream>

namespace e47 {

CUDAManager& CUDAManager::getInstance() {
    static CUDAManager instance;
    return instance;
}

CUDAManager::~CUDAManager() {
    shutdown();
}

bool CUDAManager::initialize() {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (m_initialized) {
        return m_cudaAvailable;
    }
    
    m_initialized = true;
    
#ifdef AUDIOGRIDDER_ENABLE_CUDA
    // Check if CUDA is available
    cudaError_t error = cudaGetDeviceCount(&m_deviceCount);
    if (error != cudaSuccess) {
        setError("CUDA not available: " + std::string(cudaGetErrorString(error)));
        return false;
    }
    
    if (m_deviceCount == 0) {
        setError("No CUDA devices found");
        return false;
    }
    
    // Set default device
    if (!setDevice(0)) {
        return false;
    }
    
    // Initialize cuBLAS
    if (!initializeCuBLAS()) {
        return false;
    }
    
    m_cudaAvailable = true;
    return true;
#else
    setError("CUDA support not compiled in");
    return false;
#endif
}

void CUDAManager::shutdown() {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (!m_initialized) {
        return;
    }
    
#ifdef AUDIOGRIDDER_ENABLE_CUDA
    if (m_cublasHandle) {
        cublasDestroy(static_cast<cublasHandle_t>(m_cublasHandle));
        m_cublasHandle = nullptr;
    }
    
    if (m_currentDevice >= 0) {
        cudaDeviceReset();
    }
#endif
    
    m_cudaAvailable = false;
    m_currentDevice = -1;
}

std::string CUDAManager::getDeviceName(int device) const {
#ifdef AUDIOGRIDDER_ENABLE_CUDA
    if (device < 0 || device >= m_deviceCount) {
        return "";
    }
    
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, device);
    if (error != cudaSuccess) {
        return "";
    }
    
    return std::string(prop.name);
#else
    return "";
#endif
}

size_t CUDAManager::getDeviceMemory(int device) const {
#ifdef AUDIOGRIDDER_ENABLE_CUDA
    if (device < 0 || device >= m_deviceCount) {
        return 0;
    }
    
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, device);
    if (error != cudaSuccess) {
        return 0;
    }
    
    return prop.totalGlobalMem;
#else
    return 0;
#endif
}

bool CUDAManager::setDevice(int device) {
#ifdef AUDIOGRIDDER_ENABLE_CUDA
    if (device < 0 || device >= m_deviceCount) {
        setError("Invalid device index");
        return false;
    }
    
    cudaError_t error = cudaSetDevice(device);
    if (error != cudaSuccess) {
        setError("Failed to set CUDA device: " + std::string(cudaGetErrorString(error)));
        return false;
    }
    
    m_currentDevice = device;
    return true;
#else
    setError("CUDA support not available");
    return false;
#endif
}

void* CUDAManager::allocateDeviceMemory(size_t size) {
#ifdef AUDIOGRIDDER_ENABLE_CUDA
    void* ptr = nullptr;
    cudaError_t error = cudaMalloc(&ptr, size);
    if (error != cudaSuccess) {
        setError("Failed to allocate device memory: " + std::string(cudaGetErrorString(error)));
        return nullptr;
    }
    return ptr;
#else
    setError("CUDA support not available");
    return nullptr;
#endif
}

void CUDAManager::freeDeviceMemory(void* ptr) {
#ifdef AUDIOGRIDDER_ENABLE_CUDA
    if (ptr) {
        cudaFree(ptr);
    }
#endif
}

bool CUDAManager::copyToDevice(void* dst, const void* src, size_t size) {
#ifdef AUDIOGRIDDER_ENABLE_CUDA
    cudaError_t error = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        setError("Failed to copy to device: " + std::string(cudaGetErrorString(error)));
        return false;
    }
    return true;
#else
    setError("CUDA support not available");
    return false;
#endif
}

bool CUDAManager::copyFromDevice(void* dst, const void* src, size_t size) {
#ifdef AUDIOGRIDDER_ENABLE_CUDA
    cudaError_t error = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        setError("Failed to copy from device: " + std::string(cudaGetErrorString(error)));
        return false;
    }
    return true;
#else
    setError("CUDA support not available");
    return false;
#endif
}

bool CUDAManager::copyDeviceToDevice(void* dst, const void* src, size_t size) {
#ifdef AUDIOGRIDDER_ENABLE_CUDA
    cudaError_t error = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    if (error != cudaSuccess) {
        setError("Failed to copy device to device: " + std::string(cudaGetErrorString(error)));
        return false;
    }
    return true;
#else
    setError("CUDA support not available");
    return false;
#endif
}

void* CUDAManager::createStream() {
#ifdef AUDIOGRIDDER_ENABLE_CUDA
    cudaStream_t stream;
    cudaError_t error = cudaStreamCreate(&stream);
    if (error != cudaSuccess) {
        setError("Failed to create CUDA stream: " + std::string(cudaGetErrorString(error)));
        return nullptr;
    }
    return static_cast<void*>(stream);
#else
    setError("CUDA support not available");
    return nullptr;
#endif
}

void CUDAManager::destroyStream(void* stream) {
#ifdef AUDIOGRIDDER_ENABLE_CUDA
    if (stream) {
        cudaStreamDestroy(static_cast<cudaStream_t>(stream));
    }
#endif
}

void CUDAManager::synchronizeStream(void* stream) {
#ifdef AUDIOGRIDDER_ENABLE_CUDA
    if (stream) {
        cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    }
#endif
}

void CUDAManager::synchronizeDevice() {
#ifdef AUDIOGRIDDER_ENABLE_CUDA
    cudaDeviceSynchronize();
#endif
}

bool CUDAManager::initializeCuBLAS() {
#ifdef AUDIOGRIDDER_ENABLE_CUDA
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        setError("Failed to initialize cuBLAS");
        return false;
    }
    
    m_cublasHandle = static_cast<void*>(handle);
    return true;
#else
    setError("CUDA support not available");
    return false;
#endif
}

void CUDAManager::setError(const std::string& error) {
    m_lastError = error;
    std::cerr << "CUDAManager Error: " << error << std::endl;
}

bool CUDAManager::checkCudaError(const std::string& operation) {
#ifdef AUDIOGRIDDER_ENABLE_CUDA
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        setError(operation + " failed: " + std::string(cudaGetErrorString(error)));
        return false;
    }
    return true;
#else
    setError("CUDA support not available");
    return false;
#endif
}

} // namespace e47
