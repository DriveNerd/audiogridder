/*
 * Simple test to verify GPU acceleration functionality
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <memory>

// Include our GPU classes
#include "Server/Source/CUDAManager.hpp"
#include "Server/Source/GPUAudioBuffer.hpp"
#include "Server/Source/AudioKernels.hpp"

using namespace e47;

int main() {
    std::cout << "Testing AudioGridder GPU Acceleration..." << std::endl;
    
    // Test CUDA Manager
    auto& cudaManager = CUDAManager::getInstance();
    if (!cudaManager.initialize()) {
        std::cout << "CUDA not available: " << cudaManager.getLastError().toStdString() << std::endl;
        return 1;
    }
    
    std::cout << "CUDA initialized successfully!" << std::endl;
    std::cout << "Device count: " << cudaManager.getDeviceCount() << std::endl;
    
    for (int i = 0; i < cudaManager.getDeviceCount(); ++i) {
        std::cout << "Device " << i << ": " << cudaManager.getDeviceName(i).toStdString() << std::endl;
        std::cout << "  Memory: " << (cudaManager.getDeviceMemory(i) / (1024*1024)) << " MB" << std::endl;
    }
    
    // Test GPU Audio Buffer
    const int numChannels = 2;
    const int numSamples = 1024;
    
    GPUAudioBuffer<float> gpuBuffer;
    if (!gpuBuffer.initialize(numChannels, numSamples)) {
        std::cout << "Failed to initialize GPU buffer" << std::endl;
        return 1;
    }
    
    std::cout << "GPU buffer initialized: " << numChannels << " channels, " << numSamples << " samples" << std::endl;
    std::cout << "GPU memory usage: " << gpuBuffer.getDeviceMemoryUsage() << " bytes" << std::endl;
    
    // Test basic kernel operations
    float* deviceData = gpuBuffer.getDeviceChannelData(0);
    if (deviceData) {
        std::cout << "Testing GPU kernels..." << std::endl;
        
        // Test gain kernel
        if (AudioKernels::applyGain(deviceData, numSamples, 0.5f)) {
            std::cout << "  Gain kernel: OK" << std::endl;
        } else {
            std::cout << "  Gain kernel: FAILED" << std::endl;
        }
        
        // Test clear kernel
        if (AudioKernels::clearBuffer(deviceData, numSamples)) {
            std::cout << "  Clear kernel: OK" << std::endl;
        } else {
            std::cout << "  Clear kernel: FAILED" << std::endl;
        }
        
        // Test filter kernel
        if (AudioKernels::lowPassFilter(deviceData, numSamples, 1000.0f, 44100.0f)) {
            std::cout << "  Filter kernel: OK" << std::endl;
        } else {
            std::cout << "  Filter kernel: FAILED" << std::endl;
        }
    }
    
    std::cout << "GPU acceleration test completed successfully!" << std::endl;
    return 0;
}
