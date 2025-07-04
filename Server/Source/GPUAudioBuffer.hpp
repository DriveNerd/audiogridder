/*
 * Copyright (c) 2024 AudioGridder GPU Extension
 * Licensed under MIT
 *
 * GPU Audio Buffer for efficient GPU-CPU audio data transfer
 */

#pragma once

#include <JuceHeader.h>
#include <memory>
#include <vector>

namespace e47 {

template<typename SampleType>
class GPUAudioBuffer {
public:
    GPUAudioBuffer();
    ~GPUAudioBuffer();
    
    // Initialize buffer with specified channels and samples
    bool initialize(int numChannels, int numSamples);
    void clear();
    
    // Size management
    void setSize(int numChannels, int numSamples, bool keepExistingContent = false);
    int getNumChannels() const { return m_numChannels; }
    int getNumSamples() const { return m_numSamples; }
    
    // Data transfer operations
    bool copyFromJUCEBuffer(const AudioBuffer<SampleType>& juceBuffer);
    bool copyToJUCEBuffer(AudioBuffer<SampleType>& juceBuffer) const;
    
    // GPU memory access
    SampleType* getDeviceChannelData(int channel);
    const SampleType* getDeviceChannelData(int channel) const;
    
    // Host memory access (for debugging/fallback)
    SampleType* getHostChannelData(int channel);
    const SampleType* getHostChannelData(int channel) const;
    
    // Synchronization
    void synchronizeHostToDevice();
    void synchronizeDeviceToHost();
    void synchronizeDeviceToHost(int channel);
    
    // GPU operations
    void clearOnDevice();
    void clearChannelOnDevice(int channel);
    
    // Memory info
    size_t getDeviceMemoryUsage() const;
    bool isInitialized() const { return m_initialized; }
    
private:
    bool m_initialized = false;
    int m_numChannels = 0;
    int m_numSamples = 0;
    
    // Device memory pointers for each channel
    std::vector<SampleType*> m_deviceChannelData;
    
    // Host memory for staging (pinned memory for faster transfers)
    std::vector<SampleType*> m_hostChannelData;
    
    // CUDA stream for async operations
    void* m_cudaStream = nullptr;
    
    void allocateMemory();
    void deallocateMemory();
    bool allocateDeviceMemory();
    bool allocateHostMemory();
    void deallocateDeviceMemory();
    void deallocateHostMemory();
};

// Explicit template instantiations
extern template class GPUAudioBuffer<float>;
extern template class GPUAudioBuffer<double>;

} // namespace e47
