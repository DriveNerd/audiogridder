/*
 * Copyright (c) 2024 AudioGridder GPU Extension
 * Licensed under MIT
 *
 * GPU Audio Buffer implementation
 */

#include "GPUAudioBuffer.hpp"
#include "CUDAManager.hpp"
#include <cstring>

#ifdef AUDIOGRIDDER_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace e47 {

template<typename SampleType>
GPUAudioBuffer<SampleType>::GPUAudioBuffer() {
    auto& cudaManager = CUDAManager::getInstance();
    if (cudaManager.isAvailable()) {
        m_cudaStream = cudaManager.createStream();
    }
}

template<typename SampleType>
GPUAudioBuffer<SampleType>::~GPUAudioBuffer() {
    clear();
    auto& cudaManager = CUDAManager::getInstance();
    if (m_cudaStream) {
        cudaManager.destroyStream(m_cudaStream);
    }
}

template<typename SampleType>
bool GPUAudioBuffer<SampleType>::initialize(int numChannels, int numSamples) {
    if (numChannels <= 0 || numSamples <= 0) {
        return false;
    }
    
    auto& cudaManager = CUDAManager::getInstance();
    if (!cudaManager.isAvailable()) {
        return false;
    }
    
    clear();
    
    m_numChannels = numChannels;
    m_numSamples = numSamples;
    
    if (!allocateDeviceMemory() || !allocateHostMemory()) {
        clear();
        return false;
    }
    
    m_initialized = true;
    return true;
}

template<typename SampleType>
void GPUAudioBuffer<SampleType>::clear() {
    deallocateMemory();
    m_numChannels = 0;
    m_numSamples = 0;
    m_initialized = false;
}

template<typename SampleType>
void GPUAudioBuffer<SampleType>::setSize(int numChannels, int numSamples, bool keepExistingContent) {
    if (numChannels == m_numChannels && numSamples == m_numSamples) {
        return;
    }
    
    if (!keepExistingContent) {
        initialize(numChannels, numSamples);
        return;
    }
    
    // TODO: Implement content preservation if needed
    initialize(numChannels, numSamples);
}

template<typename SampleType>
bool GPUAudioBuffer<SampleType>::copyFromJUCEBuffer(const AudioBuffer<SampleType>& juceBuffer) {
    if (!m_initialized) {
        return false;
    }
    
    int channels = jmin(m_numChannels, juceBuffer.getNumChannels());
    int samples = jmin(m_numSamples, juceBuffer.getNumSamples());
    
    auto& cudaManager = CUDAManager::getInstance();
    
    // Copy data channel by channel
    for (int ch = 0; ch < channels; ++ch) {
        const SampleType* sourceData = juceBuffer.getReadPointer(ch);
        
        // Copy to host staging buffer first
        std::memcpy(m_hostChannelData[ch], sourceData, samples * sizeof(SampleType));
        
        // Copy from host to device
        if (!cudaManager.copyToDevice(m_deviceChannelData[ch], m_hostChannelData[ch], 
                                     samples * sizeof(SampleType))) {
            return false;
        }
    }
    
    return true;
}

template<typename SampleType>
bool GPUAudioBuffer<SampleType>::copyToJUCEBuffer(AudioBuffer<SampleType>& juceBuffer) const {
    if (!m_initialized) {
        return false;
    }
    
    int channels = jmin(m_numChannels, juceBuffer.getNumChannels());
    int samples = jmin(m_numSamples, juceBuffer.getNumSamples());
    
    auto& cudaManager = CUDAManager::getInstance();
    
    // Copy data channel by channel
    for (int ch = 0; ch < channels; ++ch) {
        // Copy from device to host staging buffer
        if (!cudaManager.copyFromDevice(m_hostChannelData[ch], m_deviceChannelData[ch], 
                                       samples * sizeof(SampleType))) {
            return false;
        }
        
        // Copy from host staging buffer to JUCE buffer
        SampleType* destData = juceBuffer.getWritePointer(ch);
        std::memcpy(destData, m_hostChannelData[ch], samples * sizeof(SampleType));
    }
    
    return true;
}

template<typename SampleType>
SampleType* GPUAudioBuffer<SampleType>::getDeviceChannelData(int channel) {
    if (!m_initialized || channel < 0 || channel >= m_numChannels) {
        return nullptr;
    }
    return m_deviceChannelData[channel];
}

template<typename SampleType>
const SampleType* GPUAudioBuffer<SampleType>::getDeviceChannelData(int channel) const {
    if (!m_initialized || channel < 0 || channel >= m_numChannels) {
        return nullptr;
    }
    return m_deviceChannelData[channel];
}

template<typename SampleType>
SampleType* GPUAudioBuffer<SampleType>::getHostChannelData(int channel) {
    if (!m_initialized || channel < 0 || channel >= m_numChannels) {
        return nullptr;
    }
    return m_hostChannelData[channel];
}

template<typename SampleType>
const SampleType* GPUAudioBuffer<SampleType>::getHostChannelData(int channel) const {
    if (!m_initialized || channel < 0 || channel >= m_numChannels) {
        return nullptr;
    }
    return m_hostChannelData[channel];
}

template<typename SampleType>
void GPUAudioBuffer<SampleType>::synchronizeHostToDevice() {
    if (!m_initialized) {
        return;
    }
    
    auto& cudaManager = CUDAManager::getInstance();
    for (int ch = 0; ch < m_numChannels; ++ch) {
        cudaManager.copyToDevice(m_deviceChannelData[ch], m_hostChannelData[ch], 
                                m_numSamples * sizeof(SampleType));
    }
}

template<typename SampleType>
void GPUAudioBuffer<SampleType>::synchronizeDeviceToHost() {
    if (!m_initialized) {
        return;
    }
    
    auto& cudaManager = CUDAManager::getInstance();
    for (int ch = 0; ch < m_numChannels; ++ch) {
        cudaManager.copyFromDevice(m_hostChannelData[ch], m_deviceChannelData[ch], 
                                  m_numSamples * sizeof(SampleType));
    }
}

template<typename SampleType>
void GPUAudioBuffer<SampleType>::synchronizeDeviceToHost(int channel) {
    if (!m_initialized || channel < 0 || channel >= m_numChannels) {
        return;
    }
    
    auto& cudaManager = CUDAManager::getInstance();
    cudaManager.copyFromDevice(m_hostChannelData[channel], m_deviceChannelData[channel], 
                              m_numSamples * sizeof(SampleType));
}

template<typename SampleType>
void GPUAudioBuffer<SampleType>::clearOnDevice() {
    if (!m_initialized) {
        return;
    }
    
#ifdef AUDIOGRIDDER_ENABLE_CUDA
    for (int ch = 0; ch < m_numChannels; ++ch) {
        cudaMemset(m_deviceChannelData[ch], 0, m_numSamples * sizeof(SampleType));
    }
#endif
}

template<typename SampleType>
void GPUAudioBuffer<SampleType>::clearChannelOnDevice(int channel) {
    if (!m_initialized || channel < 0 || channel >= m_numChannels) {
        return;
    }
    
#ifdef AUDIOGRIDDER_ENABLE_CUDA
    cudaMemset(m_deviceChannelData[channel], 0, m_numSamples * sizeof(SampleType));
#endif
}

template<typename SampleType>
size_t GPUAudioBuffer<SampleType>::getDeviceMemoryUsage() const {
    if (!m_initialized) {
        return 0;
    }
    return m_numChannels * m_numSamples * sizeof(SampleType);
}

template<typename SampleType>
bool GPUAudioBuffer<SampleType>::allocateDeviceMemory() {
    auto& cudaManager = CUDAManager::getInstance();
    
    m_deviceChannelData.resize(m_numChannels);
    
    size_t channelSize = m_numSamples * sizeof(SampleType);
    for (int ch = 0; ch < m_numChannels; ++ch) {
        m_deviceChannelData[ch] = static_cast<SampleType*>(
            cudaManager.allocateDeviceMemory(channelSize));
        
        if (!m_deviceChannelData[ch]) {
            deallocateDeviceMemory();
            return false;
        }
    }
    
    return true;
}

template<typename SampleType>
bool GPUAudioBuffer<SampleType>::allocateHostMemory() {
    m_hostChannelData.resize(m_numChannels);
    
    size_t channelSize = m_numSamples * sizeof(SampleType);
    
#ifdef AUDIOGRIDDER_ENABLE_CUDA
    // Allocate pinned memory for faster transfers
    for (int ch = 0; ch < m_numChannels; ++ch) {
        cudaError_t error = cudaMallocHost(reinterpret_cast<void**>(&m_hostChannelData[ch]), channelSize);
        if (error != cudaSuccess) {
            deallocateHostMemory();
            return false;
        }
    }
#else
    // Fallback to regular allocation
    for (int ch = 0; ch < m_numChannels; ++ch) {
        m_hostChannelData[ch] = static_cast<SampleType*>(std::malloc(channelSize));
        if (!m_hostChannelData[ch]) {
            deallocateHostMemory();
            return false;
        }
    }
#endif
    
    return true;
}

template<typename SampleType>
void GPUAudioBuffer<SampleType>::deallocateMemory() {
    deallocateDeviceMemory();
    deallocateHostMemory();
}

template<typename SampleType>
void GPUAudioBuffer<SampleType>::deallocateDeviceMemory() {
    auto& cudaManager = CUDAManager::getInstance();
    
    for (auto* ptr : m_deviceChannelData) {
        if (ptr) {
            cudaManager.freeDeviceMemory(ptr);
        }
    }
    m_deviceChannelData.clear();
}

template<typename SampleType>
void GPUAudioBuffer<SampleType>::deallocateHostMemory() {
    for (auto* ptr : m_hostChannelData) {
        if (ptr) {
#ifdef AUDIOGRIDDER_ENABLE_CUDA
            cudaFreeHost(ptr);
#else
            std::free(ptr);
#endif
        }
    }
    m_hostChannelData.clear();
}

// Explicit template instantiations
template class GPUAudioBuffer<float>;
template class GPUAudioBuffer<double>;

} // namespace e47
