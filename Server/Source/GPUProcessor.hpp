/*
 * Copyright (c) 2024 AudioGridder GPU Extension
 * Licensed under MIT
 *
 * GPU Processor - Wrapper for GPU-accelerated plugin processing
 */

#pragma once

#include "Processor.hpp"
#include "GPUAudioBuffer.hpp"
#include "CUDAManager.hpp"
#include <memory>

namespace e47 {

class GPUProcessor : public Processor {
public:
    GPUProcessor(ProcessorChain& chain, const String& id, double sampleRate, int blockSize, bool isClient = false);
    ~GPUProcessor() override;
    
    // Override core processing methods
    bool load(const String& settings, const String& layout, uint64 monoChannels, String& err) override;
    void unload() override;
    
    bool processBlock(AudioBuffer<float>& buffer, MidiBuffer& midi, int& latency) override;
    bool processBlock(AudioBuffer<double>& buffer, MidiBuffer& midi, int& latency) override;
    
    void prepareToPlay(double sampleRate, int maximumExpectedSamplesPerBlock) override;
    void releaseResources() override;
    
    // GPU-specific methods
    bool isGPUEnabled() const { return m_gpuEnabled; }
    bool canUseGPU() const;
    void setGPUEnabled(bool enabled);
    
    // Performance monitoring
    double getGPUProcessingTime() const { return m_gpuProcessingTime; }
    double getCPUProcessingTime() const { return m_cpuProcessingTime; }
    size_t getGPUMemoryUsage() const;
    
    // GPU device management
    bool setGPUDevice(int device);
    int getCurrentGPUDevice() const { return m_currentGPUDevice; }
    
private:
    bool m_gpuEnabled = false;
    bool m_gpuInitialized = false;
    int m_currentGPUDevice = -1;
    
    // GPU audio buffers
    std::unique_ptr<GPUAudioBuffer<float>> m_gpuBufferF;
    std::unique_ptr<GPUAudioBuffer<double>> m_gpuBufferD;
    
    // Performance tracking
    double m_gpuProcessingTime = 0.0;
    double m_cpuProcessingTime = 0.0;
    
    // GPU processing methods
    template<typename SampleType>
    bool processBlockGPU(AudioBuffer<SampleType>& buffer, MidiBuffer& midi, int& latency);
    
    template<typename SampleType>
    bool processBlockCPU(AudioBuffer<SampleType>& buffer, MidiBuffer& midi, int& latency);
    
    bool initializeGPU();
    void shutdownGPU();
    
    // GPU audio processing kernels
    bool processAudioOnGPU(float* deviceData, int numChannels, int numSamples);
    bool processAudioOnGPU(double* deviceData, int numChannels, int numSamples);
    
    // Fallback to CPU processing
    bool shouldFallbackToCPU() const;
    void logGPUError(const String& operation, const String& error);
};

} // namespace e47
