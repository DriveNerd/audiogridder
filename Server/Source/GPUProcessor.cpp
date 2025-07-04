/*
 * Copyright (c) 2024 AudioGridder GPU Extension
 * Licensed under MIT
 *
 * GPU Processor implementation
 */

#include "GPUProcessor.hpp"
#include "ProcessorChain.hpp"
#include "AudioKernels.hpp"
#include "App.hpp"
#include "Server.hpp"
#include <chrono>

namespace e47 {

GPUProcessor::GPUProcessor(ProcessorChain& chain, const String& id, double sampleRate, int blockSize, bool isClient)
    : Processor(chain, id, sampleRate, blockSize, isClient) {
    
    // Initialize GPU buffers
    m_gpuBufferF = std::make_unique<GPUAudioBuffer<float>>();
    m_gpuBufferD = std::make_unique<GPUAudioBuffer<double>>();
    
    // Check if GPU is available and initialize
    auto& cudaManager = CUDAManager::getInstance();
    if (cudaManager.initialize()) {
        m_gpuEnabled = true;
        logln("GPU processor created with CUDA support");
    } else {
        logln("GPU processor created without CUDA support: " + cudaManager.getLastError());
    }
}

GPUProcessor::~GPUProcessor() {
    shutdownGPU();
}

bool GPUProcessor::load(const String& settings, const String& layout, uint64 monoChannels, String& err) {
    // First load the plugin using the base class
    if (!Processor::load(settings, layout, monoChannels, err)) {
        return false;
    }
    
    // Initialize GPU processing if enabled
    if (m_gpuEnabled && !initializeGPU()) {
        logln("Failed to initialize GPU processing, falling back to CPU");
        m_gpuEnabled = false;
    }
    
    return true;
}

void GPUProcessor::unload() {
    shutdownGPU();
    Processor::unload();
}

bool GPUProcessor::processBlock(AudioBuffer<float>& buffer, MidiBuffer& midi, int& latency) {
    if (m_gpuEnabled && m_gpuInitialized) {
        return processBlockGPU(buffer, midi, latency);
    } else {
        return processBlockCPU(buffer, midi, latency);
    }
}

bool GPUProcessor::processBlock(AudioBuffer<double>& buffer, MidiBuffer& midi, int& latency) {
    if (m_gpuEnabled && m_gpuInitialized) {
        return processBlockGPU(buffer, midi, latency);
    } else {
        return processBlockCPU(buffer, midi, latency);
    }
}

template<typename SampleType>
bool GPUProcessor::processBlockGPU(AudioBuffer<SampleType>& buffer, MidiBuffer& midi, int& latency) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Get the appropriate GPU buffer
    auto* gpuBuffer = std::is_same_v<SampleType, float> ? 
        static_cast<GPUAudioBuffer<SampleType>*>(m_gpuBufferF.get()) :
        static_cast<GPUAudioBuffer<SampleType>*>(m_gpuBufferD.get());
    
    if (!gpuBuffer || !gpuBuffer->isInitialized()) {
        logGPUError("processBlockGPU", "GPU buffer not initialized");
        return processBlockCPU(buffer, midi, latency);
    }
    
    // Ensure GPU buffer is the right size
    if (gpuBuffer->getNumChannels() != buffer.getNumChannels() || 
        gpuBuffer->getNumSamples() != buffer.getNumSamples()) {
        if (!gpuBuffer->initialize(buffer.getNumChannels(), buffer.getNumSamples())) {
            logGPUError("processBlockGPU", "Failed to resize GPU buffer");
            return processBlockCPU(buffer, midi, latency);
        }
    }
    
    // Copy audio data to GPU
    if (!gpuBuffer->copyFromJUCEBuffer(buffer)) {
        logGPUError("processBlockGPU", "Failed to copy data to GPU");
        return processBlockCPU(buffer, midi, latency);
    }
    
    // Process audio on GPU
    bool success = false;
    if constexpr (std::is_same_v<SampleType, float>) {
        success = processAudioOnGPU(gpuBuffer->getDeviceChannelData(0), 
                                   buffer.getNumChannels(), buffer.getNumSamples());
    } else {
        success = processAudioOnGPU(gpuBuffer->getDeviceChannelData(0), 
                                   buffer.getNumChannels(), buffer.getNumSamples());
    }
    
    if (!success) {
        logGPUError("processBlockGPU", "GPU processing failed");
        return processBlockCPU(buffer, midi, latency);
    }
    
    // Copy processed data back to CPU
    if (!gpuBuffer->copyToJUCEBuffer(buffer)) {
        logGPUError("processBlockGPU", "Failed to copy data from GPU");
        return processBlockCPU(buffer, midi, latency);
    }
    
    // Process MIDI and handle latency (still done on CPU)
    if (!Processor::processBlock(buffer, midi, latency)) {
        return false;
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    m_gpuProcessingTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    
    return true;
}

template<typename SampleType>
bool GPUProcessor::processBlockCPU(AudioBuffer<SampleType>& buffer, MidiBuffer& midi, int& latency) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    bool result = Processor::processBlock(buffer, midi, latency);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    m_cpuProcessingTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    
    return result;
}

void GPUProcessor::prepareToPlay(double sampleRate, int maximumExpectedSamplesPerBlock) {
    Processor::prepareToPlay(sampleRate, maximumExpectedSamplesPerBlock);
    
    if (m_gpuEnabled) {
        // Pre-allocate GPU buffers with maximum expected size
        int maxChannels = jmax(getTotalNumInputChannels(), getTotalNumOutputChannels());
        if (maxChannels > 0) {
            m_gpuBufferF->initialize(maxChannels, maximumExpectedSamplesPerBlock);
            m_gpuBufferD->initialize(maxChannels, maximumExpectedSamplesPerBlock);
        }
    }
}

void GPUProcessor::releaseResources() {
    if (m_gpuEnabled) {
        m_gpuBufferF->clear();
        m_gpuBufferD->clear();
    }
    
    Processor::releaseResources();
}

bool GPUProcessor::canUseGPU() const {
    auto& cudaManager = CUDAManager::getInstance();
    return cudaManager.isAvailable() && cudaManager.getDeviceCount() > 0;
}

void GPUProcessor::setGPUEnabled(bool enabled) {
    if (enabled && !canUseGPU()) {
        logln("Cannot enable GPU: CUDA not available");
        return;
    }
    
    if (m_gpuEnabled != enabled) {
        m_gpuEnabled = enabled;
        
        if (enabled) {
            initializeGPU();
        } else {
            shutdownGPU();
        }
        
        logln("GPU processing " + String(enabled ? "enabled" : "disabled"));
    }
}

size_t GPUProcessor::getGPUMemoryUsage() const {
    size_t usage = 0;
    if (m_gpuBufferF && m_gpuBufferF->isInitialized()) {
        usage += m_gpuBufferF->getDeviceMemoryUsage();
    }
    if (m_gpuBufferD && m_gpuBufferD->isInitialized()) {
        usage += m_gpuBufferD->getDeviceMemoryUsage();
    }
    return usage;
}

bool GPUProcessor::setGPUDevice(int device) {
    auto& cudaManager = CUDAManager::getInstance();
    if (!cudaManager.setDevice(device)) {
        logGPUError("setGPUDevice", cudaManager.getLastError());
        return false;
    }
    
    m_currentGPUDevice = device;
    logln("GPU device set to: " + String(device) + " (" + cudaManager.getDeviceName(device) + ")");
    return true;
}

bool GPUProcessor::initializeGPU() {
    if (m_gpuInitialized) {
        return true;
    }
    
    auto& cudaManager = CUDAManager::getInstance();
    if (!cudaManager.isAvailable()) {
        return false;
    }
    
    // Set GPU device if not already set
    if (m_currentGPUDevice < 0) {
        if (!setGPUDevice(0)) {
            return false;
        }
    }
    
    m_gpuInitialized = true;
    logln("GPU processing initialized on device " + String(m_currentGPUDevice));
    return true;
}

void GPUProcessor::shutdownGPU() {
    if (m_gpuInitialized) {
        if (m_gpuBufferF) {
            m_gpuBufferF->clear();
        }
        if (m_gpuBufferD) {
            m_gpuBufferD->clear();
        }
        
        m_gpuInitialized = false;
        logln("GPU processing shutdown");
    }
}

bool GPUProcessor::processAudioOnGPU(float* deviceData, int numChannels, int numSamples) {
    // This is where we implement GPU-specific audio processing using CUDA kernels
    
#ifdef AUDIOGRIDDER_ENABLE_CUDA
    auto& cudaManager = CUDAManager::getInstance();
    
    // Example processing: Apply a simple gain and low-pass filter
    // In a real implementation, this would be much more sophisticated
    
    // Process each channel
    for (int ch = 0; ch < numChannels; ++ch) {
        float* channelData = deviceData + (ch * numSamples);
        
        // Apply a subtle gain (0.9 to prevent clipping)
        if (!AudioKernels::applyGain(channelData, numSamples, 0.9f)) {
            logGPUError("processAudioOnGPU", "Failed to apply gain");
            return false;
        }
        
        // Apply a gentle low-pass filter at 8kHz
        if (!AudioKernels::lowPassFilter(channelData, numSamples, 8000.0f, 44100.0f)) {
            logGPUError("processAudioOnGPU", "Failed to apply low-pass filter");
            return false;
        }
    }
    
    // Synchronize to ensure operations complete
    cudaManager.synchronizeDevice();
    
    return true;
#else
    return false;
#endif
}

bool GPUProcessor::processAudioOnGPU(double* deviceData, int numChannels, int numSamples) {
    // Double precision version
#ifdef AUDIOGRIDDER_ENABLE_CUDA
    auto& cudaManager = CUDAManager::getInstance();
    
    // Process each channel
    for (int ch = 0; ch < numChannels; ++ch) {
        double* channelData = deviceData + (ch * numSamples);
        
        // Apply a subtle gain (0.9 to prevent clipping)
        if (!AudioKernels::applyGain(channelData, numSamples, 0.9)) {
            logGPUError("processAudioOnGPU", "Failed to apply gain");
            return false;
        }
        
        // Apply a gentle low-pass filter at 8kHz
        if (!AudioKernels::lowPassFilter(channelData, numSamples, 8000.0, 44100.0)) {
            logGPUError("processAudioOnGPU", "Failed to apply low-pass filter");
            return false;
        }
    }
    
    cudaManager.synchronizeDevice();
    return true;
#else
    return false;
#endif
}

bool GPUProcessor::shouldFallbackToCPU() const {
    // Determine if we should fallback to CPU processing
    // This could be based on GPU memory usage, processing time, etc.
    return !m_gpuEnabled || !m_gpuInitialized;
}

void GPUProcessor::logGPUError(const String& operation, const String& error) {
    logln("GPU Error in " + operation + ": " + error);
}

// Explicit template instantiations
template bool GPUProcessor::processBlockGPU<float>(AudioBuffer<float>&, MidiBuffer&, int&);
template bool GPUProcessor::processBlockGPU<double>(AudioBuffer<double>&, MidiBuffer&, int&);
template bool GPUProcessor::processBlockCPU<float>(AudioBuffer<float>&, MidiBuffer&, int&);
template bool GPUProcessor::processBlockCPU<double>(AudioBuffer<double>&, MidiBuffer&, int&);

} // namespace e47
