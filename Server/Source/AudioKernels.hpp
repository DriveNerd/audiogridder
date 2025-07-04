/*
 * Copyright (c) 2024 AudioGridder GPU Extension
 * Licensed under MIT
 *
 * CUDA kernels header for GPU-accelerated audio processing
 */

#pragma once

#ifdef AUDIOGRIDDER_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace e47 {

class AudioKernels {
public:
    // Gain operations
    static bool applyGain(float* deviceData, int numSamples, float gain, cudaStream_t stream = 0);
    static bool applyGain(double* deviceData, int numSamples, double gain, cudaStream_t stream = 0);
    
    // Buffer mixing
    static bool mixBuffers(float* dest, const float* src, int numSamples, float mixLevel, cudaStream_t stream = 0);
    static bool mixBuffers(double* dest, const double* src, int numSamples, double mixLevel, cudaStream_t stream = 0);
    
    // Buffer clearing
    static bool clearBuffer(float* deviceData, int numSamples, cudaStream_t stream = 0);
    static bool clearBuffer(double* deviceData, int numSamples, cudaStream_t stream = 0);
    
    // Simple filtering
    static bool lowPassFilter(float* deviceData, int numSamples, float cutoff, float sampleRate, cudaStream_t stream = 0);
    static bool lowPassFilter(double* deviceData, int numSamples, double cutoff, double sampleRate, cudaStream_t stream = 0);
};

} // namespace e47
