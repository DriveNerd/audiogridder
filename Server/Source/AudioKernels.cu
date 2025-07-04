/*
 * Copyright (c) 2024 AudioGridder GPU Extension
 * Licensed under MIT
 *
 * CUDA kernels for GPU-accelerated audio processing
 */

#include "AudioKernels.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace e47 {

// Simple gain kernel
__global__ void applyGainKernel(float* data, int numSamples, float gain) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSamples) {
        data[idx] *= gain;
    }
}

__global__ void applyGainKernel(double* data, int numSamples, double gain) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSamples) {
        data[idx] *= gain;
    }
}

// Mix two audio buffers
__global__ void mixBuffersKernel(float* dest, const float* src, int numSamples, float mixLevel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSamples) {
        dest[idx] += src[idx] * mixLevel;
    }
}

__global__ void mixBuffersKernel(double* dest, const double* src, int numSamples, double mixLevel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSamples) {
        dest[idx] += src[idx] * mixLevel;
    }
}

// Clear buffer
__global__ void clearBufferKernel(float* data, int numSamples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSamples) {
        data[idx] = 0.0f;
    }
}

__global__ void clearBufferKernel(double* data, int numSamples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSamples) {
        data[idx] = 0.0;
    }
}

// Simple low-pass filter kernel
__global__ void lowPassFilterKernel(float* data, int numSamples, float cutoff, float sampleRate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSamples && idx > 0) {
        float alpha = cutoff / (cutoff + sampleRate);
        data[idx] = alpha * data[idx] + (1.0f - alpha) * data[idx - 1];
    }
}

__global__ void lowPassFilterKernel(double* data, int numSamples, double cutoff, double sampleRate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSamples && idx > 0) {
        double alpha = cutoff / (cutoff + sampleRate);
        data[idx] = alpha * data[idx] + (1.0 - alpha) * data[idx - 1];
    }
}

// Host functions to launch kernels
bool AudioKernels::applyGain(float* deviceData, int numSamples, float gain, cudaStream_t stream) {
    const int blockSize = 256;
    const int numBlocks = (numSamples + blockSize - 1) / blockSize;
    
    applyGainKernel<<<numBlocks, blockSize, 0, stream>>>(deviceData, numSamples, gain);
    
    return cudaGetLastError() == cudaSuccess;
}

bool AudioKernels::applyGain(double* deviceData, int numSamples, double gain, cudaStream_t stream) {
    const int blockSize = 256;
    const int numBlocks = (numSamples + blockSize - 1) / blockSize;
    
    applyGainKernel<<<numBlocks, blockSize, 0, stream>>>(deviceData, numSamples, gain);
    
    return cudaGetLastError() == cudaSuccess;
}

bool AudioKernels::mixBuffers(float* dest, const float* src, int numSamples, float mixLevel, cudaStream_t stream) {
    const int blockSize = 256;
    const int numBlocks = (numSamples + blockSize - 1) / blockSize;
    
    mixBuffersKernel<<<numBlocks, blockSize, 0, stream>>>(dest, src, numSamples, mixLevel);
    
    return cudaGetLastError() == cudaSuccess;
}

bool AudioKernels::mixBuffers(double* dest, const double* src, int numSamples, double mixLevel, cudaStream_t stream) {
    const int blockSize = 256;
    const int numBlocks = (numSamples + blockSize - 1) / blockSize;
    
    mixBuffersKernel<<<numBlocks, blockSize, 0, stream>>>(dest, src, numSamples, mixLevel);
    
    return cudaGetLastError() == cudaSuccess;
}

bool AudioKernels::clearBuffer(float* deviceData, int numSamples, cudaStream_t stream) {
    const int blockSize = 256;
    const int numBlocks = (numSamples + blockSize - 1) / blockSize;
    
    clearBufferKernel<<<numBlocks, blockSize, 0, stream>>>(deviceData, numSamples);
    
    return cudaGetLastError() == cudaSuccess;
}

bool AudioKernels::clearBuffer(double* deviceData, int numSamples, cudaStream_t stream) {
    const int blockSize = 256;
    const int numBlocks = (numSamples + blockSize - 1) / blockSize;
    
    clearBufferKernel<<<numBlocks, blockSize, 0, stream>>>(deviceData, numSamples);
    
    return cudaGetLastError() == cudaSuccess;
}

bool AudioKernels::lowPassFilter(float* deviceData, int numSamples, float cutoff, float sampleRate, cudaStream_t stream) {
    const int blockSize = 256;
    const int numBlocks = (numSamples + blockSize - 1) / blockSize;
    
    lowPassFilterKernel<<<numBlocks, blockSize, 0, stream>>>(deviceData, numSamples, cutoff, sampleRate);
    
    return cudaGetLastError() == cudaSuccess;
}

bool AudioKernels::lowPassFilter(double* deviceData, int numSamples, double cutoff, double sampleRate, cudaStream_t stream) {
    const int blockSize = 256;
    const int numBlocks = (numSamples + blockSize - 1) / blockSize;
    
    lowPassFilterKernel<<<numBlocks, blockSize, 0, stream>>>(deviceData, numSamples, cutoff, sampleRate);
    
    return cudaGetLastError() == cudaSuccess;
}

} // namespace e47
