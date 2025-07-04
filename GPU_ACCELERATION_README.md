# AudioGridder GPU Acceleration

This document describes the GPU acceleration feature added to AudioGridder, which uses CUDA to accelerate audio processing on compatible NVIDIA GPUs.

## Overview

The GPU acceleration feature provides:
- **GPU-accelerated audio processing** using CUDA kernels
- **Automatic fallback to CPU** when GPU is unavailable or fails
- **Memory-efficient GPU buffer management** with automatic resizing
- **Performance monitoring** to track GPU vs CPU processing times
- **Multi-GPU support** with device selection capabilities

## Architecture

### Core Components

1. **CUDAManager** (`CUDAManager.hpp/cpp`)
   - Singleton class managing CUDA runtime and devices
   - Device enumeration and selection
   - Memory management and error handling
   - Thread-safe operations

2. **GPUAudioBuffer** (`GPUAudioBuffer.hpp/cpp`)
   - Template class for GPU audio buffer management
   - Supports both float and double precision
   - Automatic memory allocation and deallocation
   - Efficient host-device memory transfers

3. **GPUProcessor** (`GPUProcessor.hpp/cpp`)
   - Extends the base Processor class
   - Implements GPU-accelerated audio processing
   - Automatic fallback to CPU processing
   - Performance monitoring and statistics

4. **AudioKernels** (`AudioKernels.hpp/cu`)
   - CUDA kernels for common audio operations
   - Gain, mixing, filtering, and buffer operations
   - Optimized for parallel execution

### Integration

The GPU acceleration is integrated into the existing AudioGridder architecture:

- **ProcessorChain** automatically creates GPUProcessor instances when CUDA is available
- **Transparent operation** - existing plugins work without modification
- **Configuration options** to enable/disable GPU processing per plugin
- **CMake integration** with optional CUDA support

## Features

### Supported Operations

- **Gain processing** - Apply gain/attenuation to audio signals
- **Buffer mixing** - Mix multiple audio streams
- **Low-pass filtering** - Basic frequency filtering
- **Buffer clearing** - Efficient zero-filling of audio buffers

### Performance Benefits

- **Parallel processing** - Process multiple audio channels simultaneously
- **High throughput** - Optimized for large audio buffers
- **Low latency** - Minimal overhead for GPU operations
- **Memory efficiency** - Reuse GPU buffers across processing calls

### Automatic Fallback

The system automatically falls back to CPU processing when:
- CUDA is not available or not installed
- GPU memory is insufficient
- GPU processing fails or times out
- Plugin is not compatible with GPU processing

## Configuration

### Build Configuration

Enable GPU acceleration during build:

```bash
cmake -DAG_ENABLE_CUDA=ON ..
make
```

### Runtime Configuration

GPU processing can be controlled via:
- **Server settings** - Global enable/disable
- **Per-plugin settings** - Individual plugin control
- **Performance thresholds** - Automatic switching based on load

## Requirements

### Hardware Requirements

- **NVIDIA GPU** with CUDA Compute Capability 3.5 or higher
- **Minimum 2GB GPU memory** recommended
- **PCIe 3.0 or higher** for optimal performance

### Software Requirements

- **CUDA Toolkit 11.0 or higher**
- **NVIDIA Driver 450.80.02 or higher**
- **CMake 3.15 or higher** with CUDA support

## Performance Monitoring

The GPU acceleration system provides detailed performance metrics:

```cpp
// Get processing times
double gpuTime = processor->getGPUProcessingTime();
double cpuTime = processor->getCPUProcessingTime();

// Get memory usage
size_t gpuMemory = processor->getGPUMemoryUsage();

// Check GPU availability
bool canUseGPU = processor->canUseGPU();
```

## API Reference

### CUDAManager

```cpp
// Get singleton instance
auto& manager = CUDAManager::getInstance();

// Initialize CUDA
bool success = manager.initialize();

// Get device information
int deviceCount = manager.getDeviceCount();
String deviceName = manager.getDeviceName(0);
size_t deviceMemory = manager.getDeviceMemory(0);

// Set active device
bool success = manager.setDevice(0);
```

### GPUAudioBuffer

```cpp
// Create and initialize buffer
GPUAudioBuffer<float> buffer;
bool success = buffer.initialize(numChannels, numSamples);

// Copy data to/from GPU
buffer.copyFromJUCEBuffer(juceBuffer);
buffer.copyToJUCEBuffer(juceBuffer);

// Get device pointers
float* deviceData = buffer.getDeviceChannelData(channel);
```

### GPUProcessor

```cpp
// Create GPU processor
auto processor = std::make_shared<GPUProcessor>(chain, id, sampleRate, blockSize);

// Enable/disable GPU processing
processor->setGPUEnabled(true);

// Check GPU status
bool gpuEnabled = processor->isGPUEnabled();
bool canUseGPU = processor->canUseGPU();

// Get performance metrics
double gpuTime = processor->getGPUProcessingTime();
size_t memoryUsage = processor->getGPUMemoryUsage();
```

## Troubleshooting

### Common Issues

1. **CUDA not found during build**
   - Install CUDA Toolkit
   - Set CUDA_PATH environment variable
   - Verify CMake can find CUDA

2. **GPU processing fails at runtime**
   - Check NVIDIA driver version
   - Verify GPU memory availability
   - Check CUDA runtime installation

3. **Performance issues**
   - Monitor GPU memory usage
   - Check for memory fragmentation
   - Verify optimal buffer sizes

### Debug Information

Enable debug logging to troubleshoot issues:

```cpp
// Check CUDA availability
if (!CUDAManager::getInstance().isAvailable()) {
    logln("CUDA Error: " + CUDAManager::getInstance().getLastError());
}

// Monitor GPU processing
processor->setGPUEnabled(true);
if (!processor->canUseGPU()) {
    logln("GPU processing not available");
}
```

## Future Enhancements

Planned improvements include:

- **Advanced DSP kernels** - FFT, convolution, dynamics processing
- **Multi-GPU support** - Distribute processing across multiple GPUs
- **Adaptive processing** - Dynamic CPU/GPU switching based on load
- **Plugin-specific optimizations** - Tailored kernels for specific plugin types
- **Real-time performance tuning** - Automatic parameter optimization

## Contributing

To contribute to GPU acceleration development:

1. **Follow CUDA best practices** for kernel development
2. **Test on multiple GPU architectures** (Pascal, Turing, Ampere, etc.)
3. **Profile performance** using NVIDIA Nsight tools
4. **Maintain CPU compatibility** - always provide CPU fallback
5. **Document new kernels** and their performance characteristics

## License

The GPU acceleration feature is licensed under the same MIT license as AudioGridder.
