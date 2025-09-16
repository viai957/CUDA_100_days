## GPU 100 days Learning Journey
##### This document serves as a log of the progress and knowledge I gained while working on GPU programming and studying the PMPP (Parallel Programming and Optimization) book.

Mentor: https://github.com/hkproj/
Bro in the 100 days challenge: https://github.com/1y33/100Days

## Day 1

Files: rope.py, RoPE.cu, RoPE_triton.py
Summary:
Implemented Rotary Position Embedding (RoPE) in three different approaches:

**PyTorch Implementation (rope.py):**
- Complex number-based rotation using torch.polar and torch.view_as_complex
- Precomputed frequency tensors for efficient batch processing
- Supports arbitrary sequence lengths and head dimensions

**CUDA Implementation (RoPE.cu):**
- Direct GPU kernel with manual memory management
- Optimized for coalesced memory access patterns
- Block-based parallelism across batch, sequence, and head dimensions
- Fused rotation operations for both query and key tensors

**Triton Implementation (RoPE_triton.py):**
- Production-grade kernel with autotuning for different hardware configurations
- Advanced memory layout optimization with configurable block sizes
- Mixed precision support (FP16/BF16) with FP32 reductions
- Comprehensive error handling and bounds checking
- Module wrapper for seamless PyTorch integration
- Built-in performance benchmarking and unit testing

**Key Learnings:**
- RoPE position-dependent rotations to query and key vectors
- Complex number representation enables efficient 2D rotations
- Memory layout optimization critical for GPU performance
- Triton provides higher-level abstractions while maintaining CUDA-level performance
- Autotuning essential for optimal performance across different hardware

**Mathematical Foundation:**
- Frequency calculation: θ_i = 10000^(-2i/d) for i ∈ [0, d/2)
- Position encoding: freq[m,i] = m * θ_i
- Rotation matrix: R(θ) = [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
- Applied to consecutive pairs of vector dimensions

Reading:

## Day 2

File: vector_add.cu
Summary:
Implemented vector addition using CUDA programming. Created a parallel kernel where each thread computes the sum of corresponding elements from two input arrays. Explored CUDA's execution model with proper grid and block configuration, memory management, and performance measurement using CUDA events.

**Key Implementation Details:**
- Kernel function `cuda_vectorAdd` with thread indexing: `i = blockIdx.x * blockDim.x + threadIdx.x`
- Dynamic grid sizing: `num_blocks = ceil((float)N / block_size)` to handle arbitrary vector sizes
- Comprehensive error checking using `CUDA_CHECK` macro for all CUDA API calls
- Performance profiling with `cudaEventRecord` and `cudaEventElapsedTime`
- Correctness verification by comparing GPU results with expected CPU computation

**Memory Management:**
- Host memory allocation using `malloc()` for input/output arrays
- Device memory allocation using `cudaMalloc()` for GPU arrays
- Bidirectional memory transfers using `cudaMemcpy()` (Host→Device and Device→Host)
- Proper cleanup with `cudaFree()` and `free()` to prevent memory leaks

**Performance Analysis:**
- Kernel execution time measurement using CUDA events
- CPU verification timing using `gettimeofday()` for comparison
- Configurable block size (128 threads per block) for optimization testing
- Tested with 1,000,000 elements demonstrating scalability

Learned:
- **CUDA Execution Model**: Understanding of grid, block, and thread hierarchy
- **Memory Management**: Device memory allocation, transfer, and cleanup patterns
- **Error Handling**: Importance of checking CUDA API return values for debugging
- **Performance Measurement**: Using CUDA events for accurate kernel timing
- **Thread Indexing**: Calculating global thread ID from block and thread indices
- **Grid Configuration**: Dynamic sizing based on problem size and block dimensions
- **Memory Bandwidth**: Impact of host-device memory transfers on performance

Reading:
Read Chapter 1 of the PMPP book.
Learned about the fundamentals of parallel programming, CUDA architecture, and the GPU execution model. Gained insights into how GPUs achieve massive parallelism through thousands of concurrent threads and the importance of memory coalescing for optimal performance.

## Day 3

Files: vector_add.cu, vector_add_triton.py, vector_add.py
Summary:
Implemented comprehensive vector addition using three different approaches, following the same design patterns established in day1. Created production-grade implementations with advanced optimization techniques, comprehensive testing, and performance analysis.

**CUDA Implementation (vector_add.cu):**
- High-performance kernel with optimized memory access patterns
- Multiple kernel variants: standard and shared memory optimized
- Unrolled loops for better instruction-level parallelism
- Comprehensive error handling and bounds checking
- Built-in performance benchmarking and analysis
- Configurable data types and block sizes
- GPU device information printing and optimization utilities

**Triton Implementation (vector_add_triton.py):**
- Production-grade kernel with autotuning for different hardware configurations
- Vectorized operations for better memory bandwidth utilization
- Advanced memory layout optimization with configurable block sizes
- Mixed precision support (FP16/BF16) with FP32 reductions
- Comprehensive error handling and bounds checking
- Module wrapper for seamless PyTorch integration
- Built-in performance benchmarking and unit testing
- Distributed training hooks via tl.comm_* primitives

**PyTorch Implementation (vector_add.py):**
- High-level tensor operations with automatic parallelization
- Advanced module wrapper with performance tracking
- Mixed precision support with automatic type conversion
- Configurable data types and device placement
- Memory usage monitoring and optimization
- Comprehensive benchmarking across different configurations
- Gradient checkpointing for memory efficiency
- Built-in performance statistics and analysis

**Key Implementation Details:**
- **Memory Management**: Optimized host-device transfers with proper cleanup
- **Performance Analysis**: Comprehensive timing, bandwidth, and GFLOPS calculations
- **Error Handling**: Robust error checking and validation throughout
- **Scalability**: Support for vectors from 1K to 10M+ elements
- **Hardware Optimization**: Autotuning for different GPU architectures
- **Memory Bandwidth**: Optimized for coalesced memory access patterns

**Advanced Features:**
- **Multiple Kernel Variants**: Standard and shared memory optimized kernels
- **Vectorized Operations**: Process multiple elements per thread for better throughput
- **Autotuning**: Automatic selection of optimal block sizes and configurations
- **Performance Monitoring**: Real-time tracking of execution time and memory usage
- **Comprehensive Testing**: Unit tests, correctness verification, and determinism checks
- **Benchmarking Suite**: Automated performance testing across different sizes and configurations

**Mathematical Foundation:**
- Element-wise addition: C[i] = A[i] + B[i] for all i in [0, N)
- Memory complexity: O(N) bytes moved (3N reads/writes)
- Computational complexity: O(N) operations
- Parallel efficiency: Near-linear speedup with proper block sizing

**Performance Optimizations:**
- **Memory Coalescing**: Contiguous memory access patterns for optimal bandwidth
- **Loop Unrolling**: Process multiple elements per thread for better ILP
- **Shared Memory**: Cache frequently accessed data for better locality
- **Vectorization**: Use SIMD operations for better throughput
- **Mixed Precision**: FP16/BF16 for memory-bound operations, FP32 for accuracy

**Key Learnings:**
- **CUDA Execution Model**: Advanced understanding of grid/block configuration and thread indexing
- **Memory Hierarchy**: Importance of shared memory and cache utilization
- **Performance Profiling**: Using CUDA events and Triton profiling for optimization
- **Kernel Optimization**: Techniques for improving memory bandwidth and compute efficiency
- **Hardware Utilization**: Maximizing SM occupancy and memory bandwidth
- **Autotuning**: Importance of finding optimal configurations for different hardware
- **Error Handling**: Robust error checking and validation for production code

**Benchmarking Results:**
- Tested across vector sizes from 1K to 10M elements
- Multiple data types: FP32, FP16, INT32
- Performance comparison between standard and optimized implementations
- Memory bandwidth utilization analysis
- Speedup measurements against PyTorch baseline

Reading:
Read Chapter 2 of the PMPP book.
Learned about CUDA memory hierarchy, shared memory optimization, and kernel optimization techniques. Gained insights into memory coalescing, bank conflicts, and the importance of proper memory access patterns for optimal GPU performance.
