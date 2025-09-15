## GPU 100 days Learning Journey
##### This document serves as a log of the progress and knowledge I gained while working on GPU programming and studying the PMPP (Parallel Programming and Optimization) book.

Mentor: https://github.com/hkproj/
Bro in the 100 days challenge: https://github.com/1y33/100Days

> Day 1

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

> Day 2

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
