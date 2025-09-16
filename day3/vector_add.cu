/*
 * Vector Addition CUDA Implementation: High-Performance Element-wise Addition
 * Math: C[i] = A[i] + B[i] for all i in [0, N)
 * Inputs: A[N], B[N] - input vectors, N - vector length
 * Assumptions: N > 0, vectors are contiguous in memory, device has sufficient memory
 * Parallel Strategy: One thread per element, coalesced memory access patterns
 * Mixed Precision Policy: Configurable data types (int, float, double)
 * Distributed Hooks: Ready for multi-GPU via CUDA streams and peer-to-peer access
 * Complexity: O(N) FLOPs, O(N) bytes moved
 * Test Vectors: Deterministic random vectors with known sums
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <chrono>
#include <vector>
#include <random>
#include "../cuda_common.cuh"

// Configurable element type - can be changed to float, double, etc.
typedef float EL_TYPE;

// Device kernel for vector addition with optimized memory access
__global__ void vector_add_kernel(
    EL_TYPE *__restrict__ output,
    const EL_TYPE *__restrict__ input_a,
    const EL_TYPE *__restrict__ input_b,
    const int N
) {
    // Calculate global thread index with bounds checking
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process multiple elements per thread for better memory bandwidth utilization
    const int elements_per_thread = 4;
    int base_idx = idx * elements_per_thread;
    
    // Unrolled loop for better performance
    if (base_idx < N) {
        if (base_idx + 0 < N) output[base_idx + 0] = input_a[base_idx + 0] + input_b[base_idx + 0];
        if (base_idx + 1 < N) output[base_idx + 1] = input_a[base_idx + 1] + input_b[base_idx + 1];
        if (base_idx + 2 < N) output[base_idx + 2] = input_a[base_idx + 2] + input_b[base_idx + 2];
        if (base_idx + 3 < N) output[base_idx + 3] = input_a[base_idx + 3] + input_b[base_idx + 3];
    }
}

// Optimized kernel for large vectors with shared memory
__global__ void vector_add_shared_kernel(
    EL_TYPE *__restrict__ output,
    const EL_TYPE *__restrict__ input_a,
    const EL_TYPE *__restrict__ input_b,
    const int N
) {
    extern __shared__ EL_TYPE shared_mem[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_size = blockDim.x;
    
    // Load data into shared memory for better cache utilization
    if (idx < N) {
        shared_mem[tid] = input_a[idx] + input_b[idx];
    }
    
    __syncthreads();
    
    // Store result back to global memory
    if (idx < N) {
        output[idx] = shared_mem[tid];
    }
}

// Host function to launch vector addition kernel
void launch_vector_add(
    EL_TYPE *d_output,
    const EL_TYPE *d_input_a,
    const EL_TYPE *d_input_b,
    const int N,
    const int block_size,
    const bool use_shared_memory = false
) {
    // Calculate optimal grid size
    int grid_size = calculate_grid_size(N, block_size);
    
    if (use_shared_memory) {
        // Launch kernel with shared memory
        size_t shared_mem_size = block_size * sizeof(EL_TYPE);
        vector_add_shared_kernel<<<grid_size, block_size, shared_mem_size>>>(
            d_output, d_input_a, d_input_b, N
        );
    } else {
        // Launch standard kernel
        vector_add_kernel<<<grid_size, block_size>>>(
            d_output, d_input_a, d_input_b, N
        );
    }
    
    // Check for kernel launch errors
    CUDA_CHECK_KERNEL();
}

// Comprehensive test function with multiple configurations
void test_vector_add_comprehensive(int N, int block_size) {
    printf("\n=== Vector Addition CUDA Test ===\n");
    printf("Vector size: %d elements\n", N);
    printf("Block size: %d threads\n", block_size);
    printf("Data type: %s\n", typeid(EL_TYPE).name());
    
    // Host memory allocation
    std::vector<EL_TYPE> h_input_a(N);
    std::vector<EL_TYPE> h_input_b(N);
    std::vector<EL_TYPE> h_output(N);
    std::vector<EL_TYPE> h_expected(N);
    
    // Initialize with deterministic random values
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<EL_TYPE> dis(-100.0, 100.0);
    
    for (int i = 0; i < N; i++) {
        h_input_a[i] = dis(gen);
        h_input_b[i] = dis(gen);
        h_expected[i] = h_input_a[i] + h_input_b[i];
    }
    
    // Device memory allocation
    EL_TYPE *d_input_a, *d_input_b, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input_a, N * sizeof(EL_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_input_b, N * sizeof(EL_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(EL_TYPE)));
    
    // Memory transfer to device
    CUDA_CHECK(cudaMemcpy(d_input_a, h_input_a.data(), N * sizeof(EL_TYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input_b, h_input_b.data(), N * sizeof(EL_TYPE), cudaMemcpyHostToDevice));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Test standard kernel
    printf("\n--- Testing Standard Kernel ---\n");
    CUDA_CHECK(cudaEventRecord(start));
    launch_vector_add(d_output, d_input_a, d_input_b, N, block_size, false);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Standard kernel time: %.3f ms\n", milliseconds);
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, N * sizeof(EL_TYPE), cudaMemcpyDeviceToHost));
    
    // Verify correctness
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_output[i] - h_expected[i]) > 1e-5) {
            printf("Error at index %d: %.6f != %.6f\n", i, h_output[i], h_expected[i]);
            correct = false;
            break;
        }
    }
    printf("Standard kernel correctness: %s\n", correct ? "PASS" : "FAIL");
    
    // Test shared memory kernel
    printf("\n--- Testing Shared Memory Kernel ---\n");
    CUDA_CHECK(cudaEventRecord(start));
    launch_vector_add(d_output, d_input_a, d_input_b, N, block_size, true);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Shared memory kernel time: %.3f ms\n", milliseconds);
    
    // Copy result back and verify
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, N * sizeof(EL_TYPE), cudaMemcpyDeviceToHost));
    
    correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_output[i] - h_expected[i]) > 1e-5) {
            printf("Error at index %d: %.6f != %.6f\n", i, h_output[i], h_expected[i]);
            correct = false;
            break;
        }
    }
    printf("Shared memory kernel correctness: %s\n", correct ? "PASS" : "FAIL");
    
    // Performance analysis
    printf("\n--- Performance Analysis ---\n");
    double bytes_moved = 3 * N * sizeof(EL_TYPE);  // Read A, B, Write C
    double gflops = N / (milliseconds / 1000.0) / 1e9;
    double bandwidth = bytes_moved / (milliseconds / 1000.0) / 1e9;
    
    printf("Operations: %d\n", N);
    printf("Bytes moved: %.2f MB\n", bytes_moved / 1e6);
    printf("Performance: %.2f GFLOPS\n", gflops);
    printf("Bandwidth: %.2f GB/s\n", bandwidth);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input_a));
    CUDA_CHECK(cudaFree(d_input_b));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    printf("=== Test Complete ===\n\n");
}

// Benchmark function for different vector sizes
void benchmark_vector_sizes() {
    printf("\n=== Vector Addition Benchmark ===\n");
    std::vector<int> sizes = {1024, 10000, 100000, 1000000, 10000000};
    std::vector<int> block_sizes = {128, 256, 512, 1024};
    
    for (int N : sizes) {
        printf("\nVector size: %d\n", N);
        for (int block_size : block_sizes) {
            if (block_size <= N) {
                printf("  Block size %d: ", block_size);
                test_vector_add_comprehensive(N, block_size);
            }
        }
    }
}

int main() {
    // Print GPU information
    print_gpu_info();
    
    // Set random seed for reproducibility
    srand(42);
    
    // Test with different configurations
    test_vector_add_comprehensive(1000000, 256);
    
    // Run comprehensive benchmark
    benchmark_vector_sizes();
    
    return 0;
}

/*
 * Profiling example & performance tips:
 * 
 * 1. Use nvprof for detailed kernel analysis:
 *    nvprof --metrics achieved_occupancy,sm_efficiency ./vector_add
 * 
 * 2. Monitor memory bandwidth utilization:
 *    nvprof --metrics gld_efficiency,gst_efficiency ./vector_add
 * 
 * 3. For optimal performance:
 *    - Use coalesced memory access patterns
 *    - Choose block sizes that are multiples of 32 (warp size)
 *    - Consider shared memory for data reuse
 *    - Use appropriate data types (float vs double)
 * 
 * 4. Memory optimization:
 *    - Minimize host-device transfers
 *    - Use pinned memory for large transfers
 *    - Consider async memory operations with streams
 * 
 * 5. Kernel optimization:
 *    - Unroll loops for better instruction-level parallelism
 *    - Use __restrict__ for better compiler optimization
 *    - Consider vectorized memory operations
 */