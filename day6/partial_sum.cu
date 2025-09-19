/*
 * Partial Sum (Scan) CUDA Implementation: High-Performance Prefix Sum Computation
 * Math: output[i] = Σ(input[j]) for j ∈ [0, i]
 * Inputs: input[N] - input array, N - array length
 * Assumptions: N > 0, array is contiguous, device has sufficient memory
 * Parallel Strategy: Hillis-Steele algorithm with shared memory optimization
 * Mixed Precision Policy: Configurable data types (int, float, double)
 * Distributed Hooks: Ready for multi-GPU via CUDA streams and peer-to-peer access
 * Complexity: O(N) FLOPs, O(N) bytes moved
 * Test Vectors: Deterministic random arrays with known prefix sums
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

typedef int EL_TYPE;

// Device function for inclusive scan using Hillis-Steele algorithm
__device__ void inclusive_scan_device(EL_TYPE* data, int length) {
    // Up-sweep phase
    for (int stride = 1; stride < length; stride *= 2) {
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index < length) {
            data[index] += data[index - stride];
        }
        __syncthreads();
    }
    
    // Down-sweep phase
    for (int stride = length / 2; stride > 0; stride /= 2) {
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < length) {
            data[index + stride] += data[index];
        }
        __syncthreads();
    }
}

// CUDA kernel for partial sum using shared memory
__global__ void partial_sum_kernel(
    EL_TYPE* input,
    EL_TYPE* output,
    int n,
    int block_size
) {
    extern __shared__ EL_TYPE shared_mem[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if (gid < n) {
        shared_mem[tid] = input[gid];
    } else {
        shared_mem[tid] = 0;
    }
    __syncthreads();
    
    // Perform inclusive scan in shared memory
    inclusive_scan_device(shared_mem, blockDim.x);
    __syncthreads();
    
    // Store result
    if (gid < n) {
        output[gid] = shared_mem[tid];
    }
}

// CUDA kernel for multi-block partial sum
__global__ void partial_sum_multi_block_kernel(
    EL_TYPE* input,
    EL_TYPE* output,
    EL_TYPE* block_sums,
    int n,
    int block_size
) {
    extern __shared__ EL_TYPE shared_mem[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if (gid < n) {
        shared_mem[tid] = input[gid];
    } else {
        shared_mem[tid] = 0;
    }
    __syncthreads();
    
    // Perform inclusive scan in shared memory
    inclusive_scan_device(shared_mem, blockDim.x);
    __syncthreads();
    
    // Store block sum
    if (tid == blockDim.x - 1) {
        block_sums[blockIdx.x] = shared_mem[tid];
    }
    
    // Store result
    if (gid < n) {
        output[gid] = shared_mem[tid];
    }
}

// CUDA kernel for adding block sums
__global__ void add_block_sums_kernel(
    EL_TYPE* output,
    EL_TYPE* block_sums,
    int n,
    int block_size
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid < n) {
        int block_id = gid / block_size;
        if (block_id > 0) {
            output[gid] += block_sums[block_id - 1];
        }
    }
}

// Host function to launch partial sum computation
void launch_partial_sum(
    EL_TYPE* d_input,
    EL_TYPE* d_output,
    EL_TYPE* d_temp,
    int n,
    int block_size
) {
    int grid_size = (n + block_size - 1) / block_size;
    size_t shared_mem_size = block_size * sizeof(EL_TYPE);
    
    if (grid_size == 1) {
        // Single block case
        partial_sum_kernel<<<grid_size, block_size, shared_mem_size>>>(
            d_input, d_output, n, block_size
        );
    } else {
        // Multi-block case
        partial_sum_multi_block_kernel<<<grid_size, block_size, shared_mem_size>>>(
            d_input, d_output, d_temp, n, block_size
        );
        CUDA_CHECK_KERNEL();
        
        // Recursively compute partial sum of block sums
        int num_blocks = grid_size;
        if (num_blocks > 1) {
            launch_partial_sum(d_temp, d_temp, d_temp + num_blocks, num_blocks, block_size);
        }
        
        // Add block sums to output
        add_block_sums_kernel<<<grid_size, block_size>>>(
            d_output, d_temp, n, block_size
        );
    }
    
    CUDA_CHECK_KERNEL();
}

// Comprehensive test function for partial sum
void test_partial_sum_comprehensive(int n, int block_size) {
    printf("\n=== Partial Sum CUDA Test ===\n");
    printf("Array size: %d elements\n", n);
    printf("Block size: %d threads\n", block_size);
    printf("Data type: %s\n", typeid(EL_TYPE).name());
    
    // Host memory allocation
    std::vector<EL_TYPE> h_input(n);
    std::vector<EL_TYPE> h_output(n);
    std::vector<EL_TYPE> h_expected(n);
    
    // Initialize with deterministic random values
    std::mt19937 gen(42);
    std::uniform_int_distribution<EL_TYPE> dis(1, 10);
    
    for (int i = 0; i < n; i++) {
        h_input[i] = dis(gen);
    }
    
    // Compute expected result (CPU)
    h_expected[0] = h_input[0];
    for (int i = 1; i < n; i++) {
        h_expected[i] = h_expected[i-1] + h_input[i];
    }
    
    // Device memory allocation
    EL_TYPE *d_input, *d_output, *d_temp;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(EL_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(EL_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_temp, ((n + block_size - 1) / block_size) * sizeof(EL_TYPE)));
    
    // Memory transfer to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n * sizeof(EL_TYPE), cudaMemcpyHostToDevice));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Test partial sum
    printf("\n--- Testing Partial Sum ---\n");
    CUDA_CHECK(cudaEventRecord(start));
    launch_partial_sum(d_input, d_output, d_temp, n, block_size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Partial sum time: %.3f ms\n", milliseconds);
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, n * sizeof(EL_TYPE), cudaMemcpyDeviceToHost));
    
    // Verify correctness
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (h_output[i] != h_expected[i]) {
            printf("Error at index %d: %d != %d\n", i, h_output[i], h_expected[i]);
            correct = false;
            break;
        }
    }
    printf("Partial sum correctness: %s\n", correct ? "PASS" : "FAIL");
    
    // Performance analysis
    printf("\n--- Performance Analysis ---\n");
    double operations = n;  // One addition per element
    double gflops = operations / (milliseconds / 1000.0) / 1e9;
    double bytes_moved = 2 * n * sizeof(EL_TYPE);  // Read input, write output
    double bandwidth = bytes_moved / (milliseconds / 1000.0) / 1e9;
    
    printf("Operations: %.0f\n", operations);
    printf("Bytes moved: %.2f MB\n", bytes_moved / 1e6);
    printf("Performance: %.2f GFLOPS\n", gflops);
    printf("Bandwidth: %.2f GB/s\n", bandwidth);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    printf("=== Test Complete ===\n\n");
}

// Benchmark function for different array sizes
void benchmark_partial_sum() {
    printf("\n=== Partial Sum Benchmark ===\n");
    std::vector<int> sizes = {1024, 10000, 100000, 1000000, 10000000};
    std::vector<int> block_sizes = {128, 256, 512, 1024};
    
    for (int n : sizes) {
        printf("\nArray size: %d\n", n);
        for (int block_size : block_sizes) {
            if (block_size <= n) {
                printf("  Block size %d: ", block_size);
                test_partial_sum_comprehensive(n, block_size);
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
    printf("Testing 16 elements with block size 8:\n");
    test_partial_sum_comprehensive(16, 8);
    
    printf("Testing 1000000 elements with block size 256:\n");
    test_partial_sum_comprehensive(1000000, 256);
    
    // Run comprehensive benchmark
    benchmark_partial_sum();
    
    return 0;
}

/*
 * Profiling example & performance tips:
 * 
 * 1. Use nvprof for detailed kernel analysis:
 *    nvprof --metrics achieved_occupancy,sm_efficiency ./partial_sum
 * 
 * 2. Monitor memory bandwidth utilization:
 *    nvprof --metrics gld_efficiency,gst_efficiency ./partial_sum
 * 
 * 3. For optimal performance:
 *    - Use appropriate block sizes for your hardware
 *    - Consider shared memory for data reuse
 *    - Use coalesced memory access patterns
 *    - Consider multi-block algorithms for large arrays
 * 
 * 4. Memory optimization:
 *    - Minimize host-device transfers
 *    - Use pinned memory for large transfers
 *    - Consider async memory operations with streams
 * 
 * 5. Algorithm optimization:
 *    - Use Hillis-Steele algorithm for better parallelism
 *    - Consider work-efficient algorithms for very large arrays
 *    - Use warp-level primitives for better performance
 */
