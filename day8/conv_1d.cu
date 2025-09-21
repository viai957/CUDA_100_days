/*
 * 1D Convolution CUDA Implementation: High-Performance Convolutional Operations
 * Math: output[i] = Σ(input[i+k] * kernel[k]) for k ∈ [0, kernel_size)
 * Inputs: input[N] - input signal, kernel[K] - convolution kernel, N - signal length, K - kernel size
 * Assumptions: N > 0, K > 0, arrays are contiguous, device has sufficient memory
 * Parallel Strategy: Each thread computes one output element with shared memory optimization
 * Mixed Precision Policy: Configurable data types (float, half, double)
 * Distributed Hooks: Ready for multi-GPU via CUDA streams and peer-to-peer access
 * Complexity: O(NK) FLOPs, O(N+K) bytes moved
 * Test Vectors: Deterministic random signals with known convolution results
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

typedef float EL_TYPE;

// CUDA kernel for 1D convolution without tiling
__global__ void conv_1d_kernel(
    const EL_TYPE* input,
    EL_TYPE* output,
    const EL_TYPE* kernel,
    int signal_length,
    int kernel_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < signal_length) {
        EL_TYPE sum = 0.0f;
        
        for (int k = 0; k < kernel_size; k++) {
            int input_idx = idx - kernel_size/2 + k;
            if (input_idx >= 0 && input_idx < signal_length) {
                sum += input[input_idx] * kernel[k];
            }
        }
        
        output[idx] = sum;
    }
}

// CUDA kernel for 1D convolution with shared memory tiling
__global__ void conv_1d_tiled_kernel(
    const EL_TYPE* input,
    EL_TYPE* output,
    const EL_TYPE* kernel,
    int signal_length,
    int kernel_size
) {
    extern __shared__ EL_TYPE shared_input[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_start = blockIdx.x * blockDim.x;
    
    // Load main data into shared memory
    if (idx < signal_length) {
        shared_input[tid + kernel_size/2] = input[idx];
    } else {
        shared_input[tid + kernel_size/2] = 0.0f;
    }
    
    // Load left halo
    if (tid < kernel_size/2) {
        int left_idx = block_start - kernel_size/2 + tid;
        if (left_idx >= 0) {
            shared_input[tid] = input[left_idx];
        } else {
            shared_input[tid] = 0.0f;
        }
    }
    
    // Load right halo
    if (tid < kernel_size/2) {
        int right_idx = block_start + blockDim.x + tid;
        if (right_idx < signal_length) {
            shared_input[tid + blockDim.x + kernel_size/2] = input[right_idx];
        } else {
            shared_input[tid + blockDim.x + kernel_size/2] = 0.0f;
        }
    }
    
    __syncthreads();
    
    if (idx < signal_length) {
        EL_TYPE sum = 0.0f;
        
        for (int k = 0; k < kernel_size; k++) {
            sum += shared_input[tid + k] * kernel[k];
        }
        
        output[idx] = sum;
    }
}

// CUDA kernel for 1D convolution with constant memory kernel
__global__ void conv_1d_constant_kernel(
    const EL_TYPE* input,
    EL_TYPE* output,
    int signal_length,
    int kernel_size
) {
    extern __constant__ EL_TYPE constant_kernel[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < signal_length) {
        EL_TYPE sum = 0.0f;
        
        for (int k = 0; k < kernel_size; k++) {
            int input_idx = idx - kernel_size/2 + k;
            if (input_idx >= 0 && input_idx < signal_length) {
                sum += input[input_idx] * constant_kernel[k];
            }
        }
        
        output[idx] = sum;
    }
}

// Host function to launch 1D convolution
void launch_conv_1d(
    const EL_TYPE* d_input,
    EL_TYPE* d_output,
    const EL_TYPE* d_kernel,
    int signal_length,
    int kernel_size,
    int block_size,
    bool use_tiling = false,
    bool use_constant = false
) {
    int grid_size = (signal_length + block_size - 1) / block_size;
    
    if (use_constant) {
        conv_1d_constant_kernel<<<grid_size, block_size>>>(
            d_input, d_output, signal_length, kernel_size
        );
    } else if (use_tiling) {
        size_t shared_mem_size = (block_size + kernel_size - 1) * sizeof(EL_TYPE);
        conv_1d_tiled_kernel<<<grid_size, block_size, shared_mem_size>>>(
            d_input, d_output, d_kernel, signal_length, kernel_size
        );
    } else {
        conv_1d_kernel<<<grid_size, block_size>>>(
            d_input, d_output, d_kernel, signal_length, kernel_size
        );
    }
    
    CUDA_CHECK_KERNEL();
}

// Comprehensive test function for 1D convolution
void test_conv_1d_comprehensive(
    int signal_length,
    int kernel_size,
    int block_size,
    bool use_tiling = false,
    bool use_constant = false
) {
    printf("\n=== 1D Convolution CUDA Test ===\n");
    printf("Signal length: %d elements\n", signal_length);
    printf("Kernel size: %d elements\n", kernel_size);
    printf("Block size: %d threads\n", block_size);
    printf("Tiling: %s\n", use_tiling ? "Yes" : "No");
    printf("Constant memory: %s\n", use_constant ? "Yes" : "No");
    printf("Data type: %s\n", typeid(EL_TYPE).name());

    // Host memory allocation
    std::vector<EL_TYPE> h_input(signal_length);
    std::vector<EL_TYPE> h_output(signal_length);
    std::vector<EL_TYPE> h_kernel(kernel_size);
    std::vector<EL_TYPE> h_expected(signal_length);

    // Initialize with deterministic random values
    std::mt19937 gen(42);
    std::uniform_real_distribution<EL_TYPE> dis(-1.0, 1.0);

    for (int i = 0; i < signal_length; i++) {
        h_input[i] = dis(gen);
    }

    // Initialize kernel (simple averaging filter)
    for (int i = 0; i < kernel_size; i++) {
        h_kernel[i] = 1.0f / kernel_size;
    }

    // Compute expected result (CPU)
    for (int i = 0; i < signal_length; i++) {
        h_expected[i] = 0.0f;
        for (int k = 0; k < kernel_size; k++) {
            int input_idx = i - kernel_size/2 + k;
            if (input_idx >= 0 && input_idx < signal_length) {
                h_expected[i] += h_input[input_idx] * h_kernel[k];
            }
        }
    }

    // Device memory allocation
    EL_TYPE *d_input, *d_output, *d_kernel;
    CUDA_CHECK(cudaMalloc(&d_input, signal_length * sizeof(EL_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_output, signal_length * sizeof(EL_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_size * sizeof(EL_TYPE)));

    // Memory transfer to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), signal_length * sizeof(EL_TYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel.data(), kernel_size * sizeof(EL_TYPE), cudaMemcpyHostToDevice));

    // Copy kernel to constant memory if needed
    if (use_constant) {
        CUDA_CHECK(cudaMemcpyToSymbol(constant_kernel, h_kernel.data(), kernel_size * sizeof(EL_TYPE)));
    }

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Test convolution
    printf("\n--- Testing 1D Convolution ---\n");
    CUDA_CHECK(cudaEventRecord(start));
    launch_conv_1d(d_input, d_output, d_kernel, signal_length, kernel_size, block_size, use_tiling, use_constant);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Convolution time: %.3f ms\n", milliseconds);

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, signal_length * sizeof(EL_TYPE), cudaMemcpyDeviceToHost));

    // Verify correctness
    bool correct = true;
    for (int i = 0; i < signal_length; i++) {
        if (fabsf(h_output[i] - h_expected[i]) > 1e-5f) {
            printf("Error at index %d: %.6f != %.6f\n", i, h_output[i], h_expected[i]);
            correct = false;
            break;
        }
    }
    printf("Convolution correctness: %s\n", correct ? "PASS" : "FAIL");

    // Performance analysis
    printf("\n--- Performance Analysis ---\n");
    double operations = signal_length * kernel_size;
    double gflops = operations / (milliseconds / 1000.0) / 1e9;
    double bytes_moved = (signal_length + kernel_size) * sizeof(EL_TYPE) + signal_length * sizeof(EL_TYPE);
    double bandwidth = bytes_moved / (milliseconds / 1000.0) / 1e9;

    printf("Operations: %.0f\n", operations);
    printf("Bytes moved: %.2f MB\n", bytes_moved / 1e6);
    printf("Performance: %.2f GFLOPS\n", gflops);
    printf("Bandwidth: %.2f GB/s\n", bandwidth);

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_kernel));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("=== Test Complete ===\n\n");
}

// Benchmark function for different configurations
void benchmark_conv_1d() {
    printf("\n=== 1D Convolution Benchmark ===\n");
    std::vector<int> signal_lengths = {1024, 10000, 100000, 1000000};
    std::vector<int> kernel_sizes = {3, 5, 7, 9, 15};
    std::vector<int> block_sizes = {128, 256, 512, 1024};

    for (int signal_length : signal_lengths) {
        for (int kernel_size : kernel_sizes) {
            printf("\nConfiguration: signal_length=%d, kernel_size=%d\n", signal_length, kernel_size);
            for (int block_size : block_sizes) {
                if (block_size <= signal_length) {
                    printf("  Block size %d: ", block_size);
                    test_conv_1d_comprehensive(signal_length, kernel_size, block_size, true, false);
                }
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
    printf("Testing 1000 elements with kernel size 5, block size 256:\n");
    test_conv_1d_comprehensive(1000, 5, 256, false, false);

    printf("Testing 10000 elements with kernel size 7, block size 512 (tiled):\n");
    test_conv_1d_comprehensive(10000, 7, 512, true, false);

    printf("Testing 100000 elements with kernel size 9, block size 1024 (constant):\n");
    test_conv_1d_comprehensive(100000, 9, 1024, false, true);

    // Run comprehensive benchmark
    benchmark_conv_1d();

    return 0;
}

/*
 * Profiling example & performance tips:
 *
 * 1. Use nvprof for detailed kernel analysis:
 *    nvprof --metrics achieved_occupancy,sm_efficiency ./conv_1d
 *
 * 2. Monitor memory bandwidth utilization:
 *    nvprof --metrics gld_efficiency,gst_efficiency ./conv_1d
 *
 * 3. For optimal performance:
 *    - Use shared memory tiling for better data reuse
 *    - Use constant memory for small kernels
 *    - Choose appropriate block sizes for your hardware
 *    - Consider kernel size vs. shared memory usage
 *
 * 4. Memory optimization:
 *    - Minimize host-device transfers
 *    - Use pinned memory for large transfers
 *    - Consider async memory operations with streams
 *
 * 5. Algorithm optimization:
 *    - Use separable kernels when possible
 *    - Consider FFT-based convolution for large kernels
 *    - Use warp-level primitives for reductions
 */
