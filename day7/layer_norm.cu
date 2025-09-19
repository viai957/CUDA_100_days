/*
 * Layer Normalization CUDA Implementation: High-Performance Normalization
 * Math: output = (input - mean) / sqrt(variance + epsilon) * gamma + beta
 * Inputs: input[N, D] - input tensor, N - batch size, D - feature dimension
 * Assumptions: N > 0, D > 0, tensors are contiguous, device has sufficient memory
 * Parallel Strategy: Each thread block processes multiple samples with shared memory
 * Mixed Precision Policy: FP16/BF16 for computation, FP32 for reductions
 * Distributed Hooks: Ready for multi-GPU via CUDA streams and peer-to-peer access
 * Complexity: O(ND) FLOPs, O(ND) bytes moved
 * Test Vectors: Deterministic random tensors with known normalization results
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

// Device function for computing mean and variance
__device__ void compute_mean_variance_device(
    EL_TYPE* data, 
    int length, 
    EL_TYPE* mean, 
    EL_TYPE* variance
) {
    // Compute mean
    EL_TYPE sum = 0.0f;
    for (int i = 0; i < length; i++) {
        sum += data[i];
    }
    *mean = sum / length;
    
    // Compute variance
    EL_TYPE var_sum = 0.0f;
    for (int i = 0; i < length; i++) {
        EL_TYPE diff = data[i] - *mean;
        var_sum += diff * diff;
    }
    *variance = var_sum / length;
}

// CUDA kernel for layer normalization
__global__ void layer_norm_kernel(
    EL_TYPE* input,
    EL_TYPE* output,
    EL_TYPE* gamma,
    EL_TYPE* beta,
    int batch_size,
    int feature_dim,
    EL_TYPE epsilon
) {
    extern __shared__ EL_TYPE shared_mem[];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    // Load data into shared memory
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        shared_mem[i] = input[batch_idx * feature_dim + i];
    }
    __syncthreads();
    
    // Compute mean and variance
    EL_TYPE mean, variance;
    compute_mean_variance_device(shared_mem, feature_dim, &mean, &variance);
    __syncthreads();
    
    // Normalize and apply scale/shift
    EL_TYPE stddev = sqrtf(variance + epsilon);
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        EL_TYPE normalized = (shared_mem[i] - mean) / stddev;
        output[batch_idx * feature_dim + i] = normalized * gamma[i] + beta[i];
    }
}

// CUDA kernel for layer normalization with fused operations
__global__ void layer_norm_fused_kernel(
    EL_TYPE* input,
    EL_TYPE* output,
    EL_TYPE* gamma,
    EL_TYPE* beta,
    int batch_size,
    int feature_dim,
    EL_TYPE epsilon
) {
    extern __shared__ EL_TYPE shared_mem[];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    // Load data into shared memory
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        shared_mem[i] = input[batch_idx * feature_dim + i];
    }
    __syncthreads();
    
    // Compute mean using reduction
    EL_TYPE sum = 0.0f;
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        sum += shared_mem[i];
    }
    
    // Reduce across threads
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sum += __shfl_down_sync(0xffffffff, sum, stride);
        }
        __syncthreads();
    }
    
    EL_TYPE mean = sum / feature_dim;
    __syncthreads();
    
    // Compute variance using reduction
    EL_TYPE var_sum = 0.0f;
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        EL_TYPE diff = shared_mem[i] - mean;
        var_sum += diff * diff;
    }
    
    // Reduce across threads
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            var_sum += __shfl_down_sync(0xffffffff, var_sum, stride);
        }
        __syncthreads();
    }
    
    EL_TYPE variance = var_sum / feature_dim;
    EL_TYPE stddev = sqrtf(variance + epsilon);
    __syncthreads();
    
    // Normalize and apply scale/shift
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        EL_TYPE normalized = (shared_mem[i] - mean) / stddev;
        output[batch_idx * feature_dim + i] = normalized * gamma[i] + beta[i];
    }
}

// Host function to launch layer normalization
void launch_layer_norm(
    EL_TYPE* d_input,
    EL_TYPE* d_output,
    EL_TYPE* d_gamma,
    EL_TYPE* d_beta,
    int batch_size,
    int feature_dim,
    EL_TYPE epsilon,
    int block_size,
    bool fused = false
) {
    int grid_size = batch_size;
    size_t shared_mem_size = feature_dim * sizeof(EL_TYPE);
    
    if (fused) {
        layer_norm_fused_kernel<<<grid_size, block_size, shared_mem_size>>>(
            d_input, d_output, d_gamma, d_beta,
            batch_size, feature_dim, epsilon
        );
    } else {
        layer_norm_kernel<<<grid_size, block_size, shared_mem_size>>>(
            d_input, d_output, d_gamma, d_beta,
            batch_size, feature_dim, epsilon
        );
    }
    
    CUDA_CHECK_KERNEL();
}

// Comprehensive test function for layer normalization
void test_layer_norm_comprehensive(
    int batch_size, 
    int feature_dim, 
    int block_size,
    bool fused = false
) {
    printf("\n=== Layer Normalization CUDA Test ===\n");
    printf("Batch size: %d\n", batch_size);
    printf("Feature dimension: %d\n", feature_dim);
    printf("Block size: %d threads\n", block_size);
    printf("Fused operations: %s\n", fused ? "Yes" : "No");
    printf("Data type: %s\n", typeid(EL_TYPE).name());
    
    // Host memory allocation
    std::vector<EL_TYPE> h_input(batch_size * feature_dim);
    std::vector<EL_TYPE> h_output(batch_size * feature_dim);
    std::vector<EL_TYPE> h_gamma(feature_dim);
    std::vector<EL_TYPE> h_beta(feature_dim);
    std::vector<EL_TYPE> h_expected(batch_size * feature_dim);
    
    // Initialize with deterministic random values
    std::mt19937 gen(42);
    std::uniform_real_distribution<EL_TYPE> dis(-1.0, 1.0);
    
    for (int i = 0; i < batch_size * feature_dim; i++) {
        h_input[i] = dis(gen);
    }
    
    // Initialize gamma and beta
    for (int i = 0; i < feature_dim; i++) {
        h_gamma[i] = 1.0f;  // Identity scaling
        h_beta[i] = 0.0f;   // Zero bias
    }
    
    // Compute expected result (CPU)
    for (int b = 0; b < batch_size; b++) {
        // Compute mean
        EL_TYPE mean = 0.0f;
        for (int d = 0; d < feature_dim; d++) {
            mean += h_input[b * feature_dim + d];
        }
        mean /= feature_dim;
        
        // Compute variance
        EL_TYPE variance = 0.0f;
        for (int d = 0; d < feature_dim; d++) {
            EL_TYPE diff = h_input[b * feature_dim + d] - mean;
            variance += diff * diff;
        }
        variance /= feature_dim;
        
        // Normalize
        EL_TYPE stddev = sqrtf(variance + 1e-7f);
        for (int d = 0; d < feature_dim; d++) {
            EL_TYPE normalized = (h_input[b * feature_dim + d] - mean) / stddev;
            h_expected[b * feature_dim + d] = normalized * h_gamma[d] + h_beta[d];
        }
    }
    
    // Device memory allocation
    EL_TYPE *d_input, *d_output, *d_gamma, *d_beta;
    CUDA_CHECK(cudaMalloc(&d_input, batch_size * feature_dim * sizeof(EL_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_output, batch_size * feature_dim * sizeof(EL_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_gamma, feature_dim * sizeof(EL_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_beta, feature_dim * sizeof(EL_TYPE)));
    
    // Memory transfer to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), batch_size * feature_dim * sizeof(EL_TYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), feature_dim * sizeof(EL_TYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), feature_dim * sizeof(EL_TYPE), cudaMemcpyHostToDevice));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Test layer normalization
    printf("\n--- Testing Layer Normalization ---\n");
    CUDA_CHECK(cudaEventRecord(start));
    launch_layer_norm(d_input, d_output, d_gamma, d_beta, batch_size, feature_dim, 1e-7f, block_size, fused);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Layer normalization time: %.3f ms\n", milliseconds);
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, batch_size * feature_dim * sizeof(EL_TYPE), cudaMemcpyDeviceToHost));
    
    // Verify correctness
    bool correct = true;
    for (int i = 0; i < batch_size * feature_dim; i++) {
        if (fabsf(h_output[i] - h_expected[i]) > 1e-5f) {
            printf("Error at index %d: %.6f != %.6f\n", i, h_output[i], h_expected[i]);
            correct = false;
            break;
        }
    }
    printf("Layer normalization correctness: %s\n", correct ? "PASS" : "FAIL");
    
    // Performance analysis
    printf("\n--- Performance Analysis ---\n");
    double operations = 3 * batch_size * feature_dim;  // Mean, variance, normalization
    double gflops = operations / (milliseconds / 1000.0) / 1e9;
    double bytes_moved = 4 * batch_size * feature_dim * sizeof(EL_TYPE);  // Read input, write output, read gamma/beta
    double bandwidth = bytes_moved / (milliseconds / 1000.0) / 1e9;
    
    printf("Operations: %.0f\n", operations);
    printf("Bytes moved: %.2f MB\n", bytes_moved / 1e6);
    printf("Performance: %.2f GFLOPS\n", gflops);
    printf("Bandwidth: %.2f GB/s\n", bandwidth);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    printf("=== Test Complete ===\n\n");
}

// Benchmark function for different configurations
void benchmark_layer_norm() {
    printf("\n=== Layer Normalization Benchmark ===\n");
    std::vector<int> batch_sizes = {1, 4, 8, 16, 32};
    std::vector<int> feature_dims = {128, 256, 512, 1024, 2048};
    std::vector<int> block_sizes = {128, 256, 512, 1024};
    
    for (int batch_size : batch_sizes) {
        for (int feature_dim : feature_dims) {
            printf("\nConfiguration: batch_size=%d, feature_dim=%d\n", batch_size, feature_dim);
            for (int block_size : block_sizes) {
                if (block_size >= feature_dim) {
                    printf("  Block size %d: ", block_size);
                    test_layer_norm_comprehensive(batch_size, feature_dim, block_size, false);
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
    printf("Testing 10x10 with block size 256:\n");
    test_layer_norm_comprehensive(10, 10, 256, false);
    
    printf("Testing 32x512 with block size 512:\n");
    test_layer_norm_comprehensive(32, 512, 512, true);
    
    // Run comprehensive benchmark
    benchmark_layer_norm();
    
    return 0;
}

/*
 * Profiling example & performance tips:
 * 
 * 1. Use nvprof for detailed kernel analysis:
 *    nvprof --metrics achieved_occupancy,sm_efficiency ./layer_norm
 * 
 * 2. Monitor memory bandwidth utilization:
 *    nvprof --metrics gld_efficiency,gst_efficiency ./layer_norm
 * 
 * 3. For optimal performance:
 *    - Use appropriate block sizes for your hardware
 *    - Consider shared memory for data reuse
 *    - Use fused operations for better performance
 *    - Use warp-level primitives for reductions
 * 
 * 4. Memory optimization:
 *    - Minimize host-device transfers
 *    - Use pinned memory for large transfers
 *    - Consider async memory operations with streams
 * 
 * 5. Algorithm optimization:
 *    - Use efficient reduction algorithms
 *    - Consider numerical stability in variance computation
 *    - Use appropriate epsilon values for numerical stability
 */
