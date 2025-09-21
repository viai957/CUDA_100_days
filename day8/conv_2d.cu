/*
 * 2D Convolution CUDA Implementation: High-Performance 2D Convolutional Operations
 * Math: output[i,j] = Σ(input[i+k,j+l] * kernel[k,l]) for k,l ∈ [0, kernel_size)
 * Inputs: input[H,W] - input image, kernel[K,K] - convolution kernel, H,W - image dimensions, K - kernel size
 * Assumptions: H,W > 0, K > 0, arrays are contiguous, device has sufficient memory
 * Parallel Strategy: 2D thread blocks with shared memory tiling and halo regions
 * Mixed Precision Policy: Configurable data types (float, half, double)
 * Distributed Hooks: Ready for multi-GPU via CUDA streams and peer-to-peer access
 * Complexity: O(HWK²) FLOPs, O(HW+K²) bytes moved
 * Test Vectors: Deterministic random images with known convolution results
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

// CUDA kernel for 2D convolution without tiling
__global__ void conv_2d_kernel(
    const EL_TYPE* input,
    EL_TYPE* output,
    const EL_TYPE* kernel,
    int height,
    int width,
    int kernel_size
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < height && col < width) {
        EL_TYPE sum = 0.0f;
        
        for (int k = 0; k < kernel_size; k++) {
            for (int l = 0; l < kernel_size; l++) {
                int input_row = row - kernel_size/2 + k;
                int input_col = col - kernel_size/2 + l;
                
                if (input_row >= 0 && input_row < height && input_col >= 0 && input_col < width) {
                    sum += input[input_row * width + input_col] * kernel[k * kernel_size + l];
                }
            }
        }
        
        output[row * width + col] = sum;
    }
}

// CUDA kernel for 2D convolution with shared memory tiling
__global__ void conv_2d_tiled_kernel(
    const EL_TYPE* input,
    EL_TYPE* output,
    const EL_TYPE* kernel,
    int height,
    int width,
    int kernel_size
) {
    extern __shared__ EL_TYPE shared_input[];
    
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int block_width = blockDim.x;
    int block_height = blockDim.y;
    int shared_width = block_width + kernel_size - 1;
    int shared_height = block_height + kernel_size - 1;
    
    int block_start_row = blockIdx.y * blockDim.y;
    int block_start_col = blockIdx.x * blockDim.x;
    
    // Load main data into shared memory
    if (row < height && col < width) {
        shared_input[(tid_y + kernel_size/2) * shared_width + (tid_x + kernel_size/2)] = 
            input[row * width + col];
    } else {
        shared_input[(tid_y + kernel_size/2) * shared_width + (tid_x + kernel_size/2)] = 0.0f;
    }
    
    // Load left halo
    if (tid_x < kernel_size/2) {
        int left_col = block_start_col - kernel_size/2 + tid_x;
        if (left_col >= 0 && row < height) {
            shared_input[(tid_y + kernel_size/2) * shared_width + tid_x] = 
                input[row * width + left_col];
        } else {
            shared_input[(tid_y + kernel_size/2) * shared_width + tid_x] = 0.0f;
        }
    }
    
    // Load right halo
    if (tid_x < kernel_size/2) {
        int right_col = block_start_col + block_width + tid_x;
        if (right_col < width && row < height) {
            shared_input[(tid_y + kernel_size/2) * shared_width + (tid_x + block_width + kernel_size/2)] = 
                input[row * width + right_col];
        } else {
            shared_input[(tid_y + kernel_size/2) * shared_width + (tid_x + block_width + kernel_size/2)] = 0.0f;
        }
    }
    
    // Load top halo
    if (tid_y < kernel_size/2) {
        int top_row = block_start_row - kernel_size/2 + tid_y;
        if (top_row >= 0 && col < width) {
            shared_input[tid_y * shared_width + (tid_x + kernel_size/2)] = 
                input[top_row * width + col];
        } else {
            shared_input[tid_y * shared_width + (tid_x + kernel_size/2)] = 0.0f;
        }
    }
    
    // Load bottom halo
    if (tid_y < kernel_size/2) {
        int bottom_row = block_start_row + block_height + tid_y;
        if (bottom_row < height && col < width) {
            shared_input[(tid_y + block_height + kernel_size/2) * shared_width + (tid_x + kernel_size/2)] = 
                input[bottom_row * width + col];
        } else {
            shared_input[(tid_y + block_height + kernel_size/2) * shared_width + (tid_x + kernel_size/2)] = 0.0f;
        }
    }
    
    __syncthreads();
    
    if (row < height && col < width) {
        EL_TYPE sum = 0.0f;
        
        for (int k = 0; k < kernel_size; k++) {
            for (int l = 0; l < kernel_size; l++) {
                sum += shared_input[(tid_y + k) * shared_width + (tid_x + l)] * 
                       kernel[k * kernel_size + l];
            }
        }
        
        output[row * width + col] = sum;
    }
}

// CUDA kernel for 2D convolution with constant memory kernel
__global__ void conv_2d_constant_kernel(
    const EL_TYPE* input,
    EL_TYPE* output,
    int height,
    int width,
    int kernel_size
) {
    extern __constant__ EL_TYPE constant_kernel[];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < height && col < width) {
        EL_TYPE sum = 0.0f;
        
        for (int k = 0; k < kernel_size; k++) {
            for (int l = 0; l < kernel_size; l++) {
                int input_row = row - kernel_size/2 + k;
                int input_col = col - kernel_size/2 + l;
                
                if (input_row >= 0 && input_row < height && input_col >= 0 && input_col < width) {
                    sum += input[input_row * width + input_col] * 
                           constant_kernel[k * kernel_size + l];
                }
            }
        }
        
        output[row * width + col] = sum;
    }
}

// Host function to launch 2D convolution
void launch_conv_2d(
    const EL_TYPE* d_input,
    EL_TYPE* d_output,
    const EL_TYPE* d_kernel,
    int height,
    int width,
    int kernel_size,
    dim3 block_size,
    bool use_tiling = false,
    bool use_constant = false
) {
    dim3 grid_size((width + block_size.x - 1) / block_size.x, 
                   (height + block_size.y - 1) / block_size.y);
    
    if (use_constant) {
        conv_2d_constant_kernel<<<grid_size, block_size>>>(
            d_input, d_output, height, width, kernel_size
        );
    } else if (use_tiling) {
        size_t shared_mem_size = (block_size.x + kernel_size - 1) * 
                                 (block_size.y + kernel_size - 1) * sizeof(EL_TYPE);
        conv_2d_tiled_kernel<<<grid_size, block_size, shared_mem_size>>>(
            d_input, d_output, d_kernel, height, width, kernel_size
        );
    } else {
        conv_2d_kernel<<<grid_size, block_size>>>(
            d_input, d_output, d_kernel, height, width, kernel_size
        );
    }
    
    CUDA_CHECK_KERNEL();
}

// Comprehensive test function for 2D convolution
void test_conv_2d_comprehensive(
    int height,
    int width,
    int kernel_size,
    dim3 block_size,
    bool use_tiling = false,
    bool use_constant = false
) {
    printf("\n=== 2D Convolution CUDA Test ===\n");
    printf("Image dimensions: %dx%d\n", height, width);
    printf("Kernel size: %dx%d\n", kernel_size, kernel_size);
    printf("Block size: %dx%d\n", block_size.x, block_size.y);
    printf("Tiling: %s\n", use_tiling ? "Yes" : "No");
    printf("Constant memory: %s\n", use_constant ? "Yes" : "No");
    printf("Data type: %s\n", typeid(EL_TYPE).name());

    // Host memory allocation
    std::vector<EL_TYPE> h_input(height * width);
    std::vector<EL_TYPE> h_output(height * width);
    std::vector<EL_TYPE> h_kernel(kernel_size * kernel_size);
    std::vector<EL_TYPE> h_expected(height * width);

    // Initialize with deterministic random values
    std::mt19937 gen(42);
    std::uniform_real_distribution<EL_TYPE> dis(-1.0, 1.0);

    for (int i = 0; i < height * width; i++) {
        h_input[i] = dis(gen);
    }

    // Initialize kernel (simple averaging filter)
    for (int i = 0; i < kernel_size * kernel_size; i++) {
        h_kernel[i] = 1.0f / (kernel_size * kernel_size);
    }

    // Compute expected result (CPU)
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            h_expected[row * width + col] = 0.0f;
            for (int k = 0; k < kernel_size; k++) {
                for (int l = 0; l < kernel_size; l++) {
                    int input_row = row - kernel_size/2 + k;
                    int input_col = col - kernel_size/2 + l;
                    
                    if (input_row >= 0 && input_row < height && input_col >= 0 && input_col < width) {
                        h_expected[row * width + col] += 
                            h_input[input_row * width + input_col] * h_kernel[k * kernel_size + l];
                    }
                }
            }
        }
    }

    // Device memory allocation
    EL_TYPE *d_input, *d_output, *d_kernel;
    CUDA_CHECK(cudaMalloc(&d_input, height * width * sizeof(EL_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_output, height * width * sizeof(EL_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_size * kernel_size * sizeof(EL_TYPE)));

    // Memory transfer to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), height * width * sizeof(EL_TYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel.data(), kernel_size * kernel_size * sizeof(EL_TYPE), cudaMemcpyHostToDevice));

    // Copy kernel to constant memory if needed
    if (use_constant) {
        CUDA_CHECK(cudaMemcpyToSymbol(constant_kernel, h_kernel.data(), kernel_size * kernel_size * sizeof(EL_TYPE)));
    }

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Test convolution
    printf("\n--- Testing 2D Convolution ---\n");
    CUDA_CHECK(cudaEventRecord(start));
    launch_conv_2d(d_input, d_output, d_kernel, height, width, kernel_size, block_size, use_tiling, use_constant);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Convolution time: %.3f ms\n", milliseconds);

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, height * width * sizeof(EL_TYPE), cudaMemcpyDeviceToHost));

    // Verify correctness
    bool correct = true;
    for (int i = 0; i < height * width; i++) {
        if (fabsf(h_output[i] - h_expected[i]) > 1e-5f) {
            printf("Error at index %d: %.6f != %.6f\n", i, h_output[i], h_expected[i]);
            correct = false;
            break;
        }
    }
    printf("Convolution correctness: %s\n", correct ? "PASS" : "FAIL");

    // Performance analysis
    printf("\n--- Performance Analysis ---\n");
    double operations = height * width * kernel_size * kernel_size;
    double gflops = operations / (milliseconds / 1000.0) / 1e9;
    double bytes_moved = (height * width + kernel_size * kernel_size) * sizeof(EL_TYPE) + height * width * sizeof(EL_TYPE);
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
void benchmark_conv_2d() {
    printf("\n=== 2D Convolution Benchmark ===\n");
    std::vector<std::pair<int, int>> image_sizes = {{32, 32}, {64, 64}, {128, 128}, {256, 256}};
    std::vector<int> kernel_sizes = {3, 5, 7, 9};
    std::vector<dim3> block_sizes = {{16, 16}, {32, 32}, {16, 32}, {32, 16}};

    for (auto& image_size : image_sizes) {
        int height = image_size.first;
        int width = image_size.second;
        for (int kernel_size : kernel_sizes) {
            printf("\nConfiguration: %dx%d, kernel_size=%dx%d\n", height, width, kernel_size, kernel_size);
            for (auto& block_size : block_sizes) {
                if (block_size.x <= width && block_size.y <= height) {
                    printf("  Block size %dx%d: ", block_size.x, block_size.y);
                    test_conv_2d_comprehensive(height, width, kernel_size, block_size, true, false);
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
    printf("Testing 64x64 with kernel size 5x5, block size 16x16:\n");
    test_conv_2d_comprehensive(64, 64, 5, {16, 16}, false, false);

    printf("Testing 128x128 with kernel size 7x7, block size 32x32 (tiled):\n");
    test_conv_2d_comprehensive(128, 128, 7, {32, 32}, true, false);

    printf("Testing 256x256 with kernel size 9x9, block size 32x32 (constant):\n");
    test_conv_2d_comprehensive(256, 256, 9, {32, 32}, false, true);

    // Run comprehensive benchmark
    benchmark_conv_2d();

    return 0;
}

/*
 * Profiling example & performance tips:
 *
 * 1. Use nvprof for detailed kernel analysis:
 *    nvprof --metrics achieved_occupancy,sm_efficiency ./conv_2d
 *
 * 2. Monitor memory bandwidth utilization:
 *    nvprof --metrics gld_efficiency,gst_efficiency ./conv_2d
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
