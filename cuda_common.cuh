/*
 * CUDA Common Utilities: Error handling and debugging macros
 * Purpose: Centralized CUDA error checking and debugging utilities
 * Assumptions: CUDA runtime API available, proper error propagation
 * Usage: Include in all CUDA source files for consistent error handling
 */

#ifndef CUDA_COMMON_CUH
#define CUDA_COMMON_CUH
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Macro for CUDA error checking with automatic error reporting
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Macro for checking kernel launch errors
#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t error = cudaPeekAtLastError(); \
        if (error != cudaSuccess) { \
            fprintf(stderr, "Kernel launch error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
        error = cudaDeviceSynchronize(); \
        if (error != cudaSuccess) { \
            fprintf(stderr, "Kernel execution error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Utility function to print GPU device information
inline void print_gpu_info() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    printf("Number of CUDA devices: %d\n", device_count);
    
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Threads per Dimension: [%d, %d, %d]\n", 
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max Grid Size: [%d, %d, %d]\n", 
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  Memory Clock Rate: %.2f MHz\n", prop.memoryClockRate / 1000.0);
        printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth: %.2f GB/s\n", 
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    }
}

// Utility function to calculate optimal block size
inline int calculate_optimal_block_size(int problem_size, int max_threads_per_block = 1024) {
    // Start with a reasonable block size and adjust based on problem size
    int block_size = 256;
    
    // Ensure block size doesn't exceed problem size
    if (block_size > problem_size) {
        block_size = problem_size;
    }
    
    // Ensure block size doesn't exceed maximum threads per block
    if (block_size > max_threads_per_block) {
        block_size = max_threads_per_block;
    }
    
    return block_size;
}

// Utility function to calculate grid size
inline int calculate_grid_size(int problem_size, int block_size) {
    return (problem_size + block_size - 1) / block_size;
}

#endif // CUDA_COMMON_CUH
