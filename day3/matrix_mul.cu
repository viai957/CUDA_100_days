#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>
#include "cuda_common.cuh"


typedef float EL_TYPE;

__global__ void cuda_matrix_mul(EL_TYPE *OUT, EL_TYPE *A, EL_TYPE *B, int NUM_ROWS, int NUM_COLS)
{
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx < NUM_ROWS && col_idx < NUM_COLS)
    {
        size_t idx = static_cast<size_t>(row_idx * NUM_COLS + col_idx); // OUT[row_idx][col_idx]
        OUT[idx] = 0.00f;
        for (int k = 0; k < NUM_COLS; k++)
        {
            size_t idx_k = static_cast<size_t>(row_idx * NUM_COLS + k); // A[row_idx][k]
            size_t idx_b = static_cast<size_t>(k * NUM_COLS + col_idx); // B[k][col_idx]
            OUT[idx] += A[idx_k] * B[idx_b];
        }
    }
}

void test_matrix_mul(int NUM_ROWS, int NUM_COLS, int ROWS_block_size, int COLS_block_size)
{
    EL_TYPE *A, *B, *OUT;
    EL_TYPE *d_A, *d_B, *d_OUT;

    // Allocate the matrices on the host device: CPU
    A = (EL_TYPE *)malloc(sizeof(EL_TYPE) * NUM_ROWS * NUM_COLS);
    B = (EL_TYPE *)malloc(sizeof(EL_TYPE) * NUM_COLS * NUM_COLS);
    OUT = (EL_TYPE *)malloc(sizeof(EL_TYPE) * NUM_ROWS * NUM_COLS);

    // Initialize the matrices with random values
    for (int i = 0; i < NUM_ROWS; i++){
        for (int j = 0; j < NUM_COLS; j++){
            size_t idx = static_cast<size_t>(i) * NUM_COLS + j;
            A[idx] = (EL_TYPE)(rand() % 100) / 10.0f; // Scale to avoid overflow
        }
    }
    
    for (int i = 0; i < NUM_COLS; i++){
        for (int j = 0; j < NUM_COLS; j++){
            size_t idx = static_cast<size_t>(i) * NUM_COLS + j;
            B[idx] = (EL_TYPE)(rand() % 100) / 10.0f; // Scale to avoid overflow
        }
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_A, sizeof(EL_TYPE) * NUM_ROWS * NUM_COLS));
    CUDA_CHECK(cudaMalloc((void **)&d_B, sizeof(EL_TYPE) * NUM_COLS * NUM_COLS));
    CUDA_CHECK(cudaMalloc((void **)&d_OUT, sizeof(EL_TYPE) * NUM_ROWS * NUM_COLS));

    // Transfer the matrices to the device
    CUDA_CHECK(cudaMemcpy(d_A, A, sizeof(EL_TYPE) * NUM_ROWS * NUM_COLS, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, sizeof(EL_TYPE) * NUM_COLS * NUM_COLS, cudaMemcpyHostToDevice));

    cudaEvent_t start_kernel, stop_kernel;
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));

    CUDA_CHECK(cudaEventRecord(start_kernel));

    // Define the launch grid
    int num_blocks_rows = (NUM_ROWS + ROWS_block_size - 1) / ROWS_block_size; // ceil(NUM_ROWS / ROWS_block_size)
    int num_blocks_cols = (NUM_COLS + COLS_block_size - 1) / COLS_block_size; // ceil(NUM_COLS / COLS_block_size)
    printf("Matrix Mul - NUM_ROWS: %d, NUM_COLS: %d will be processed by %d x %d blocks of size %d x %d\n", NUM_ROWS, NUM_COLS, num_blocks_rows, num_blocks_cols, ROWS_block_size, COLS_block_size);
    dim3 grid(num_blocks_cols, num_blocks_rows, 1);
    dim3 block(COLS_block_size, ROWS_block_size, 1);
    // Run the kernel
    cuda_matrix_mul<<<grid, block>>>(d_OUT, d_A, d_B, NUM_ROWS, NUM_COLS);

    // Check for launch errors
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaEventRecord(stop_kernel));
    CUDA_CHECK(cudaEventSynchronize(stop_kernel));

    // Calculate elapsed milliseconds
    float milliseconds_kernel = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds_kernel, start_kernel, stop_kernel));
    printf("Matrix Mul - elapsed time: %f ms\n", milliseconds_kernel);

    // Copy back the result from the device to the host
    CUDA_CHECK(cudaMemcpy(OUT, d_OUT, sizeof(EL_TYPE) * NUM_ROWS * NUM_COLS, cudaMemcpyDeviceToHost));

    // Free the memory on the device
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_OUT));

    // Time the operation 
    struct timeval start_check, end_check;
    gettimeofday(&start_check, NULL);

    // CPU verification: compute expected result
    EL_TYPE *expected = (EL_TYPE *)malloc(sizeof(EL_TYPE) * NUM_ROWS * NUM_COLS);
    for (int i = 0; i < NUM_ROWS; i++){
        for (int j = 0; j < NUM_COLS; j++){
            size_t idx = static_cast<size_t>(i) * NUM_COLS + j;
            expected[idx] = 0.0f;
            for (int k = 0; k < NUM_COLS; k++){
                size_t idx_a = static_cast<size_t>(i) * NUM_COLS + k;
                size_t idx_b = static_cast<size_t>(k) * NUM_COLS + j;
                expected[idx] += A[idx_a] * B[idx_b];
            }
        }
    }

    // Verify results with tolerance for floating point
    float tolerance = 1e-5f;
    int errors = 0;
    for (int i = 0; i < NUM_ROWS * NUM_COLS; i++){
        if (fabs(OUT[i] - expected[i]) > tolerance){
            if (errors < 10) { // Print first 10 errors
                printf("Error at index %d: GPU=%.6f, CPU=%.6f, diff=%.6f\n", i, OUT[i], expected[i], fabs(OUT[i] - expected[i]));
            }
            errors++;
        }
    }

    // Calculate elapsed time
    gettimeofday(&end_check, NULL);
    float elapsed = (end_check.tv_sec - start_check.tv_sec) * 1000.0 + (end_check.tv_usec - start_check.tv_usec) / 1000.0;
    printf("Matrix Mul - Check elapsed time: %f ms\n", elapsed);
    
    if (errors == 0){
        printf("Matrix Mul - result OK\n");
    } else {
        printf("Matrix Mul - result FAILED: %d errors found\n", errors);
        exit(1);
    }

    // Free the memory on the host
    free(A);
    free(B);
    free(OUT);
    free(expected);
}

int main()
{   
    // set your seed
    srand(0);
    
    test_matrix_mul(1024, 1024, 32, 32);
    return 0;
}