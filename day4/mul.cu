 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <assert.h>
 #include <cuda.h>
 #include <cuda_runtime.h>
 #include <sys/time.h>
 #include "cuda_common.cuh"

 typedef float EL_TYPE;
 
__global__ void cuda_matrix_multiply(EL_TYPE *OUT, EL_TYPE *A, EL_TYPE *B, int M, int N, int K)
{
    int row = blockIdx.y * blockIdx.x + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        EL_TYPE sum = 0.0f;
        for (int k = 0; k < K; k++)
        {
            size_t a_index = static_cast<size_t>(row) * K + k; // A[row][k]
            size_t b_index = static_cast<size_t>(k) * N + col; // B[k][col]
            sum += A[a_index] * B[b_index];
        }
        size_t out_index = static_cast<size_t>(row) * N + col; // OUT[row][col]
        OUT[out_index] = sum;
    }
}

void test_matrix_multiply(int M, int N, int K, int ROWS_block_size, int COLS_block_size)
{
    EL_TYPE *A, *B, *OUT;
    EL_TYPE *d_A, *d_B, *d_OUT;

    // Allocate the matrices on the host device
    // A is M x K, B is K x N, OUT is M x N
    A = (EL_TYPE *)malloc(sizeof(EL_TYPE) * M * K);
    B = (EL_TYPE *)malloc(sizeof(EL_TYPE) * K * N);
    OUT = (EL_TYPE *)malloc(sizeof(EL_TYPE) * M * N);


    // Initialize the matrices with random values
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            size_t index = static_cast<size_t>(i) * K + j;
            A[index] = (rand() % 100) / 10.0f; // Scale down for better numerical stability
        }
    }

    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < N; j++)
        {
            size_t index = static_cast<size_t>(i) * N + j;
            B[index] = (rand() % 100) / 10.0f; // Scale down for better numerical stability
        }
    }

    // Allocate device meory
    CUDA_CHECK(cudaMalloc((void **)&d_A, sizeof(EL_TYPE) * M * K));
    CUDA_CHECK(cudaMalloc((void **)&d_B, sizeof(EL_TYPE) * K * N));
    CUDA_CHECK(cudaMalloc((void **)&d_OUT, sizeof(EL_TYPE) * M * N));

    // Transfer the matrices to the device 
    CUDA_CHECK(cudaMemcpy(d_A, A, sizeof(EL_TYPE) * M * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, sizeof(EL_TYPE) * K * N, cudaMemcpyHostToDevice));

    cudaEvent_t start_kernel, stop_kernel;
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));

    CUDA_CHECK(cudaEventRecord(start_kernel));

    // Define the launch grid
    int num_blocks_ROWS = (M + ROWS_block_size - 1) / ROWS_block_size; // ceil(M / ROWS_block_size)
    int num_blocks_COLS = (N + COLS_block_size - 1) / COLS_block_size; // ceil(N / COLS_block_size)
    printf("Matrix Multiply - A: %dx%d, B: %dx%d, OUT: %dx%d will be processed by (%d x %d) blocks of size (%d x %d)\n", 
           M, K, K, N, M, N, num_blocks_ROWS, num_blocks_COLS, ROWS_block_size, COLS_block_size);
    
    dim3 grid(num_blocks_COLS, num_blocks_ROWS, 1);
    dim3 block(COLS_block_size, ROWS_block_size, 1);

    // Run the kernel
    cuda_matrix_multiply<<<grid, block>>>(d_OUT, d_A, d_B, M, N, K);

    // Check for launch errors
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaEventRecord(stop_kernel));
    CUDA_CHECK(cudaEventSynchronize(stop_kernel));

    // Calculate elapsed milliseconds
    float milliseconds_kernel = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds_kernel, start_kernel, stop_kernel));
    printf("Matrix Multiply - Elapsed time: %f ms\n", milliseconds_kernel);

    // Copy back the result from the device to the host
    CUDA_CHECK(cudaMemcpy(OUT, d_OUT, sizeof(EL_TYPE) * M * N, cudaMemcpyDeviceToHost));
    
    // Free the memory on the device
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_OUT));

    // Time the operation 
    struct timeval start_check, end_check;
    gettimeofday(&start_check, NULL);
    
    // Verify the result using CPU computation
    bool error_found = false;
    for (int i = 0; i < M && !error_found; i++)
    {
        for (int j = 0; j < N && !error_found; j++)
        {
            
            EL_TYPE expected = 0.0f;
            for (int k = 0; k < K; k++)
            {
                size_t a_index = static_cast<size_t>(i) * K + k;
                size_t b_index = static_cast<size_t>(k) * N + j;
                expected += A[a_index] * B[b_index];
            }
        }
    }

    // Calculate elapsed time
    gettimeofday(&end_check, NULL);

    // Calculate elapsed time
    float elapsed_time = (end_check.tv_sec - start_check.tv_sec) * 1000.0f + (end_check.tv_usec - start_check.tv_usec) / 1000.0f;
    printf("Matrix Multiply - Elapsed time: %f ms\n", elapsed_time);
    
    // Verify the result using CPU computation
    bool error_found = false;
    for (int i = 0; i < M && !error_found; i++)
    {
        for (int j = 0; j < N && !error_found; j++)
        {
            EL_TYPE expected = 0.0f;
            for (int k = 0; k < K; k++)
            {
                size_t a_index = static_cast<size_t>(i) * K + k;
                size_t b_index = static_cast<size_t>(k) * N + j;
                expected += A[a_index] * B[b_index];
            }
            size_t out_index = static_cast<size_t>(i) * N + j;
            float diff = fabsf(OUT[out_index] - expected);
            if (diff > 1e-5f)  // Allow for small floating point errors
            {
                printf("Error at index (%d, %d): %.6f != %.6f (diff: %.6f)\n", 
                       i, j, OUT[out_index], expected, diff);
                error_found = true;
            }
        }
    }

    // Calculate elapsed time
    gettimeofday(&end_check, NULL);

    // Calculate elapsed time
    float elapsed_time = (end_check.tv_sec - start_check.tv_sec) * 1000.0f + (end_check.tv_usec - start_check.tv_usec) / 1000.0f;
    printf("Matrix Multiply - Elapsed time: %f ms\n", elapsed_time);
    
    if (!error_found)
        printf("Matrix Multiply - Result OK\n");

    // Free the memory on the host
    free(A);
    free(B);
    free(OUT);
    return 0;
}