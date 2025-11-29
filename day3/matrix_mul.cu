#include <stdio.h>
#include <sdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
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
        EL_TYPE sum = 0.0f;
        // For matrix mul A(MxK) x  B(KxN) -> C(MxN)
        for (int k = 0; k < NUM_COLS; k++)
        {
            size_t idx_a = static_cast<size_t>(row_idx * NUM_COLS + k); // A[row_idx][k]
            size_t idx_b = static_cast<size_t>(k * NUM_COLS + col_idx); // B[k][col_idx]
        }
        OUT[idx] = sum;
}