#inlcude <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>
#include "cuda_common.cuh"

typedef float EL_TYPE;

__global__ void cuda_matrix_add(EL_TYPE *C, EL_TYPE *A, EL_TYPE *B, int M, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < M && j < N)
    {
        C[i * N + j] = A[i * N + j] + B[i * N + j];
    }
}
