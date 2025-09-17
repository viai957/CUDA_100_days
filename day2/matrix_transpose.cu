#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>
#include <chrono>
#include <vector>
#include <random>
#include "../cuda_common.cuh"

__global__ void matrix_transpose(int* in, int* out){
    // Select an element from the input matrix by using the threadIdx.x and threadIdx.y values.
    // blockDim.x contains the number of rows in the 2D Thread Block
    const int threadIdx = threadIdx.x + threadIdx.y * blockDim.x;

    // Select the corresponding position in the output matrix
    // blockDim.y contains the number of columns in the 2D Thread Block
    const int outIdx = threadIdx.y + threadIdx.x * blockDim.y;
    out[outIdx] = in[threadIdx];
}

int main(){
    // Allocate host & GPU memory; Copy the input array to GPU memory
    int* in, *out;
    int* d_in, *d_out;
    CUDA_CHECK(cudaMalloc((void**)&d_in, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc((void**)&d_out, sizeof(int) * N));
    CUDA_CHECK(cudaMemcpy(d_in, in, sizeof(int) * N, cudaMemcpyHostToDevice));

    // Define the launch grid
    dim3 numThreadsPerBlock(rows, cols);
    transpose<<<1, numThreadsPerBlock>>>(d_in, d_out);

    // Release host and GPU memory
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(in);
    free(out);
    return 0;
}