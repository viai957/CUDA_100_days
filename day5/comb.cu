#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>
#include <chrono>
#include <vector>
#include <random>

typedef int EL_TYPE;

__global__ void softmax_kernel(EL_TYPE* input, EL_TYPE* output, int length){
    // Find maximum value for numerical stability
    EL_TYPE max_val = input[0];
    for (int i = 1; i < length; i++){
        if (input[i] > max_val){
            max_val = input[i];
        }
    }


    // Compute exponentials and sum
    EL_TYPE sum = 0.0f;
    for (int i = 0; i < length; i++){
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    // Normalize
    for (int i = 0; i < length; i++){
        output[i] /= sum;
    }
}

// Matrix multiplication
__global__ void cuda_matrix_multiplication(EL_TYPE *OUT, EL_TYPE *A, EL_TYPE *B, int M, int N, int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N){
        EL_TYPE sum = 0.0f;
        for (int k = 0; k < K; k++){
            size_t a_index = static_cast<size_t>(row) * K + k;
            size_t b_index = static_cast<size_t>(k) * N + col;
            sum += A[a_index] * B[b_index];
        }
        size_t out_index = static_cast<size_t>(row) * N + col;
        OUT[out_index] = sum;
    }
}

// matrix transpose
__global__ void matrix_transpose(EL_TYPE* in, EL_TYPE* out, int rows, int cols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols){
        int in_idx = row * cols + col;
        int out_idx = col * rows + row;
        out[out_idx] = in[in_idx];
    }
}

// Scaled dot product attention
__global__ void attention_kernel(EL_TYPE* Q, EL_TYPE* K, EL_TYPE* V, EL_TYPE* output,
    int seq_len, int head_dim, int num_heads, int head_idx, EL_TYPE scale_factor)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;
    int elements_per_thread = (seq_len * seq_len + total_threads - 1) / total_threads;

    int start_idx = tid * elements_per_thread;
    
}
