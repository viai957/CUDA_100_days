#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>
#include <chrono>
#include <vector>
#include <random>

typedef int EL_TYPE;

__global__ void softmax_kernel(EL_TYPE* input, EL_TYPE* output, int length)
{
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
__global__ void cuda_matrix_multiply(EL_TYPE *OUT, EL_TYPE *A, EL_TYPE *B, int M, int N, int K)
{
    
}