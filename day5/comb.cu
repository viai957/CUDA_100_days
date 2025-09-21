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

    // Compute exponentials and use 
}