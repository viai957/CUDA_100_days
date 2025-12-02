#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>
#include <chrono>
#include <vector>
#include <random>
#include "../cuda_common.cuh"

const int array_size = 10;

// Allocate host memory for the array
// The amount of memory allocated is equal to the length of the array times the size of the int 
int* host_array = (int*)malloc(array_size * sizeof(int));

// Initialize the array with random values
for (int i = 0; i < array_size; i++){
    host_array[i] = rand() % 100;
}

// Allocate device memory for the array
int* device_array;
CUDA_CHECK(cudaMalloc((void**)&device_array, array_size * sizeof(int)));

// Copy the array to device memeory
CUDA_CHECK(cudaMemcpy(device_array, host_array, array_size * sizeof(int), cudaMemcpyHostToDevice));

// Define the kernel
array_increment<<<1, array_size>>>(device_array);

// Copy the array back to the host
CUDA_CHECK(cudaMemcpy(host_array, device_array, array_size * sizeof(int), cudaMemcpyDeviceToHost));

void printArray(int* array, int arraySize){
    printf("[");
    for (int i = 0; i < arraySize; i++){
        printf("%d", array[i]);
        if (i < arraySize - 1){
            printf(", ");
        }
    }
    printf("]\n");
}

int main(){
    const int array_size = 10;

    // Allocate host memory for the input array
    int* array = (int*)malloc(array_size * sizeof(int));

    // Copy the input array from host to GPU memory
    cudaMemcpy(device_array, array, array_size * sizeof(int), cudaMemcpyHostToDevice);

    // Define the kernel
    array_increment<<<1, array_size>>>(device_array);

    // Copy the result array from GPU memory back to host memory
    cudaMemcpy(array, device_array, array_size * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the original and incremented arrays
    printf("Original array: ");
    printArray(array, array_size);
    printf("Incremented array: ");
    printArray(array, array_size);

    // Free the memory
    free(array);
    cudaFree(device_array);
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>
#include <chrono>
#include <vector>
#include <random>
#include "../cuda_common.cuh"

// CUDA kernel to increment each elment of the array by 1
__global__ void array_increment(int* array, int arraySize){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < arraySize){
        array[idx] = array[idx] + 1;
    }
}

// Function to print the array contents
void printArray()
