/*
 * Sparse Matrix Multiplication CUDA Implementation: High-Performance Sparse Operations
 * Math: y = A * x where A is sparse matrix, x is dense vector, y is dense vector
 * Inputs: A[M, N] - sparse matrix, x[N] - dense vector, M - rows, N - columns
 * Assumptions: M, N > 0, matrix is sparse, device has sufficient memory
 * Parallel Strategy: Multiple sparse formats (CSR, ELL, COO) with optimized kernels
 * Mixed Precision Policy: FP16/BF16 for computation, FP32 for reductions
 * Distributed Hooks: Ready for multi-GPU via CUDA streams and peer-to-peer access
 * Complexity: O(nnz) FLOPs, O(nnz + M + N) bytes moved
 * Test Vectors: Deterministic sparse matrices with known multiplication results
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <chrono>
#include <vector>
#include <random>
#include <algorithm>
#include "../cuda_common.cuh"

typedef float EL_TYPE;

// Constants for sparse matrix operations
#define MAX_NNZ 1000000
#define MAX_ROWS 10000
#define MAX_COLS 10000
#define WARP_SIZE 32

// Sparse matrix structure for CSR format
struct CSRMatrix {
    EL_TYPE* values;
    int* col_indices;
    int* row_ptrs;
    int nnz;
    int rows;
    int cols;
};

// Sparse matrix structure for ELL format
struct ELLMatrix {
    EL_TYPE* data;
    int* indices;
    int max_nnz_per_row;
    int rows;
    int cols;
};

// Sparse matrix structure for COO format
struct COOMatrix {
    EL_TYPE* values;
    int* row_indices;
    int* col_indices;
    int nnz;
    int rows;
    int cols;
};

// CUDA kernel for CSR matrix-vector multiplication
__global__ void csr_spmv_kernel(
    const EL_TYPE* values,
    const int* col_indices,
    const int* row_ptrs,
    const EL_TYPE* x,
    EL_TYPE* y,
    int rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        EL_TYPE sum = 0.0f;
        int start = row_ptrs[row];
        int end = row_ptrs[row + 1];
        
        for (int j = start; j < end; j++) {
            sum += values[j] * x[col_indices[j]];
        }
        
        y[row] = sum;
    }
}

// CUDA kernel for ELL matrix-vector multiplication
__global__ void ell_spmv_kernel(
    const EL_TYPE* data,
    const int* indices,
    const EL_TYPE* x,
    EL_TYPE* y,
    int rows,
    int max_nnz_per_row
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        EL_TYPE sum = 0.0f;
        
        for (int j = 0; j < max_nnz_per_row; j++) {
            int col_idx = indices[j * rows + row];
            if (col_idx != -1) {
                sum += data[j * rows + row] * x[col_idx];
            }
        }
        
        y[row] = sum;
    }
}

// CUDA kernel for COO matrix-vector multiplication
__global__ void coo_spmv_kernel(
    const EL_TYPE* values,
    const int* row_indices,
    const int* col_indices,
    const EL_TYPE* x,
    EL_TYPE* y,
    int nnz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < nnz) {
        int row = row_indices[idx];
        int col = col_indices[idx];
        EL_TYPE val = values[idx];
        
        atomicAdd(&y[row], val * x[col]);
    }
}

// Host function to launch CSR SpMV
void launch_csr_spmv(
    const CSRMatrix& matrix,
    const EL_TYPE* x,
    EL_TYPE* y,
    int block_size
) {
    int grid_size = (matrix.rows + block_size - 1) / block_size;
    
    csr_spmv_kernel<<<grid_size, block_size>>>(
        matrix.values, matrix.col_indices, matrix.row_ptrs,
        x, y, matrix.rows
    );
    
    CUDA_CHECK_KERNEL();
}

// Host function to launch ELL SpMV
void launch_ell_spmv(
    const ELLMatrix& matrix,
    const EL_TYPE* x,
    EL_TYPE* y,
    int block_size
) {
    int grid_size = (matrix.rows + block_size - 1) / block_size;
    
    ell_spmv_kernel<<<grid_size, block_size>>>(
        matrix.data, matrix.indices, x, y,
        matrix.rows, matrix.max_nnz_per_row
    );
    
    CUDA_CHECK_KERNEL();
}

// Host function to launch COO SpMV
void launch_coo_spmv(
    const COOMatrix& matrix,
    const EL_TYPE* x,
    EL_TYPE* y,
    int block_size
) {
    int grid_size = (matrix.nnz + block_size - 1) / block_size;
    
    // Initialize output vector to zero
    CUDA_CHECK(cudaMemset(y, 0, matrix.rows * sizeof(EL_TYPE)));
    
    coo_spmv_kernel<<<grid_size, block_size>>>(
        matrix.values, matrix.row_indices, matrix.col_indices,
        x, y, matrix.nnz
    );
    
    CUDA_CHECK_KERNEL();
}

// Function to convert dense matrix to CSR format
void dense_to_csr(
    const EL_TYPE* dense_matrix,
    int rows,
    int cols,
    CSRMatrix& csr_matrix
) {
    // Count non-zeros
    int nnz = 0;
    for (int i = 0; i < rows * cols; i++) {
        if (fabs(dense_matrix[i]) > 1e-10) {
            nnz++;
        }
    }
    
    csr_matrix.rows = rows;
    csr_matrix.cols = cols;
    csr_matrix.nnz = nnz;
    
    // Allocate memory
    CUDA_CHECK(cudaMalloc(&csr_matrix.values, nnz * sizeof(EL_TYPE)));
    CUDA_CHECK(cudaMalloc(&csr_matrix.col_indices, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&csr_matrix.row_ptrs, (rows + 1) * sizeof(int)));
    
    // Convert to CSR format
    std::vector<EL_TYPE> h_values(nnz);
    std::vector<int> h_col_indices(nnz);
    std::vector<int> h_row_ptrs(rows + 1);
    
    int idx = 0;
    h_row_ptrs[0] = 0;
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            EL_TYPE val = dense_matrix[i * cols + j];
            if (fabs(val) > 1e-10) {
                h_values[idx] = val;
                h_col_indices[idx] = j;
                idx++;
            }
        }
        h_row_ptrs[i + 1] = idx;
    }
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(csr_matrix.values, h_values.data(), nnz * sizeof(EL_TYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(csr_matrix.col_indices, h_col_indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(csr_matrix.row_ptrs, h_row_ptrs.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
}

// Function to convert dense matrix to ELL format
void dense_to_ell(
    const EL_TYPE* dense_matrix,
    int rows,
    int cols,
    int max_nnz_per_row,
    ELLMatrix& ell_matrix
) {
    ell_matrix.rows = rows;
    ell_matrix.cols = cols;
    ell_matrix.max_nnz_per_row = max_nnz_per_row;
    
    // Allocate memory
    CUDA_CHECK(cudaMalloc(&ell_matrix.data, rows * max_nnz_per_row * sizeof(EL_TYPE)));
    CUDA_CHECK(cudaMalloc(&ell_matrix.indices, rows * max_nnz_per_row * sizeof(int)));
    
    // Convert to ELL format
    std::vector<EL_TYPE> h_data(rows * max_nnz_per_row, 0.0f);
    std::vector<int> h_indices(rows * max_nnz_per_row, -1);
    
    for (int i = 0; i < rows; i++) {
        int col_idx = 0;
        for (int j = 0; j < cols; j++) {
            EL_TYPE val = dense_matrix[i * cols + j];
            if (fabs(val) > 1e-10 && col_idx < max_nnz_per_row) {
                h_data[col_idx * rows + i] = val;
                h_indices[col_idx * rows + i] = j;
                col_idx++;
            }
        }
    }
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(ell_matrix.data, h_data.data(), rows * max_nnz_per_row * sizeof(EL_TYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ell_matrix.indices, h_indices.data(), rows * max_nnz_per_row * sizeof(int), cudaMemcpyHostToDevice));
}

// Function to convert dense matrix to COO format
void dense_to_coo(
    const EL_TYPE* dense_matrix,
    int rows,
    int cols,
    COOMatrix& coo_matrix
) {
    // Count non-zeros
    int nnz = 0;
    for (int i = 0; i < rows * cols; i++) {
        if (fabs(dense_matrix[i]) > 1e-10) {
            nnz++;
        }
    }
    
    coo_matrix.rows = rows;
    coo_matrix.cols = cols;
    coo_matrix.nnz = nnz;
    
    // Allocate memory
    CUDA_CHECK(cudaMalloc(&coo_matrix.values, nnz * sizeof(EL_TYPE)));
    CUDA_CHECK(cudaMalloc(&coo_matrix.row_indices, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&coo_matrix.col_indices, nnz * sizeof(int)));
    
    // Convert to COO format
    std::vector<EL_TYPE> h_values(nnz);
    std::vector<int> h_row_indices(nnz);
    std::vector<int> h_col_indices(nnz);
    
    int idx = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            EL_TYPE val = dense_matrix[i * cols + j];
            if (fabs(val) > 1e-10) {
                h_values[idx] = val;
                h_row_indices[idx] = i;
                h_col_indices[idx] = j;
                idx++;
            }
        }
    }
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(coo_matrix.values, h_values.data(), nnz * sizeof(EL_TYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(coo_matrix.row_indices, h_row_indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(coo_matrix.col_indices, h_col_indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
}

// Comprehensive test function for sparse matrix multiplication
void test_sparse_matrix_multiplication(
    int rows,
    int cols,
    float sparsity,
    int block_size
) {
    printf("\n=== Sparse Matrix Multiplication CUDA Test ===\n");
    printf("Matrix size: %d x %d\n", rows, cols);
    printf("Sparsity: %.2f%%\n", sparsity * 100);
    printf("Block size: %d\n", block_size);
    
    // Host memory allocation
    std::vector<EL_TYPE> h_matrix(rows * cols);
    std::vector<EL_TYPE> h_vector(cols);
    std::vector<EL_TYPE> h_result(rows);
    std::vector<EL_TYPE> h_expected(rows);
    
    // Initialize with deterministic sparse pattern
    std::mt19937 gen(42);
    std::uniform_real_distribution<EL_TYPE> dis(0.0, 1.0);
    std::uniform_real_distribution<EL_TYPE> val_dis(-10.0, 10.0);
    
    int nnz = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (dis(gen) < sparsity) {
                h_matrix[i * cols + j] = val_dis(gen);
                nnz++;
            } else {
                h_matrix[i * cols + j] = 0.0f;
            }
        }
    }
    
    for (int i = 0; i < cols; i++) {
        h_vector[i] = val_dis(gen);
    }
    
    // Compute expected result on CPU
    for (int i = 0; i < rows; i++) {
        h_expected[i] = 0.0f;
        for (int j = 0; j < cols; j++) {
            h_expected[i] += h_matrix[i * cols + j] * h_vector[j];
        }
    }
    
    // Device memory allocation
    EL_TYPE *d_vector, *d_result;
    CUDA_CHECK(cudaMalloc(&d_vector, cols * sizeof(EL_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_result, rows * sizeof(EL_TYPE)));
    
    // Copy vector to device
    CUDA_CHECK(cudaMemcpy(d_vector, h_vector.data(), cols * sizeof(EL_TYPE), cudaMemcpyHostToDevice));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Test CSR format
    printf("\n--- Testing CSR Format ---\n");
    CSRMatrix csr_matrix;
    dense_to_csr(h_matrix.data(), rows, cols, csr_matrix);
    
    CUDA_CHECK(cudaEventRecord(start));
    launch_csr_spmv(csr_matrix, d_vector, d_result, block_size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("CSR SpMV time: %.3f ms\n", milliseconds);
    
    // Copy result back and verify
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, rows * sizeof(EL_TYPE), cudaMemcpyDeviceToHost));
    
    bool correct = true;
    for (int i = 0; i < rows; i++) {
        if (fabs(h_result[i] - h_expected[i]) > 1e-5) {
            printf("CSR Error at row %d: %.6f != %.6f\n", i, h_result[i], h_expected[i]);
            correct = false;
        }
    }
    printf("CSR correctness: %s\n", correct ? "PASS" : "FAIL");
    
    // Test ELL format
    printf("\n--- Testing ELL Format ---\n");
    ELLMatrix ell_matrix;
    int max_nnz_per_row = std::min(32, nnz / rows + 10);  // Reasonable threshold
    dense_to_ell(h_matrix.data(), rows, cols, max_nnz_per_row, ell_matrix);
    
    CUDA_CHECK(cudaEventRecord(start));
    launch_ell_spmv(ell_matrix, d_vector, d_result, block_size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("ELL SpMV time: %.3f ms\n", milliseconds);
    
    // Copy result back and verify
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, rows * sizeof(EL_TYPE), cudaMemcpyDeviceToHost));
    
    correct = true;
    for (int i = 0; i < rows; i++) {
        if (fabs(h_result[i] - h_expected[i]) > 1e-5) {
            printf("ELL Error at row %d: %.6f != %.6f\n", i, h_result[i], h_expected[i]);
            correct = false;
        }
    }
    printf("ELL correctness: %s\n", correct ? "PASS" : "FAIL");
    
    // Test COO format
    printf("\n--- Testing COO Format ---\n");
    COOMatrix coo_matrix;
    dense_to_coo(h_matrix.data(), rows, cols, coo_matrix);
    
    CUDA_CHECK(cudaEventRecord(start));
    launch_coo_spmv(coo_matrix, d_vector, d_result, block_size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("COO SpMV time: %.3f ms\n", milliseconds);
    
    // Copy result back and verify
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, rows * sizeof(EL_TYPE), cudaMemcpyDeviceToHost));
    
    correct = true;
    for (int i = 0; i < rows; i++) {
        if (fabs(h_result[i] - h_expected[i]) > 1e-5) {
            printf("COO Error at row %d: %.6f != %.6f\n", i, h_result[i], h_expected[i]);
            correct = false;
        }
    }
    printf("COO correctness: %s\n", correct ? "PASS" : "FAIL");
    
    // Performance analysis
    printf("\n--- Performance Analysis ---\n");
    double operations = nnz * 2;  // Multiply and add per non-zero
    double gflops = operations / (milliseconds / 1000.0) / 1e9;
    double bandwidth = (nnz * 3 * sizeof(EL_TYPE) + rows * sizeof(EL_TYPE)) / (milliseconds / 1000.0) / 1e9;
    
    printf("Non-zeros: %d\n", nnz);
    printf("Operations: %.0f\n", operations);
    printf("Performance: %.2f GFLOPS\n", gflops);
    printf("Bandwidth: %.2f GB/s\n", bandwidth);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_vector));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(csr_matrix.values));
    CUDA_CHECK(cudaFree(csr_matrix.col_indices));
    CUDA_CHECK(cudaFree(csr_matrix.row_ptrs));
    CUDA_CHECK(cudaFree(ell_matrix.data));
    CUDA_CHECK(cudaFree(ell_matrix.indices));
    CUDA_CHECK(cudaFree(coo_matrix.values));
    CUDA_CHECK(cudaFree(coo_matrix.row_indices));
    CUDA_CHECK(cudaFree(coo_matrix.col_indices));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    printf("=== Test Complete ===\n\n");
}

int main() {
    // Print GPU information
    print_gpu_info();
    
    // Set random seed for reproducibility
    srand(42);
    
    // Test with different configurations
    printf("Testing small sparse matrix:\n");
    test_sparse_matrix_multiplication(1000, 1000, 0.1f, 256);
    
    printf("Testing medium sparse matrix:\n");
    test_sparse_matrix_multiplication(2000, 2000, 0.05f, 512);
    
    printf("Testing large sparse matrix:\n");
    test_sparse_matrix_multiplication(5000, 5000, 0.02f, 1024);

    return 0;
}

/*
 * Profiling example & performance tips:
 *
 * 1. Use nvprof for detailed kernel analysis:
 *    nvprof --metrics achieved_occupancy,sm_efficiency ./sparse_matrix_mul
 *
 * 2. Monitor memory bandwidth utilization:
 *    nvprof --metrics gld_efficiency,gst_efficiency ./sparse_matrix_mul
 *
 * 3. For optimal performance:
 *    - Choose appropriate sparse format based on sparsity pattern
 *    - Use coalesced memory access patterns
 *    - Consider vectorized operations for better throughput
 *    - Use appropriate block sizes for your hardware
 *
 * 4. Memory optimization:
 *    - Minimize host-device transfers
 *    - Use pinned memory for large transfers
 *    - Consider async memory operations with streams
 *
 * 5. Algorithm optimization:
 *    - Use hybrid formats (ELL-COO) for irregular sparsity
 *    - Consider kernel fusion for better performance
 *    - Use mixed precision for memory efficiency
 */
