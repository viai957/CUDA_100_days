/*
 * Day 2 - Matrix Multiply (CUDA): Tiled GEMM (row-major)
 *
 * Math:
 *   C[m, n] = sum_{k=0..K-1} A[m, k] * B[k, n]
 *
 * Inputs / Outputs:
 *   A: float[M, K] row-major
 *   B: float[K, N] row-major
 *   C: float[M, N] row-major
 *
 * Assumptions:
 * - A, B, C are contiguous row-major buffers (leading dims = K, N, N respectively)
 * - M, N, K > 0
 *
 * Parallel Strategy:
 * - 2D grid of threadblocks; each block computes one TILE_M x TILE_N output tile
 * - Each thread computes one C element; shared-memory tiles A[TILE_M x TILE_K], B[TILE_K x TILE_N]
 *
 * Mixed Precision Policy:
 * - FP32 loads and FP32 accumulation (educational baseline)
 *
 * Complexity:
 * - FLOPs: ~2*M*N*K
 * - Bytes moved (naive lower bound): (M*K + K*N + M*N) * sizeof(float)
 *
 * Build:
 *   nvcc -O3 -lineinfo -std=c++17 day_2/matrix_mul.cu -o /tmp/matrix_mul
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_common.cuh"

using EL_TYPE = float;

static inline EL_TYPE frand01() {
    return (EL_TYPE)rand() / (EL_TYPE)RAND_MAX;
}

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void cuda_gemm_tiled(const EL_TYPE *__restrict__ A,
                               const EL_TYPE *__restrict__ B,
                               EL_TYPE *__restrict__ C,
                               int M, int N, int K,
                               int lda, int ldb, int ldc) {
    // Block tile origin
    const int block_row = (int)blockIdx.y * TILE_M;
    const int block_col = (int)blockIdx.x * TILE_N;

    // Thread coordinates within the tile
    const int ty = (int)threadIdx.y; // [0, TILE_M)
    const int tx = (int)threadIdx.x; // [0, TILE_N)

    // Global coordinates of the output element computed by this thread
    const int m = block_row + ty;
    const int n = block_col + tx;

    __shared__ EL_TYPE As[TILE_M][TILE_K];
    __shared__ EL_TYPE Bs[TILE_K][TILE_N];

    EL_TYPE acc = 0.0f;

    // Iterate over K dimension tiles
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Load A tile element As[ty][tx] when TILE_K == TILE_N (we use square 16x16x16 here)
        {
            const int a_k = k0 + tx;
            EL_TYPE a_val = 0.0f;
            if (m < M && a_k < K) {
                a_val = A[(size_t)m * lda + (size_t)a_k];
            }
            As[ty][tx] = a_val;
        }

        // Load B tile element Bs[ty][tx] when TILE_K == TILE_M
        {
            const int b_k = k0 + ty;
            EL_TYPE b_val = 0.0f;
            if (b_k < K && n < N) {
                b_val = B[(size_t)b_k * ldb + (size_t)n];
            }
            Bs[ty][tx] = b_val;
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk++) {
            acc += As[ty][kk] * Bs[kk][tx];
        }

        __syncthreads();
    }

    if (m < M && n < N) {
        C[(size_t)m * ldc + (size_t)n] = acc;
    }
}

static void cpu_gemm_ref(const EL_TYPE *A, const EL_TYPE *B, EL_TYPE *C, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            EL_TYPE acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += A[(size_t)m * K + (size_t)k] * B[(size_t)k * N + (size_t)n];
            }
            C[(size_t)m * N + (size_t)n] = acc;
        }
    }
}

static void test_gemm(int M, int N, int K, int tile) {
    assert(tile == 16 && "This Day-2 tiled kernel is wired for TILE=16 (TILE_M=TILE_N=TILE_K=16).");

    printf("GEMM: A[%d x %d] * B[%d x %d] = C[%d x %d]\n", M, K, K, N, M, N);

    const size_t bytes_A = (size_t)M * (size_t)K * sizeof(EL_TYPE);
    const size_t bytes_B = (size_t)K * (size_t)N * sizeof(EL_TYPE);
    const size_t bytes_C = (size_t)M * (size_t)N * sizeof(EL_TYPE);

    EL_TYPE *hA = (EL_TYPE *)malloc(bytes_A);
    EL_TYPE *hB = (EL_TYPE *)malloc(bytes_B);
    EL_TYPE *hC = (EL_TYPE *)malloc(bytes_C);
    EL_TYPE *hC_ref = (EL_TYPE *)malloc(bytes_C);
    assert(hA && hB && hC && hC_ref);

    for (int i = 0; i < M * K; i++) hA[i] = (frand01() - 0.5f) * 2.0f;
    for (int i = 0; i < K * N; i++) hB[i] = (frand01() - 0.5f) * 2.0f;

    EL_TYPE *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&dA, bytes_A));
    CUDA_CHECK(cudaMalloc((void **)&dB, bytes_B));
    CUDA_CHECK(cudaMalloc((void **)&dC, bytes_C));
    CUDA_CHECK(cudaMemcpy(dA, hA, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes_B, cudaMemcpyHostToDevice));

    dim3 block(16, 16, 1);
    dim3 grid((N + 16 - 1) / 16, (M + 16 - 1) / 16, 1);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < 5; i++) {
        cuda_gemm_tiled<16, 16, 16><<<grid, block>>>(dA, dB, dC, M, N, K, K, N, N);
    }
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaEventRecord(start));
    cuda_gemm_tiled<16, 16, 16><<<grid, block>>>(dA, dB, dC, M, N, K, K, N, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("CUDA kernel time: %.3f ms\n", ms);

    CUDA_CHECK(cudaMemcpy(hC, dC, bytes_C, cudaMemcpyDeviceToHost));

    // CPU reference + check
    cpu_gemm_ref(hA, hB, hC_ref, M, N, K);
    float max_abs_err = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float diff = fabsf((float)hC[i] - (float)hC_ref[i]);
        if (diff > max_abs_err) max_abs_err = diff;
    }
    printf("max|C - C_ref| = %.6e\n", max_abs_err);
    if (max_abs_err < 1e-3f) {
        printf("Result OK\n");
    } else {
        printf("Result MISMATCH\n");
    }

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    free(hA);
    free(hB);
    free(hC);
    free(hC_ref);
}

int main() {
    srand(0);

    test_gemm(128, 128, 128, 16);
    test_gemm(512, 512, 512, 16);
    test_gemm(1024, 1024, 512, 16);
    return 0;
}