"""
Matrix Multiplication Triton Implementation (Day 3): High-Performance GEMM on 2D Grids
Math: C[i, j] = Σ(A[i, k] * B[k, j]) for k ∈ [0, K) where A[M, K], B[K, N] -> C[M, N]
Inputs / Outputs: A[M, K], B[K, N] -> C[M, N] contiguous row-major tensors
Assumptions: M, K, N > 0; tensors contiguous on CUDA device; shapes compatible for multiplication
Parallel Strategy: 2D grid where each Triton program processes a BLOCK_M × BLOCK_N tile with K reduction
Mixed Precision Policy: FP16/BF16 storage allowed, accumulation in FP32 for accuracy; FP32 recommended for validation
Distributed Hooks: Can be wrapped with data-parallel all-reduce on gradients or outputs
Complexity: O(MNK) FLOPs, O((M*K + K*N + M*N) * sizeof(dtype)) bytes moved
Test Vectors: Deterministic random matrices for sizes like 32×32, 512×512 with max|Δ| < 1e-4 in FP32
"""

import math
import time
from typing import List, Dict, Any, Optional

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matrix_mul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    2D tiled matrix multiplication kernel with K reduction.
    Memory layout: row-major (stride_am = K, stride_ak = 1, stride_bk = N, stride_bn = 1, stride_cm = N, stride_cn = 1).
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        
        mask_a = (offs_m < M) & (k_offs < K)[None, :]
        mask_b = (k_offs < K)[:, None] & (offs_n < N)[None, :]
        
        a = tl.load(
            a_ptr + offs_m * stride_am + k_offs * stride_ak,
            mask=mask_a,
            other=0.0,
        )
        b = tl.load(
            b_ptr + k_offs * stride_bk + offs_n * stride_bn,
            mask=mask_b,
            other=0.0,
        )
        
        accumulator += tl.dot(a, b)

    mask_c = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    tl.store(
        c_ptr + offs_m * stride_cm + offs_n * stride_cn,
        accumulator.to(c_ptr.dtype.element_ty),
        mask=mask_c,
    )


def _validate_inputs(a: torch.Tensor, b: torch.Tensor) -> (int, int, int):
    assert a.device.type == "cuda", "Inputs must be on CUDA device"
    assert b.device.type == "cuda", "Inputs must be on CUDA device"
    assert a.dim() == 2, "Input A must be 2D tensor [M, K]"
    assert b.dim() == 2, "Input B must be 2D tensor [K, N]"
    assert a.dtype == b.dtype, "Inputs must share dtype"
    assert a.is_contiguous(), "Input A must be contiguous"
    assert b.is_contiguous(), "Input B must be contiguous"
    M, K_a = a.shape
    K_b, N = b.shape
    assert K_a == K_b, f"Inner dimensions must match: A has {K_a}, B has {K_b}"
    assert M > 0 and N > 0 and K_a > 0, "Matrix dimensions must be > 0"
    return M, N, K_a


def matrix_mul_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    block_m: int = 32,
    block_n: int = 32,
    block_k: int = 16,
) -> torch.Tensor:
    """
    Public API: matrix multiplication using Triton kernel.
    Computes C = A @ B where A is [M, K] and B is [K, N].
    """
    M, N, K = _validate_inputs(a, b)
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = (
        triton.cdiv(M, block_m),
        triton.cdiv(N, block_n),
    )

    stride_am = a.stride(0)
    stride_ak = a.stride(1)
    stride_bk = b.stride(0)
    stride_bn = b.stride(1)
    stride_cm = c.stride(0)
    stride_cn = c.stride(1)

    matrix_mul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return c


class MatrixMulModule(torch.nn.Module):
    """
    PyTorch nn.Module wrapper to plug Day 3 matrix mul into larger models.
    """

    def __init__(self, block_m: int = 32, block_n: int = 32, block_k: int = 16):
        super().__init__()
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return matrix_mul_triton(a, b, block_m=self.block_m, block_n=self.block_n, block_k=self.block_k)


def _run_correctness_tests() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available; skipping Triton tests.")
        return

    torch.manual_seed(42)
    dtypes = [torch.float32, torch.float16]
    shapes = [(32, 32, 32), (128, 128, 128), (512, 512, 512)]

    for dtype in dtypes:
        for (M, K, N) in shapes:
            a = torch.randn(M, K, device=device, dtype=dtype)
            b = torch.randn(K, N, device=device, dtype=dtype)
            ref = torch.matmul(a, b)

            out = matrix_mul_triton(a, b, block_m=32, block_n=32, block_k=16)
            torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
            max_diff = (out - ref).abs().max().item()
            print(f"[Day3][OK] M={M}, K={K}, N={N}, dtype={dtype}, max_diff={max_diff:.2e}")


def benchmark_matrix_mul(
    shapes: Optional[List[tuple]] = None,
    dtype: torch.dtype = torch.float32,
) -> List[Dict[str, Any]]:
    if shapes is None:
        shapes = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available; skipping benchmark.")
        return []

    torch.manual_seed(123)
    results: List[Dict[str, Any]] = []

    for (M, K, N) in shapes:
        a = torch.randn(M, K, device=device, dtype=dtype)
        b = torch.randn(K, N, device=device, dtype=dtype)

        # Warmup
        for _ in range(10):
            _ = matrix_mul_triton(a, b, block_m=32, block_n=32, block_k=16)
        torch.cuda.synchronize()

        iters = 100
        start = time.perf_counter()
        for _ in range(iters):
            out = matrix_mul_triton(a, b, block_m=32, block_n=32, block_k=16)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1e3 / iters

        ref = torch.matmul(a, b)
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)

        num_flops = 2 * M * N * K  # 2 operations per multiply-add
        gflops = (num_flops / (elapsed_ms / 1e3)) / 1e9
        bandwidth = ((M * K + K * N + M * N) * a.element_size()) / (elapsed_ms / 1e3) / 1e9

        print(
            f"[Day3][BENCH] M={M}, K={K}, N={N}, dtype={dtype}, time={elapsed_ms:.3f} ms, "
            f"GFLOPS={gflops:.2f}, BW={bandwidth:.2f} GB/s"
        )

        results.append(
            {
                "M": M,
                "K": K,
                "N": N,
                "dtype": str(dtype),
                "time_ms": elapsed_ms,
                "gflops": gflops,
                "bandwidth_gb_s": bandwidth,
            }
        )

    return results


if __name__ == "__main__":
    _run_correctness_tests()
    benchmark_matrix_mul()

