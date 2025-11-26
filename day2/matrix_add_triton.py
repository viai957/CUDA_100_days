"""
Matrix Addition Triton Implementation (Day 2): High-Performance Element-wise Add on 2D Grids
Math: C[i, j] = A[i, j] + B[i, j] for i ∈ [0, M), j ∈ [0, N)
Inputs / Outputs: A[M, N], B[M, N] -> C[M, N] contiguous row-major tensors
Assumptions: M, N > 0; tensors contiguous on CUDA device; shapes and dtypes match
Parallel Strategy: 2D grid where each Triton program processes a BLOCK_M × BLOCK_N tile
Mixed Precision Policy: FP16/BF16 storage allowed, accumulation in same dtype; FP32 recommended for validation
Distributed Hooks: Can be wrapped with data-parallel all-reduce on gradients or outputs
Complexity: O(MN) FLOPs, O(3 * M * N * sizeof(dtype)) bytes moved
Test Vectors: Deterministic random matrices for sizes like 32×32, 512×512 with max|Δ| < 1e-5 in FP32
"""

import math
import time
from typing import List, Dict, Any, Optional

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def matrix_add_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    stride_am,
    stride_an,
    stride_bm,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    2D tiled matrix addition kernel.
    Memory layout: row-major (stride_am = stride_bm = stride_cm = N, stride_an = stride_bn = stride_cn = 1).
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

    mask = (offs_m < M) & (offs_n < N)

    a = tl.load(a_ptr + offs_m * stride_am + offs_n * stride_an, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs_m * stride_bm + offs_n * stride_bn, mask=mask, other=0.0)
    c = a + b

    tl.store(c_ptr + offs_m * stride_cm + offs_n * stride_cn, c, mask=mask)


def _validate_inputs(a: torch.Tensor, b: torch.Tensor) -> (int, int):
    assert a.device.type == "cuda", "Inputs must be on CUDA device"
    assert b.device.type == "cuda", "Inputs must be on CUDA device"
    assert a.shape == b.shape, "Inputs must have the same shape"
    assert a.dim() == 2, "Inputs must be 2D tensors [M, N]"
    assert a.dtype == b.dtype, "Inputs must share dtype"
    assert a.is_contiguous(), "Input A must be contiguous"
    assert b.is_contiguous(), "Input B must be contiguous"
    M, N = a.shape
    assert M > 0 and N > 0, "Matrix dimensions must be > 0"
    return M, N


def matrix_add_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    block_m: int = 32,
    block_n: int = 32,
) -> torch.Tensor:
    """
    Public API: matrix addition using Triton kernel.
    """
    M, N = _validate_inputs(a, b)
    c = torch.empty_like(a)

    grid = (
        triton.cdiv(M, block_m),
        triton.cdiv(N, block_n),
    )

    stride_am = a.stride(0)
    stride_an = a.stride(1)
    stride_bm = b.stride(0)
    stride_bn = b.stride(1)
    stride_cm = c.stride(0)
    stride_cn = c.stride(1)

    matrix_add_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        stride_am,
        stride_an,
        stride_bm,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return c


class MatrixAddModule(torch.nn.Module):
    """
    PyTorch nn.Module wrapper to plug Day 2 matrix add into larger models.
    """

    def __init__(self, block_m: int = 32, block_n: int = 32):
        super().__init__()
        self.block_m = block_m
        self.block_n = block_n

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return matrix_add_triton(a, b, block_m=self.block_m, block_n=self.block_n)


def _run_correctness_tests() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available; skipping Triton tests.")
        return

    torch.manual_seed(42)
    dtypes = [torch.float32, torch.float16]
    shapes = [(32, 32), (128, 128), (512, 512)]

    for dtype in dtypes:
        for (M, N) in shapes:
            a = torch.randn(M, N, device=device, dtype=dtype)
            b = torch.randn(M, N, device=device, dtype=dtype)
            ref = a + b

            out = matrix_add_triton(a, b, block_m=32, block_n=32)
            torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
            max_diff = (out - ref).abs().max().item()
            print(f"[Day2][OK] M={M}, N={N}, dtype={dtype}, max_diff={max_diff:.2e}")


def benchmark_matrix_add(
    shapes: Optional[List[tuple]] = None,
    dtype: torch.dtype = torch.float32,
) -> List[Dict[str, Any]]:
    if shapes is None:
        shapes = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available; skipping benchmark.")
        return []

    torch.manual_seed(123)
    results: List[Dict[str, Any]] = []

    for (M, N) in shapes:
        a = torch.randn(M, N, device=device, dtype=dtype)
        b = torch.randn(M, N, device=device, dtype=dtype)

        # Warmup
        for _ in range(10):
            _ = matrix_add_triton(a, b, block_m=32, block_n=32)
        torch.cuda.synchronize()

        iters = 100
        start = time.perf_counter()
        for _ in range(iters):
            out = matrix_add_triton(a, b, block_m=32, block_n=32)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1e3 / iters

        ref = a + b
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

        num_el = M * N
        gflops = (num_el / (elapsed_ms / 1e3)) / 1e9
        bandwidth = (3 * num_el * a.element_size()) / (elapsed_ms / 1e3) / 1e9

        print(
            f"[Day2][BENCH] M={M}, N={N}, dtype={dtype}, time={elapsed_ms:.3f} ms, "
            f"GFLOPS={gflops:.2f}, BW={bandwidth:.2f} GB/s"
        )

        results.append(
            {
                "M": M,
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
    benchmark_matrix_add()


