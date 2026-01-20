"""
Matrix Multiplication Triton Implementation (Day 2): Tiled GEMM using tl.dot
Math: C[m, n] = Σ_k A[m, k] * B[k, n]
Inputs / Outputs: A[M, K], B[K, N] -> C[M, N] contiguous row-major tensors
Assumptions: CUDA available; inputs are 2D, contiguous; A.shape[1] == B.shape[0]
Parallel Strategy: 2D grid; each Triton program computes a BLOCK_M x BLOCK_N tile of C
Mixed Precision Policy: FP16/BF16 inputs allowed; accumulation in FP32; output stored in input dtype
Distributed Hooks: Can be wrapped externally (e.g., torch.distributed) for TP/DP; kernel is local GEMM
Complexity: ~2*M*N*K FLOPs; bytes ~ (M*K + K*N + M*N)*sizeof(dtype)
Test Vectors: Small deterministic shapes (32, 64, 128) and odd sizes (127, 255) vs torch.matmul
"""

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 16}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
    ],
    key=["M", "N", "K", "DTYPE"],
)
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)[None, :]
        offs_k_t = k0 + tl.arange(0, BLOCK_K)[:, None]

        a_mask = (offs_m < M) & (offs_k < K)
        b_mask = (offs_k_t < K) & (offs_n < N)

        a = tl.load(a_ptr + offs_m * stride_am + offs_k * stride_ak, mask=a_mask, other=0.0)
        b = tl.load(b_ptr + offs_k_t * stride_bk + offs_n * stride_bn, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

    out = acc
    if DTYPE == 16:
        out = out.to(tl.float16)
    elif DTYPE == 17:
        out = out.to(tl.bfloat16)
    else:
        out = out.to(tl.float32)

    c_mask = (offs_m < M) & (offs_n < N)
    tl.store(c_ptr + offs_m * stride_cm + offs_n * stride_cn, out, mask=c_mask)


def _dtype_key(x: torch.dtype) -> int:
    # Triton constexpr-friendly "enum"
    if x == torch.float16:
        return 16
    if x == torch.bfloat16:
        return 17
    if x == torch.float32:
        return 32
    raise AssertionError(f"Unsupported dtype: {x}")


def _validate(a: torch.Tensor, b: torch.Tensor) -> Tuple[int, int, int]:
    assert a.device.type == "cuda" and b.device.type == "cuda", "Inputs must be CUDA tensors"
    assert a.dim() == 2 and b.dim() == 2, "Inputs must be 2D [M, K] and [K, N]"
    assert a.shape[1] == b.shape[0], "Inner dimensions must match"
    assert a.is_contiguous() and b.is_contiguous(), "Inputs must be contiguous"
    assert a.dtype == b.dtype, "A and B must share dtype"
    assert a.dtype in (torch.float16, torch.bfloat16, torch.float32), "dtype must be fp16/bf16/fp32"
    M, K = a.shape
    _, N = b.shape
    assert M > 0 and N > 0 and K > 0
    return M, N, K


def matmul_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, N, K = _validate(a, b)
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    matmul_kernel[grid](
        a,
        b,
        c,
        M=M,
        N=N,
        K=K,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        stride_cm=c.stride(0),
        stride_cn=c.stride(1),
        DTYPE=_dtype_key(a.dtype),
    )
    return c


def _run_correctness_tests() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available; skipping Triton matmul tests.")
        return

    torch.manual_seed(42)
    device = torch.device("cuda")
    shapes = [(32, 32, 32), (64, 64, 64), (128, 128, 64), (127, 255, 63)]
    dtypes = [torch.float16, torch.float32]

    for dtype in dtypes:
        for (M, N, K) in shapes:
            a = torch.randn(M, K, device=device, dtype=dtype)
            b = torch.randn(K, N, device=device, dtype=dtype)
            ref = a @ b
            out = matmul_triton(a, b)
            torch.testing.assert_close(out, ref, rtol=1e-3 if dtype != torch.float32 else 1e-4,
                                      atol=1e-3 if dtype != torch.float32 else 1e-4)
            max_diff = (out - ref).abs().max().item()
            print(f"[Day2][OK] M={M}, N={N}, K={K}, dtype={dtype}, max_diff={max_diff:.2e}")


def benchmark_matmul(
    shapes: Optional[List[Tuple[int, int, int]]] = None,
    dtype: torch.dtype = torch.float16,
    iters: int = 100,
) -> List[Dict[str, Any]]:
    if shapes is None:
        shapes = [(256, 256, 256), (512, 512, 512), (1024, 1024, 512)]
    if not torch.cuda.is_available():
        print("CUDA not available; skipping benchmark.")
        return []

    device = torch.device("cuda")
    torch.manual_seed(123)
    results: List[Dict[str, Any]] = []

    for (M, N, K) in shapes:
        a = torch.randn(M, K, device=device, dtype=dtype)
        b = torch.randn(K, N, device=device, dtype=dtype)

        # Warmup
        for _ in range(10):
            _ = matmul_triton(a, b)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            out = matmul_triton(a, b)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - start) * 1e3 / iters

        # Correctness spot-check
        ref = a @ b
        torch.testing.assert_close(out, ref, rtol=1e-3 if dtype != torch.float32 else 1e-4,
                                  atol=1e-3 if dtype != torch.float32 else 1e-4)

        flops = 2.0 * M * N * K
        t = ms / 1e3
        tflops = flops / t / 1e12
        bytes_moved = (M * K + K * N + M * N) * a.element_size()
        bw_gbs = bytes_moved / t / 1e9

        print(f"[Day2][BENCH] M={M}, N={N}, K={K}, dtype={dtype}, time={ms:.3f} ms, TFLOPs={tflops:.2f}, BW={bw_gbs:.2f} GB/s")
        results.append(
            {"M": M, "N": N, "K": K, "dtype": str(dtype), "time_ms": ms, "tflops": tflops, "bandwidth_gb_s": bw_gbs}
        )

    return results


if __name__ == "__main__":
    _run_correctness_tests()
    benchmark_matmul()

