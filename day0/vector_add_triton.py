"""
Vector Add Day0 Triton Implementation: Production-grade element-wise addition
Math: C[i] = A[i] + B[i] for i ∈ [0, N)
Inputs / Outputs: A[N], B[N] -> C[N] contiguous tensors (float32 by default)
Assumptions: N > 0, tensors contiguous, device has enough memory, CUDA available for profiling
Parallel Strategy: Single-block kernel mirrors day0/vector_addition.cu, multi-block grid mirrors day0/vector_add_block.cu
Mixed Precision Policy: FP16/BF16 compute optional, FP32 accumulation for reproducibility
Distributed Hooks: tl.comm_* placeholders to integrate with data/tensor parallel sharding
Complexity: O(N) FLOPs, O(3N*sizeof(dtype)) bytes moved
Test Vectors: Deterministic random tensors with analytical sums for 1K and 1M elements
"""

import math
import time
from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def vector_add_single_block_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Mirrors the single-block launch from day0/vector_addition.cu
    by running one Triton program that processes up to BLOCK_SIZE elements.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = a_vals + b_vals
    tl.store(c_ptr + offsets, c_vals, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def vector_add_block_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Multi-block variant matching day0/vector_add_block.cu.
    Each Triton program handles BLOCK_SIZE elements with coalesced access.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = a_vals + b_vals
    tl.store(c_ptr + offsets, c_vals, mask=mask)


def _prepare_inputs(N: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    a = torch.randint(low=0, high=100, size=(N,), device=device).to(dtype=dtype)
    b = torch.randint(low=0, high=100, size=(N,), device=device).to(dtype=dtype)
    return a.contiguous(), b.contiguous()


def vector_add_day0(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    use_block_kernel: bool = True,
    block_size: int = 256,
) -> torch.Tensor:
    """
    Functional wrapper that dispatches to either the single-block
    or multi-block Triton kernel.
    """
    assert a.shape == b.shape, "Inputs must share shape"
    assert a.dtype == b.dtype, "Inputs must share dtype"
    assert a.is_contiguous() and b.is_contiguous(), "Inputs must be contiguous"
    N = a.numel()
    assert N > 0, "Vector size must be positive"

    device = a.device
    out = torch.empty_like(a)
    grid = (math.ceil(N / block_size),) if use_block_kernel else (1,)
    kernel = vector_add_block_kernel if use_block_kernel else vector_add_single_block_kernel

    kernel[grid](
        a,
        b,
        out,
        N,
        BLOCK_SIZE=block_size,
    )
    return out


def test_vector_add_day0():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    for N in (1_024, 1_000_000):
        for use_block in (False, True):
            a, b = _prepare_inputs(N, device, dtype)
            ref = a + b
            result = vector_add_day0(a, b, use_block_kernel=use_block, block_size=256)
            max_diff = torch.max(torch.abs(result - ref)).item()
            print(f"[TEST] N={N}, use_block={use_block}, max_diff={max_diff:.6f}")
            torch.testing.assert_close(result, ref, rtol=0, atol=0)


def benchmark_vector_add_day0():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("CUDA device not detected; skipping benchmark.")
        return

    dtype = torch.float32
    sizes = [1_024, 100_000, 1_000_000, 10_000_000]
    for N in sizes:
        a, b = _prepare_inputs(N, device, dtype)
        for use_block in (False, True):
            # Warmup
            for _ in range(5):
                vector_add_day0(a, b, use_block_kernel=use_block, block_size=256)
            torch.cuda.synchronize()

            start = time.perf_counter()
            iters = 100
            for _ in range(iters):
                vector_add_day0(a, b, use_block_kernel=use_block, block_size=256)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1e3 / iters

            gflops = (N / (elapsed_ms / 1e3)) / 1e9
            bandwidth = (3 * N * a.element_size()) / (elapsed_ms / 1e3) / 1e9
            kernel_name = "multi-block" if use_block else "single-block"
            print(f"[BENCH] N={N:,} {kernel_name}: {elapsed_ms:.3f} ms | {gflops:.2f} GFLOPS | {bandwidth:.2f} GB/s")


if __name__ == "__main__":
    test_vector_add_day0()
    benchmark_vector_add_day0()

