"""
Vector Add: Simple element-wise addition kernel
Math: OUT[i] = A[i] + B[i] for i ∈ [0, N)
Inputs / Outputs: A[N], B[N] -> OUT[N] contiguous tensors (int32)
Assumptions: N > 0, tensors contiguous, CUDA available
Parallel Strategy: Single program processes all elements with block-based tiling
Complexity: O(N) FLOPs, O(3N*sizeof(int32)) bytes moved
Test Vectors: Deterministic random tensors (0-99) matching CUDA implementation
"""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    out_ptr,
    a_ptr,
    b_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Element-wise vector addition kernel.
    Each program processes BLOCK_SIZE elements with coalesced memory access.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load with mask to handle boundary cases
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0)
    
    # Compute element-wise addition
    out_vals = a_vals + b_vals
    
    # Store with mask
    tl.store(out_ptr + offsets, out_vals, mask=mask)


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Vector addition: OUT = A + B
    
    Args:
        a: Input tensor A
        b: Input tensor B
    
    Returns:
        Output tensor OUT = A + B
    """
    assert a.shape == b.shape, "Inputs must have same shape"
    assert a.dtype == b.dtype, "Inputs must have same dtype"
    assert a.is_contiguous() and b.is_contiguous(), "Inputs must be contiguous"
    
    N = a.numel()
    assert N > 0, "Vector size must be positive"
    
    out = torch.empty_like(a)
    
    # Block size: balance between occupancy and simplicity
    # 256 is a good default for most GPUs
    BLOCK_SIZE = 256
    grid = (math.ceil(N / BLOCK_SIZE),)
    
    vector_add_kernel[grid](
        out,
        a,
        b,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def test_vector_add(N: int = 1_000_000):
    """
    Test vector addition with correctness check and timing.
    Matches the CUDA test_vector_add function behavior.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("CUDA not available, skipping test")
        return
    
    # Set seed for reproducibility (matching CUDA srand(0))
    torch.manual_seed(0)
    
    # Allocate and initialize vectors (matching CUDA: rand() % 100)
    a = torch.randint(low=0, high=100, size=(N,), device=device, dtype=torch.int32)
    b = torch.randint(low=0, high=100, size=(N,), device=device, dtype=torch.int32)
    
    # Warmup
    for _ in range(5):
        _ = vector_add(a, b)
    torch.cuda.synchronize()
    
    # Time kernel execution
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    out = vector_add(a, b)
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    
    print(f"Vector Add - elapsed time: {elapsed_ms:.3f} ms")
    
    # Verify correctness
    expected = a + b
    max_diff = torch.max(torch.abs(out - expected)).item()
    
    if max_diff == 0:
        print("Vector Add - result OK")
    else:
        print(f"Vector Add - ERROR: max difference = {max_diff}")
        # Find first error (matching CUDA behavior)
        diff_mask = out != expected
        if diff_mask.any():
            idx = torch.nonzero(diff_mask, as_tuple=False)[0].item()
            print(f"Error at index {idx}: {out[idx].item()} != {a[idx].item()} + {b[idx].item()}")


if __name__ == "__main__":
    test_vector_add(1_000_000)
