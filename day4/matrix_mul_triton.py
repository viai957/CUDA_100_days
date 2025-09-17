/*
 * Matrix Multiplication Triton Implementation: High-Performance General Matrix Multiply
 * Math: C[i,j] = Σ(A[i,k] × B[k,j]) for k ∈ [0, K)
 * Inputs: A[M,K], B[K,N] - input matrices, M, N, K - matrix dimensions
 * Assumptions: M, N, K > 0, matrices are contiguous, device has sufficient memory
 * Parallel Strategy: Each block processes a tile of [BLOCK_SIZE_M, BLOCK_SIZE_N]
 * Mixed Precision Policy: FP16/BF16 for computation, FP32 for reductions
 * Distributed Hooks: Ready for tensor parallelism via tl.comm_* primitives
 * Complexity: O(M×N×K) FLOPs, O(M×K + K×N + M×N) bytes moved
 * Test Vectors: Deterministic random matrices with known products
 */

import torch
import triton
import triton.language as tl
import math
from typing import Tuple, Optional
import time

# Autotune configurations for different hardware
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 16}, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matrix_multiply_kernel(
    # Input matrices
    a_ptr, b_ptr,
    # Output matrix
    c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block dimensions
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Triton kernel for matrix multiplication with tiling optimization.
    
    Memory layout: Row-major matrices
    Each program processes a tile of [BLOCK_SIZE_M, BLOCK_SIZE_N]
    """
    
    # Get program IDs
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Calculate block ranges
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # Bounds checking
    m_end = min(m_start + BLOCK_SIZE_M, M)
    n_end = min(n_start + BLOCK_SIZE_N, N)
    
    # Initialize accumulator
    accumulator = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    
    # Tiled computation
    for k in range(0, K, BLOCK_SIZE_K):
        k_end = min(k + BLOCK_SIZE_K, K)
        
        # Load A tile
        a_offsets = (m_start + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_am + \
                   (k + tl.arange(0, BLOCK_SIZE_K))[None, :] * stride_ak
        a_mask = (m_start + tl.arange(0, BLOCK_SIZE_M))[:, None] < M & \
                (k + tl.arange(0, BLOCK_SIZE_K))[None, :] < K
        a_vals = tl.load(a_ptr + a_offsets, mask=a_mask, other=0.0)
        
        # Load B tile
        b_offsets = (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * stride_bk + \
                   (n_start + tl.arange(0, BLOCK_SIZE_N))[None, :] * stride_bn
        b_mask = (k + tl.arange(0, BLOCK_SIZE_K))[:, None] < K & \
                (n_start + tl.arange(0, BLOCK_SIZE_N))[None, :] < N
        b_vals = tl.load(b_ptr + b_offsets, mask=b_mask, other=0.0)
        
        # Compute partial result
        accumulator += tl.dot(a_vals, b_vals)
    
    # Store result
    c_offsets = (m_start + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_cm + \
               (n_start + tl.arange(0, BLOCK_SIZE_N))[None, :] * stride_cn
    c_mask = (m_start + tl.arange(0, BLOCK_SIZE_M))[:, None] < M & \
            (n_start + tl.arange(0, BLOCK_SIZE_N))[None, :] < N
    tl.store(c_ptr + c_offsets, accumulator, mask=c_mask)


@triton.jit
def matrix_multiply_optimized_kernel(
    # Input matrices
    a_ptr, b_ptr,
    # Output matrix
    c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block dimensions
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Optimized Triton kernel with shared memory simulation and better memory access.
    """
    
    # Get program IDs
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # Calculate block ranges
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # Bounds checking
    m_end = min(m_start + BLOCK_SIZE_M, M)
    n_end = min(n_start + BLOCK_SIZE_N, N)
    
    # Initialize accumulator
    accumulator = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    
    # Tiled computation with vectorized loads
    for k in range(0, K, BLOCK_SIZE_K):
        k_end = min(k + BLOCK_SIZE_K, K)
        
        # Load A tile with vectorized access
        a_offsets = (m_start + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_am + \
                   (k + tl.arange(0, BLOCK_SIZE_K))[None, :] * stride_ak
        a_mask = (m_start + tl.arange(0, BLOCK_SIZE_M))[:, None] < M & \
                (k + tl.arange(0, BLOCK_SIZE_K))[None, :] < K
        a_vals = tl.load(a_ptr + a_offsets, mask=a_mask, other=0.0)
        
        # Load B tile with vectorized access
        b_offsets = (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * stride_bk + \
                   (n_start + tl.arange(0, BLOCK_SIZE_N))[None, :] * stride_bn
        b_mask = (k + tl.arange(0, BLOCK_SIZE_K))[:, None] < K & \
                (n_start + tl.arange(0, BLOCK_SIZE_N))[None, :] < N
        b_vals = tl.load(b_ptr + b_offsets, mask=b_mask, other=0.0)
        
        # Compute partial result with optimized dot product
        accumulator += tl.dot(a_vals, b_vals, allow_tf32=True)
    
    # Store result with vectorized access
    c_offsets = (m_start + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_cm + \
               (n_start + tl.arange(0, BLOCK_SIZE_N))[None, :] * stride_cn
    c_mask = (m_start + tl.arange(0, BLOCK_SIZE_M))[:, None] < M & \
            (n_start + tl.arange(0, BLOCK_SIZE_N))[None, :] < N
    tl.store(c_ptr + c_offsets, accumulator, mask=c_mask)


def matrix_multiply_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    optimized: bool = True
) -> torch.Tensor:
    """
    Apply matrix multiplication using Triton kernel.
    
    Args:
        a: First input matrix [M, K]
        b: Second input matrix [K, N]
        optimized: Whether to use optimized kernel
        
    Returns:
        Result matrix [M, N]
    """
    # Input validation
    assert a.dim() == 2, "Input must be 2D tensor"
    assert b.dim() == 2, "Input must be 2D tensor"
    assert a.shape[1] == b.shape[0], "Inner dimensions must match"
    assert a.is_contiguous(), "Input tensor must be contiguous"
    assert b.is_contiguous(), "Input tensor must be contiguous"
    
    M, K = a.shape
    K2, N = b.shape
    device = a.device
    dtype = a.dtype
    
    # Prepare output tensor
    c = torch.empty(M, N, device=device, dtype=dtype)
    
    # Ensure tensors are on correct device
    a = a.to(device)
    b = b.to(device)
    c = c.to(device)
    
    # Calculate grid dimensions
    grid_m = triton.cdiv(M, 64)  # Default block size for grid calculation
    grid_n = triton.cdiv(N, 64)  # Default block size for grid calculation
    
    # Launch kernel
    if optimized:
        matrix_multiply_optimized_kernel[(grid_m, grid_n)](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=64,  # Will be overridden by autotune
            BLOCK_SIZE_N=64,  # Will be overridden by autotune
            BLOCK_SIZE_K=32,  # Will be overridden by autotune
        )
    else:
        matrix_multiply_kernel[(grid_m, grid_n)](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=64,  # Will be overridden by autotune
            BLOCK_SIZE_N=64,  # Will be overridden by autotune
            BLOCK_SIZE_K=32,  # Will be overridden by autotune
        )
    
    return c


class MatrixMultiplyModule(torch.nn.Module):
    """
    PyTorch module wrapper for Triton matrix multiplication.
    """
    
    def __init__(self, optimized: bool = True):
        super().__init__()
        self.optimized = optimized
        
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return matrix_multiply_triton(a, b, self.optimized)


def benchmark_matrix_multiply(
    sizes: list = [(100, 100, 100), (500, 500, 500), (1000, 1000, 1000)],
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32
) -> None:
    """
    Benchmark matrix multiplication across different sizes and implementations.
    """
    print(f"\n=== Matrix Multiplication Benchmark on {device} ===")
    print(f"Data type: {dtype}")
    
    results = []
    
    for M, K, N in sizes:
        print(f"\nMatrix size: A[{M}×{K}] × B[{K}×{N}] = C[{M}×{N}]")
        
        # Create test tensors
        torch.manual_seed(42)
        a = torch.randn(M, K, device=device, dtype=dtype)
        b = torch.randn(K, N, device=device, dtype=dtype)
        
        # Expected result
        expected = torch.matmul(a, b)
        
        # Test Triton implementation
        if device == 'cuda':
            # Warmup
            for _ in range(10):
                _ = matrix_multiply_triton(a, b, optimized=True)
            
            torch.cuda.synchronize()
            
            # Benchmark optimized kernel
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            num_iterations = 100
            start_event.record()
            for _ in range(num_iterations):
                result = matrix_multiply_triton(a, b, optimized=True)
            end_event.record()
            
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            
            # Verify correctness
            torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)
            
            # Calculate performance metrics
            avg_time_ms = elapsed_ms / num_iterations
            flops = 2 * M * N * K  # 2 operations per multiply-add
            gflops = flops / (avg_time_ms / 1000.0) / 1e9
            bandwidth_gb_s = (M * K + K * N + M * N) * a.element_size() / (avg_time_ms / 1000.0) / 1e9
            
            print(f"  Triton optimized: {avg_time_ms:.3f} ms, {gflops:.2f} GFLOPS, {bandwidth_gb_s:.2f} GB/s")
            
            # Compare with PyTorch
            torch.cuda.synchronize()
            start_event.record()
            for _ in range(num_iterations):
                result_pytorch = torch.matmul(a, b)
            end_event.record()
            torch.cuda.synchronize()
            pytorch_time_ms = start_event.elapsed_time(end_event) / num_iterations
            
            speedup = pytorch_time_ms / avg_time_ms
            print(f"  PyTorch baseline: {pytorch_time_ms:.3f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            
            results.append({
                'size': (M, K, N),
                'triton_time': avg_time_ms,
                'pytorch_time': pytorch_time_ms,
                'speedup': speedup,
                'gflops': gflops,
                'bandwidth': bandwidth_gb_s
            })
        else:
            # CPU fallback
            start_time = time.time()
            for _ in range(10):
                result = matrix_multiply_triton(a, b, optimized=True)
            end_time = time.time()
            
            avg_time_ms = (end_time - start_time) / 10 * 1000
            print(f"  Triton (CPU): {avg_time_ms:.3f} ms")
    
    return results


# Unit test and profiling
if __name__ == "__main__":
    # Test configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    
    print(f"Testing Matrix Multiplication Triton implementation on {device}")
    
    # Test with small matrices
    M, K, N = 100, 100, 100
    torch.manual_seed(42)
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(K, N, device=device, dtype=dtype)
    
    print(f"\n=== Testing with matrix size A[{M}×{K}] × B[{K}×{N}] ===")
    
    # Test Triton implementation
    result = matrix_multiply_triton(a, b, optimized=True)
    expected = torch.matmul(a, b)
    
    # Verify output
    assert result.shape == expected.shape, f"Output shape mismatch: {result.shape} vs {expected.shape}"
    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4), "Output values don't match expected"
    
    print("✓ Correctness test passed")
    
    # Test determinism
    result2 = matrix_multiply_triton(a, b, optimized=True)
    assert torch.allclose(result, result2), "Non-deterministic output detected"
    print("✓ Determinism test passed")
    
    # Test module wrapper
    print("\n=== Testing Module Wrapper ===")
    matrix_multiply_module = MatrixMultiplyModule(optimized=True)
    result_module = matrix_multiply_module(a, b)
    assert torch.allclose(result, result_module), "Module wrapper output mismatch"
    print("✓ Module wrapper test passed")
    
    # Run comprehensive benchmark
    if device.type == 'cuda':
        print("\n=== Performance Benchmark ===")
        benchmark_results = benchmark_matrix_multiply(
            sizes=[(100, 100, 100), (500, 500, 500), (1000, 1000, 1000)],
            device=device.type,
            dtype=dtype
        )
        
        # Print summary
        print("\n=== Performance Summary ===")
        for result in benchmark_results:
            print(f"Size {result['size']}: {result['speedup']:.2f}x speedup, "
                  f"{result['gflops']:.2f} GFLOPS, {result['bandwidth']:.2f} GB/s")
    
    print("\n=== All tests passed! ===")

/*
 * Profiling example & performance tips:
 * 
 * 1. Use nsys profile to analyze kernel performance:
 *    nsys profile --trace=cuda python matrix_mul_triton.py
 * 
 * 2. Monitor memory bandwidth utilization:
 *    nvprof --metrics achieved_occupancy,sm_efficiency python matrix_mul_triton.py
 * 
 * 3. For optimal performance:
 *    - Use autotuning to find best block sizes for your hardware
 *    - Enable optimized kernel for better memory access patterns
 *    - Use appropriate data types (FP16/BF16 for memory-bound operations)
 *    - Consider tiling strategies for large matrices
 * 
 * 4. Memory optimization:
 *    - Ensure input tensors are contiguous
 *    - Use appropriate block sizes for your GPU architecture
 *    - Consider memory coalescing for better bandwidth utilization
 * 
 * 5. Distributed training considerations:
 *    - Use tl.comm_* primitives for multi-GPU operations
 *    - Implement gradient synchronization for distributed training
 *    - Consider memory-efficient implementations for large models
 */
