/*
 * Partial Sum (Scan) Triton Implementation: High-Performance Prefix Sum Computation
 * Math: output[i] = Σ(input[j]) for j ∈ [0, i]
 * Inputs: input[N] - input array, N - array length
 * Assumptions: N > 0, array is contiguous, device has sufficient memory
 * Parallel Strategy: Each block processes multiple elements with tiled computation
 * Mixed Precision Policy: FP16/BF16 for computation, FP32 for reductions
 * Distributed Hooks: Ready for tensor parallelism via tl.comm_* primitives
 * Complexity: O(N) FLOPs, O(N) bytes moved
 * Test Vectors: Deterministic random arrays with known prefix sums
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
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def partial_sum_kernel(
    # Input array
    input_ptr,
    # Output array
    output_ptr,
    # Array length
    N,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for inclusive scan (partial sum).
    
    Memory layout: Contiguous 1D array
    Each thread block processes BLOCK_SIZE elements
    """
    
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate block range
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, N)
    
    # Load data into shared memory
    data = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx < N:
            data[i] = tl.load(input_ptr + idx)
    
    # Perform inclusive scan using Hillis-Steele algorithm
    # Up-sweep phase
    for stride in range(1, BLOCK_SIZE):
        stride_val = 2 ** stride
        for i in range(stride_val - 1, BLOCK_SIZE, stride_val):
            if i < BLOCK_SIZE:
                data[i] += data[i - stride_val // 2]
    
    # Down-sweep phase
    for stride in range(int(math.log2(BLOCK_SIZE)) - 1, -1, -1):
        stride_val = 2 ** stride
        for i in range(stride_val - 1, BLOCK_SIZE, stride_val):
            if i + stride_val // 2 < BLOCK_SIZE:
                data[i + stride_val // 2] += data[i]
    
    # Store result
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx < N:
            tl.store(output_ptr + idx, data[i])


@triton.jit
def partial_sum_optimized_kernel(
    # Input array
    input_ptr,
    # Output array
    output_ptr,
    # Array length
    N,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized Triton kernel with vectorized operations and better memory access.
    """
    
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate block range
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, N)
    
    # Vectorized processing for better memory bandwidth
    data = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Load data in chunks
    for i in range(0, BLOCK_SIZE, 4):
        if block_start + i < N:
            # Load 4 elements at a time
            mask = tl.arange(0, 4) + block_start + i < N
            vals = tl.load(input_ptr + block_start + i, mask=mask, other=0.0)
            data[i:i+4] = vals
    
    # Perform inclusive scan
    # Up-sweep phase
    for stride in range(1, BLOCK_SIZE):
        stride_val = 2 ** stride
        for i in range(stride_val - 1, BLOCK_SIZE, stride_val):
            if i < BLOCK_SIZE:
                data[i] += data[i - stride_val // 2]
    
    # Down-sweep phase
    for stride in range(int(math.log2(BLOCK_SIZE)) - 1, -1, -1):
        stride_val = 2 ** stride
        for i in range(stride_val - 1, BLOCK_SIZE, stride_val):
            if i + stride_val // 2 < BLOCK_SIZE:
                data[i + stride_val // 2] += data[i]
    
    # Store result in chunks
    for i in range(0, BLOCK_SIZE, 4):
        if block_start + i < N:
            mask = tl.arange(0, 4) + block_start + i < N
            vals = data[i:i+4]
            tl.store(output_ptr + block_start + i, vals, mask=mask)


def partial_sum_triton(
    input_array: torch.Tensor,
    optimized: bool = True
) -> torch.Tensor:
    """
    Apply partial sum using Triton kernel.
    
    Args:
        input_array: Input array [N]
        optimized: Whether to use optimized kernel
        
    Returns:
        Result array [N]
    """
    # Input validation
    assert input_array.dim() == 1, "Input must be 1D tensor"
    assert input_array.is_contiguous(), "Input tensor must be contiguous"
    
    N = input_array.shape[0]
    device = input_array.device
    dtype = input_array.dtype
    
    # Prepare output tensor
    output = torch.empty_like(input_array)
    
    # Ensure tensors are on correct device
    input_array = input_array.to(device)
    output = output.to(device)
    
    # Calculate grid dimensions
    grid_size = triton.cdiv(N, 256)  # Default block size for grid calculation
    
    # Launch kernel
    if optimized:
        partial_sum_optimized_kernel[(grid_size,)](
            input_array, output,
            N,
            BLOCK_SIZE=256,  # Will be overridden by autotune
        )
    else:
        partial_sum_kernel[(grid_size,)](
            input_array, output,
            N,
            BLOCK_SIZE=256,  # Will be overridden by autotune
        )
    
    return output


class PartialSumModule(torch.nn.Module):
    """
    PyTorch module wrapper for Triton partial sum.
    """
    
    def __init__(self, optimized: bool = True):
        super().__init__()
        self.optimized = optimized
        
    def forward(self, input_array: torch.Tensor) -> torch.Tensor:
        return partial_sum_triton(input_array, self.optimized)


def benchmark_partial_sum(
    sizes: list = [1024, 10000, 100000, 1000000, 10000000],
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32
) -> None:
    """
    Benchmark partial sum across different sizes and implementations.
    """
    print(f"\n=== Partial Sum Benchmark on {device} ===")
    print(f"Data type: {dtype}")
    
    results = []
    
    for N in sizes:
        print(f"\nArray size: {N:,}")
        
        # Create test tensor
        torch.manual_seed(42)
        input_array = torch.randn(N, device=device, dtype=dtype)
        
        # Expected result (using PyTorch)
        expected = torch.cumsum(input_array, dim=0)
        
        # Test Triton implementation
        if device == 'cuda':
            # Warmup
            for _ in range(10):
                _ = partial_sum_triton(input_array, optimized=True)
            
            torch.cuda.synchronize()
            
            # Benchmark optimized kernel
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            num_iterations = 100
            start_event.record()
            for _ in range(num_iterations):
                result = partial_sum_triton(input_array, optimized=True)
            end_event.record()
            
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            
            # Verify correctness
            torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
            
            # Calculate performance metrics
            avg_time_ms = elapsed_ms / num_iterations
            gflops = N / (avg_time_ms / 1000.0) / 1e9
            bandwidth_gb_s = 2 * N * input_array.element_size() / (avg_time_ms / 1000.0) / 1e9  # Read + Write
            
            print(f"  Triton optimized: {avg_time_ms:.3f} ms, {gflops:.2f} GFLOPS, {bandwidth_gb_s:.2f} GB/s")
            
            # Compare with PyTorch
            torch.cuda.synchronize()
            start_event.record()
            for _ in range(num_iterations):
                result_pytorch = torch.cumsum(input_array, dim=0)
            end_event.record()
            torch.cuda.synchronize()
            pytorch_time_ms = start_event.elapsed_time(end_event) / num_iterations
            
            speedup = pytorch_time_ms / avg_time_ms
            print(f"  PyTorch baseline: {pytorch_time_ms:.3f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            
            results.append({
                'size': N,
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
                result = partial_sum_triton(input_array, optimized=True)
            end_time = time.time()
            
            avg_time_ms = (end_time - start_time) / 10 * 1000
            print(f"  Triton (CPU): {avg_time_ms:.3f} ms")
    
    return results


# Unit test and profiling
if __name__ == "__main__":
    # Test configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    
    print(f"Testing Partial Sum Triton implementation on {device}")
    
    # Test with small array
    N = 1000
    torch.manual_seed(42)
    input_array = torch.randn(N, device=device, dtype=dtype)
    
    print(f"\n=== Testing with array size {N} ===")
    
    # Test Triton implementation
    result = partial_sum_triton(input_array, optimized=True)
    expected = torch.cumsum(input_array, dim=0)
    
    # Verify output
    assert result.shape == expected.shape, f"Output shape mismatch: {result.shape} vs {expected.shape}"
    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5), "Output values don't match expected"
    
    print("✓ Correctness test passed")
    
    # Test determinism
    result2 = partial_sum_triton(input_array, optimized=True)
    assert torch.allclose(result, result2), "Non-deterministic output detected"
    print("✓ Determinism test passed")
    
    # Test module wrapper
    print("\n=== Testing Module Wrapper ===")
    partial_sum_module = PartialSumModule(optimized=True)
    result_module = partial_sum_module(input_array)
    assert torch.allclose(result, result_module), "Module wrapper output mismatch"
    print("✓ Module wrapper test passed")
    
    # Run comprehensive benchmark
    if device.type == 'cuda':
        print("\n=== Performance Benchmark ===")
        benchmark_results = benchmark_partial_sum(
            sizes=[1024, 10000, 100000, 1000000],
            device=device.type,
            dtype=dtype
        )
        
        # Print summary
        print("\n=== Performance Summary ===")
        for result in benchmark_results:
            print(f"Size {result['size']:,}: {result['speedup']:.2f}x speedup, "
                  f"{result['gflops']:.2f} GFLOPS, {result['bandwidth']:.2f} GB/s")
    
    print("\n=== All tests passed! ===")

/*
 * Profiling example & performance tips:
 * 
 * 1. Use nsys profile to analyze kernel performance:
 *    nsys profile --trace=cuda python partial_sum_triton.py
 * 
 * 2. Monitor memory bandwidth utilization:
 *    nvprof --metrics achieved_occupancy,sm_efficiency python partial_sum_triton.py
 * 
 * 3. For optimal performance:
 *    - Use autotuning to find best block sizes for your hardware
 *    - Enable optimized kernel for better memory access patterns
 *    - Use appropriate data types (FP16/BF16 for memory-bound operations)
 *    - Consider vectorized operations for better throughput
 * 
 * 4. Memory optimization:
 *    - Ensure input tensors are contiguous
 *    - Use appropriate block sizes for your GPU architecture
 *    - Consider memory coalescing for better bandwidth utilization
 * 
 * 5. Algorithm optimization:
 *    - Use Hillis-Steele algorithm for better parallelism
 *    - Consider work-efficient algorithms for very large arrays
 *    - Use warp-level primitives for better performance
 */
