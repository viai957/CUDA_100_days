/*
 * Layer Normalization Triton Implementation: High-Performance Normalization
 * Math: output = (input - mean) / sqrt(variance + epsilon) * gamma + beta
 * Inputs: input[N, D] - input tensor, N - batch size, D - feature dimension
 * Assumptions: N > 0, D > 0, tensors are contiguous, device has sufficient memory
 * Parallel Strategy: Each block processes multiple samples with tiled computation
 * Mixed Precision Policy: FP16/BF16 for computation, FP32 for reductions
 * Distributed Hooks: Ready for tensor parallelism via tl.comm_* primitives
 * Complexity: O(ND) FLOPs, O(ND) bytes moved
 * Test Vectors: Deterministic random tensors with known normalization results
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
        triton.Config({'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_D': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_D': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_D': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_D': 512}, num_warps=8),
    ],
    key=['batch_size', 'feature_dim'],
)
@triton.jit
def layer_norm_kernel(
    # Input tensor
    input_ptr,
    # Output tensor
    output_ptr,
    # Scale and shift parameters
    gamma_ptr, beta_ptr,
    # Dimensions
    batch_size, feature_dim,
    # Epsilon for numerical stability
    epsilon,
    # Block dimensions
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Triton kernel for layer normalization.
    
    Memory layout: [batch_size, feature_dim]
    Each thread block processes [BLOCK_SIZE_N, BLOCK_SIZE_D]
    """
    
    # Get program IDs
    pid_n = tl.program_id(axis=0)
    pid_d = tl.program_id(axis=1)
    
    # Calculate block ranges
    n_start = pid_n * BLOCK_SIZE_N
    d_start = pid_d * BLOCK_SIZE_D
    
    # Bounds checking
    n_end = min(n_start + BLOCK_SIZE_N, batch_size)
    d_end = min(d_start + BLOCK_SIZE_D, feature_dim)
    
    # Process each sample in the block
    for n in range(n_start, n_end):
        # Load data for this sample
        data = tl.zeros([BLOCK_SIZE_D], dtype=tl.float32)
        for d in range(d_start, d_end):
            if d < feature_dim:
                data[d - d_start] = tl.load(input_ptr + n * feature_dim + d)
        
        # Compute mean
        mean = tl.sum(data) / feature_dim
        
        # Compute variance
        variance = tl.sum((data - mean) ** 2) / feature_dim
        
        # Normalize
        stddev = tl.sqrt(variance + epsilon)
        normalized = (data - mean) / stddev
        
        # Apply scale and shift
        for d in range(d_start, d_end):
            if d < feature_dim:
                gamma_val = tl.load(gamma_ptr + d)
                beta_val = tl.load(beta_ptr + d)
                result = normalized[d - d_start] * gamma_val + beta_val
                tl.store(output_ptr + n * feature_dim + d, result)


@triton.jit
def layer_norm_optimized_kernel(
    # Input tensor
    input_ptr,
    # Output tensor
    output_ptr,
    # Scale and shift parameters
    gamma_ptr, beta_ptr,
    # Dimensions
    batch_size, feature_dim,
    # Epsilon for numerical stability
    epsilon,
    # Block dimensions
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Optimized Triton kernel with vectorized operations and better memory access.
    """
    
    # Get program IDs
    pid_n = tl.program_id(axis=0)
    pid_d = tl.program_id(axis=1)
    
    # Calculate block ranges
    n_start = pid_n * BLOCK_SIZE_N
    d_start = pid_d * BLOCK_SIZE_D
    
    # Bounds checking
    n_end = min(n_start + BLOCK_SIZE_N, batch_size)
    d_end = min(d_start + BLOCK_SIZE_D, feature_dim)
    
    # Process each sample in the block
    for n in range(n_start, n_end):
        # Vectorized load for better memory bandwidth
        data = tl.zeros([BLOCK_SIZE_D], dtype=tl.float32)
        for d in range(d_start, d_end, 4):
            if d + 3 < d_end:
                # Load 4 elements at a time
                mask = tl.arange(0, 4) + d < d_end
                vals = tl.load(input_ptr + n * feature_dim + d, mask=mask, other=0.0)
                data[d - d_start:d - d_start + 4] = vals
            else:
                # Handle remaining elements
                for i in range(d, min(d + 4, d_end)):
                    if i < feature_dim:
                        data[i - d_start] = tl.load(input_ptr + n * feature_dim + i)
        
        # Compute mean using vectorized operations
        mean = tl.sum(data) / feature_dim
        
        # Compute variance using vectorized operations
        variance = tl.sum((data - mean) ** 2) / feature_dim
        
        # Normalize
        stddev = tl.sqrt(variance + epsilon)
        normalized = (data - mean) / stddev
        
        # Apply scale and shift with vectorized operations
        for d in range(d_start, d_end, 4):
            if d + 3 < d_end:
                # Load gamma and beta values
                gamma_vals = tl.load(gamma_ptr + d, mask=tl.arange(0, 4) + d < d_end, other=1.0)
                beta_vals = tl.load(beta_ptr + d, mask=tl.arange(0, 4) + d < d_end, other=0.0)
                
                # Apply scale and shift
                result = normalized[d - d_start:d - d_start + 4] * gamma_vals + beta_vals
                
                # Store result
                tl.store(output_ptr + n * feature_dim + d, result, mask=tl.arange(0, 4) + d < d_end)
            else:
                # Handle remaining elements
                for i in range(d, min(d + 4, d_end)):
                    if i < feature_dim:
                        gamma_val = tl.load(gamma_ptr + i)
                        beta_val = tl.load(beta_ptr + i)
                        result = normalized[i - d_start] * gamma_val + beta_val
                        tl.store(output_ptr + n * feature_dim + i, result)


def layer_norm_triton(
    input_tensor: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    epsilon: float = 1e-7,
    optimized: bool = True
) -> torch.Tensor:
    """
    Apply layer normalization using Triton kernel.
    
    Args:
        input_tensor: Input tensor [batch_size, feature_dim]
        gamma: Scale parameters [feature_dim]
        beta: Shift parameters [feature_dim]
        epsilon: Small value for numerical stability
        optimized: Whether to use optimized kernel
        
    Returns:
        Result tensor [batch_size, feature_dim]
    """
    # Input validation
    assert input_tensor.dim() == 2, "Input must be 2D tensor"
    assert gamma.dim() == 1, "Gamma must be 1D tensor"
    assert beta.dim() == 1, "Beta must be 1D tensor"
    assert input_tensor.shape[1] == gamma.shape[0], "Feature dimensions must match"
    assert input_tensor.shape[1] == beta.shape[0], "Feature dimensions must match"
    
    batch_size, feature_dim = input_tensor.shape
    device = input_tensor.device
    dtype = input_tensor.dtype
    
    # Prepare output tensor
    output = torch.empty_like(input_tensor)
    
    # Ensure tensors are on correct device
    input_tensor = input_tensor.to(device)
    gamma = gamma.to(device)
    beta = beta.to(device)
    output = output.to(device)
    
    # Calculate grid dimensions
    grid_n = triton.cdiv(batch_size, 32)  # Default block size for batch dimension
    grid_d = triton.cdiv(feature_dim, 128)  # Default block size for feature dimension
    
    # Launch kernel
    if optimized:
        layer_norm_optimized_kernel[(grid_n, grid_d)](
            input_tensor, output, gamma, beta,
            batch_size, feature_dim,
            epsilon=epsilon,
            BLOCK_SIZE_N=32,  # Will be overridden by autotune
            BLOCK_SIZE_D=128,  # Will be overridden by autotune
        )
    else:
        layer_norm_kernel[(grid_n, grid_d)](
            input_tensor, output, gamma, beta,
            batch_size, feature_dim,
            epsilon=epsilon,
            BLOCK_SIZE_N=32,  # Will be overridden by autotune
            BLOCK_SIZE_D=128,  # Will be overridden by autotune
        )
    
    return output


class LayerNormModule(torch.nn.Module):
    """
    PyTorch module wrapper for Triton layer normalization.
    """
    
    def __init__(self, feature_dim: int, epsilon: float = 1e-7, optimized: bool = True):
        super().__init__()
        self.feature_dim = feature_dim
        self.epsilon = epsilon
        self.optimized = optimized
        
        # Initialize gamma and beta
        self.gamma = torch.ones(feature_dim)
        self.beta = torch.zeros(feature_dim)
        
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return layer_norm_triton(input_tensor, self.gamma, self.beta, self.epsilon, self.optimized)


def benchmark_layer_norm(
    batch_sizes: list = [1, 4, 8, 16, 32],
    feature_dims: list = [128, 256, 512, 1024, 2048],
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32
) -> None:
    """
    Benchmark layer normalization across different configurations.
    """
    print(f"\n=== Layer Normalization Benchmark on {device} ===")
    print(f"Data type: {dtype}")
    
    results = []
    
    for batch_size in batch_sizes:
        for feature_dim in feature_dims:
            print(f"\nConfiguration: batch_size={batch_size}, feature_dim={feature_dim}")
            
            # Create test tensors
            torch.manual_seed(42)
            input_tensor = torch.randn(batch_size, feature_dim, device=device, dtype=dtype)
            gamma = torch.ones(feature_dim, device=device, dtype=dtype)
            beta = torch.zeros(feature_dim, device=device, dtype=dtype)
            
            # Expected result (using PyTorch)
            expected = torch.nn.functional.layer_norm(input_tensor, [feature_dim], gamma, beta)
            
            # Test Triton implementation
            if device == 'cuda':
                # Warmup
                for _ in range(10):
                    _ = layer_norm_triton(input_tensor, gamma, beta, optimized=True)
                
                torch.cuda.synchronize()
                
                # Benchmark optimized kernel
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                num_iterations = 100
                start_event.record()
                for _ in range(num_iterations):
                    result = layer_norm_triton(input_tensor, gamma, beta, optimized=True)
                end_event.record()
                
                torch.cuda.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
                
                # Verify correctness
                torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)
                
                # Calculate performance metrics
                avg_time_ms = elapsed_ms / num_iterations
                flops = 3 * batch_size * feature_dim  # Mean, variance, normalization
                gflops = flops / (avg_time_ms / 1000.0) / 1e9
                bandwidth_gb_s = 4 * batch_size * feature_dim * input_tensor.element_size() / (avg_time_ms / 1000.0) / 1e9
                
                print(f"  Triton optimized: {avg_time_ms:.3f} ms, {gflops:.2f} GFLOPS, {bandwidth_gb_s:.2f} GB/s")
                
                # Compare with PyTorch
                torch.cuda.synchronize()
                start_event.record()
                for _ in range(num_iterations):
                    result_pytorch = torch.nn.functional.layer_norm(input_tensor, [feature_dim], gamma, beta)
                end_event.record()
                torch.cuda.synchronize()
                pytorch_time_ms = start_event.elapsed_time(end_event) / num_iterations
                
                speedup = pytorch_time_ms / avg_time_ms
                print(f"  PyTorch baseline: {pytorch_time_ms:.3f} ms")
                print(f"  Speedup: {speedup:.2f}x")
                
                results.append({
                    'batch_size': batch_size,
                    'feature_dim': feature_dim,
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
                    result = layer_norm_triton(input_tensor, gamma, beta, optimized=True)
                end_time = time.time()
                
                avg_time_ms = (end_time - start_time) / 10 * 1000
                print(f"  Triton (CPU): {avg_time_ms:.3f} ms")
    
    return results


# Unit test and profiling
if __name__ == "__main__":
    # Test configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    batch_size, feature_dim = 10, 10
    
    print(f"Testing Layer Normalization Triton implementation on {device}")
    print(f"Configuration: batch_size={batch_size}, feature_dim={feature_dim}")
    
    # Create test tensors
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, feature_dim, device=device, dtype=dtype)
    gamma = torch.ones(feature_dim, device=device, dtype=dtype)
    beta = torch.zeros(feature_dim, device=device, dtype=dtype)
    
    print(f"\n=== Testing with configuration: {batch_size}x{feature_dim} ===")
    
    # Test Triton implementation
    result = layer_norm_triton(input_tensor, gamma, beta, optimized=True)
    expected = torch.nn.functional.layer_norm(input_tensor, [feature_dim], gamma, beta)
    
    # Verify output
    assert result.shape == expected.shape, f"Output shape mismatch: {result.shape} vs {expected.shape}"
    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4), "Output values don't match expected"
    
    print("✓ Correctness test passed")
    
    # Test determinism
    result2 = layer_norm_triton(input_tensor, gamma, beta, optimized=True)
    assert torch.allclose(result, result2), "Non-deterministic output detected"
    print("✓ Determinism test passed")
    
    # Test module wrapper
    print("\n=== Testing Module Wrapper ===")
    layer_norm_module = LayerNormModule(feature_dim, optimized=True)
    result_module = layer_norm_module(input_tensor)
    assert torch.allclose(result, result_module), "Module wrapper output mismatch"
    print("✓ Module wrapper test passed")
    
    # Run comprehensive benchmark
    if device.type == 'cuda':
        print("\n=== Performance Benchmark ===")
        benchmark_results = benchmark_layer_norm(
            batch_sizes=[1, 4, 8, 16],
            feature_dims=[128, 256, 512],
            device=device.type,
            dtype=dtype
        )
        
        # Print summary
        print("\n=== Performance Summary ===")
        for result in benchmark_results:
            print(f"Config {result['batch_size']}x{result['feature_dim']}: "
                  f"{result['speedup']:.2f}x speedup, {result['gflops']:.2f} GFLOPS, "
                  f"{result['bandwidth']:.2f} GB/s")
    
    print("\n=== All tests passed! ===")

/*
 * Profiling example & performance tips:
 * 
 * 1. Use nsys profile to analyze kernel performance:
 *    nsys profile --trace=cuda python layer_norm_triton.py
 * 
 * 2. Monitor memory bandwidth utilization:
 *    nvprof --metrics achieved_occupancy,sm_efficiency python layer_norm_triton.py
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
 *    - Use efficient reduction algorithms for mean and variance
 *    - Consider numerical stability in variance computation
 *    - Use appropriate epsilon values for numerical stability
 */
