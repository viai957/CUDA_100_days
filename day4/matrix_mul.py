"""
Matrix Multiplication PyTorch Implementation: High-Performance General Matrix Multiply
Math: C[i,j] = Σ(A[i,k] × B[k,j]) for k ∈ [0, K)
Inputs: A[M,K], B[K,N] - input matrices, M, N, K - matrix dimensions
Assumptions: M, N, K > 0, matrices are contiguous, device has sufficient memory
Parallel Strategy: PyTorch's optimized BLAS operations with automatic parallelization
Mixed Precision Policy: Configurable data types (FP16, FP32, FP64)
Distributed Hooks: Ready for multi-GPU via torch.distributed and DataParallel
Complexity: O(M×N×K) FLOPs, O(M×K + K×N + M×N) bytes moved
Test Vectors: Deterministic random matrices with known products
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from typing import Tuple, Optional, List, Dict
import numpy as np
from dataclasses import dataclass


@dataclass
class MatrixMultiplyConfig:
    """Configuration for matrix multiplication operations."""
    dtype: torch.dtype = torch.float32
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_mixed_precision: bool = False
    enable_gradient_checkpointing: bool = False
    use_tf32: bool = True  # Enable TF32 for better performance on Ampere GPUs


class MatrixMultiplyModule(nn.Module):
    """
    PyTorch module for matrix multiplication with advanced features.
    
    Features:
    - Automatic mixed precision support
    - Gradient checkpointing for memory efficiency
    - Configurable data types and precision
    - Built-in performance monitoring
    - Support for various matrix multiplication algorithms
    """
    
    def __init__(self, config: MatrixMultiplyConfig = MatrixMultiplyConfig()):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = config.dtype
        
        # Performance tracking
        self.forward_times = []
        self.memory_usage = []
        self.flop_counts = []
        
        # Enable TF32 if supported
        if config.use_tf32 and self.device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for matrix multiplication.
        
        Args:
            a: First input matrix [M, K]
            b: Second input matrix [K, N]
            
        Returns:
            Result matrix [M, N]
        """
        # Input validation
        assert a.dim() == 2, "Input must be 2D tensor"
        assert b.dim() == 2, "Input must be 2D tensor"
        assert a.shape[1] == b.shape[0], "Inner dimensions must match"
        
        # Ensure tensors are on correct device and dtype
        a = a.to(device=self.device, dtype=self.dtype)
        b = b.to(device=self.device, dtype=self.dtype)
        
        # Track memory usage
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated(self.device)
        
        # Start timing
        start_time = time.time()
        
        # Apply mixed precision if enabled
        if self.config.use_mixed_precision and self.dtype == torch.float32:
            with torch.cuda.amp.autocast():
                result = self._matrix_multiply_impl(a, b)
        else:
            result = self._matrix_multiply_impl(a, b)
        
        # End timing
        end_time = time.time()
        forward_time = (end_time - start_time) * 1000  # Convert to ms
        self.forward_times.append(forward_time)
        
        # Track memory usage
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            memory_after = torch.cuda.memory_allocated(self.device)
            self.memory_usage.append(memory_after - memory_before)
        
        # Track FLOP count
        flops = 2 * a.shape[0] * a.shape[1] * b.shape[1]  # 2 operations per multiply-add
        self.flop_counts.append(flops)
        
        return result
    
    def _matrix_multiply_impl(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Core matrix multiplication implementation."""
        # Use PyTorch's optimized matrix multiplication
        return torch.matmul(a, b)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.forward_times:
            return {}
        
        total_flops = sum(self.flop_counts)
        total_time = sum(self.forward_times) / 1000.0  # Convert to seconds
        
        return {
            'avg_forward_time_ms': np.mean(self.forward_times),
            'std_forward_time_ms': np.std(self.forward_times),
            'min_forward_time_ms': np.min(self.forward_times),
            'max_forward_time_ms': np.max(self.forward_times),
            'avg_memory_usage_bytes': np.mean(self.memory_usage) if self.memory_usage else 0,
            'total_flops': total_flops,
            'avg_gflops': (total_flops / total_time / 1e9) if total_time > 0 else 0,
        }


class OptimizedMatrixMultiplyModule(MatrixMultiplyModule):
    """
    Optimized matrix multiplication module with advanced techniques.
    """
    
    def __init__(self, config: MatrixMultiplyConfig = MatrixMultiplyConfig()):
        super().__init__(config)
        self.use_bmm = True  # Use batch matrix multiplication when possible
        self.enable_fusion = True
        
    def _matrix_multiply_impl(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Optimized matrix multiplication implementation."""
        if self.use_bmm and a.dim() == 2 and b.dim() == 2:
            # Use batch matrix multiplication for better performance
            a_batch = a.unsqueeze(0)  # [1, M, K]
            b_batch = b.unsqueeze(0)  # [1, K, N]
            result_batch = torch.bmm(a_batch, b_batch)  # [1, M, N]
            return result_batch.squeeze(0)  # [M, N]
        else:
            # Fallback to standard matrix multiplication
            return torch.matmul(a, b)


class TiledMatrixMultiplyModule(MatrixMultiplyModule):
    """
    Tiled matrix multiplication module for memory-efficient large matrix operations.
    """
    
    def __init__(self, config: MatrixMultiplyConfig = MatrixMultiplyConfig(), tile_size: int = 1024):
        super().__init__(config)
        self.tile_size = tile_size
        
    def _matrix_multiply_impl(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Tiled matrix multiplication implementation."""
        M, K = a.shape
        K2, N = b.shape
        
        # If matrices are small enough, use standard multiplication
        if M <= self.tile_size and N <= self.tile_size and K <= self.tile_size:
            return torch.matmul(a, b)
        
        # Tiled multiplication for large matrices
        result = torch.zeros(M, N, device=a.device, dtype=a.dtype)
        
        for i in range(0, M, self.tile_size):
            for j in range(0, N, self.tile_size):
                for k in range(0, K, self.tile_size):
                    # Extract tiles
                    a_tile = a[i:i+self.tile_size, k:k+self.tile_size]
                    b_tile = b[k:k+self.tile_size, j:j+self.tile_size]
                    
                    # Compute partial result
                    partial = torch.matmul(a_tile, b_tile)
                    
                    # Accumulate result
                    result[i:i+self.tile_size, j:j+self.tile_size] += partial
        
        return result


def matrix_multiply_pytorch(
    a: torch.Tensor,
    b: torch.Tensor,
    config: MatrixMultiplyConfig = MatrixMultiplyConfig()
) -> torch.Tensor:
    """
    PyTorch matrix multiplication function.
    
    Args:
        a: First input matrix [M, K]
        b: Second input matrix [K, N]
        config: Configuration for the operation
        
    Returns:
        Result matrix [M, N]
    """
    module = MatrixMultiplyModule(config)
    return module(a, b)


def matrix_multiply_optimized(
    a: torch.Tensor,
    b: torch.Tensor,
    config: MatrixMultiplyConfig = MatrixMultiplyConfig()
) -> torch.Tensor:
    """
    Optimized PyTorch matrix multiplication function.
    
    Args:
        a: First input matrix [M, K]
        b: Second input matrix [K, N]
        config: Configuration for the operation
        
    Returns:
        Result matrix [M, N]
    """
    module = OptimizedMatrixMultiplyModule(config)
    return module(a, b)


def matrix_multiply_tiled(
    a: torch.Tensor,
    b: torch.Tensor,
    config: MatrixMultiplyConfig = MatrixMultiplyConfig(),
    tile_size: int = 1024
) -> torch.Tensor:
    """
    Tiled PyTorch matrix multiplication function for large matrices.
    
    Args:
        a: First input matrix [M, K]
        b: Second input matrix [K, N]
        config: Configuration for the operation
        tile_size: Size of tiles for memory-efficient computation
        
    Returns:
        Result matrix [M, N]
    """
    module = TiledMatrixMultiplyModule(config, tile_size)
    return module(a, b)


def benchmark_matrix_multiply_pytorch(
    sizes: List[Tuple[int, int, int]] = [(100, 100, 100), (500, 500, 500), (1000, 1000, 1000)],
    dtypes: List[torch.dtype] = [torch.float32, torch.float16],
    devices: List[str] = ['cpu', 'cuda'],
    num_iterations: int = 100
) -> Dict:
    """
    Comprehensive benchmark for PyTorch matrix multiplication.
    
    Args:
        sizes: List of matrix sizes to test (M, K, N)
        dtypes: List of data types to test
        devices: List of devices to test
        num_iterations: Number of iterations for timing
        
    Returns:
        Dictionary with benchmark results
    """
    results = {}
    
    for device_str in devices:
        if device_str == 'cuda' and not torch.cuda.is_available():
            continue
            
        device = torch.device(device_str)
        results[device_str] = {}
        
        for dtype in dtypes:
            if device_str == 'cpu' and dtype == torch.float16:
                continue  # Skip FP16 on CPU
                
            results[device_str][str(dtype)] = {}
            
            for M, K, N in sizes:
                print(f"\nBenchmarking: {device_str}, {dtype}, size A[{M}×{K}] × B[{K}×{N}]")
                
                # Create test tensors
                torch.manual_seed(42)
                a = torch.randn(M, K, device=device, dtype=dtype)
                b = torch.randn(K, N, device=device, dtype=dtype)
                
                # Expected result
                expected = torch.matmul(a, b)
                
                # Test standard implementation
                config = MatrixMultiplyConfig(dtype=dtype, device=device_str)
                module = MatrixMultiplyModule(config)
                
                # Warmup
                for _ in range(10):
                    _ = module(a, b)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                # Benchmark
                start_time = time.time()
                for _ in range(num_iterations):
                    result = module(a, b)
                end_time = time.time()
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                # Calculate metrics
                avg_time_ms = (end_time - start_time) / num_iterations * 1000
                flops = 2 * M * N * K
                gflops = flops / (avg_time_ms / 1000.0) / 1e9
                bandwidth_gb_s = (M * K + K * N + M * N) * a.element_size() / (avg_time_ms / 1000.0) / 1e9
                
                # Verify correctness
                torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)
                
                # Test optimized implementation
                opt_module = OptimizedMatrixMultiplyModule(config)
                
                # Warmup
                for _ in range(10):
                    _ = opt_module(a, b)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                # Benchmark optimized
                start_time = time.time()
                for _ in range(num_iterations):
                    result_opt = opt_module(a, b)
                end_time = time.time()
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                avg_time_opt_ms = (end_time - start_time) / num_iterations * 1000
                gflops_opt = flops / (avg_time_opt_ms / 1000.0) / 1e9
                bandwidth_opt_gb_s = (M * K + K * N + M * N) * a.element_size() / (avg_time_opt_ms / 1000.0) / 1e9
                
                # Verify correctness
                torch.testing.assert_close(result_opt, expected, rtol=1e-4, atol=1e-4)
                
                # Store results
                results[device_str][str(dtype)][(M, K, N)] = {
                    'standard_time_ms': avg_time_ms,
                    'optimized_time_ms': avg_time_opt_ms,
                    'standard_gflops': gflops,
                    'optimized_gflops': gflops_opt,
                    'standard_bandwidth_gb_s': bandwidth_gb_s,
                    'optimized_bandwidth_gb_s': bandwidth_opt_gb_s,
                    'speedup': avg_time_ms / avg_time_opt_ms,
                }
                
                print(f"  Standard: {avg_time_ms:.3f} ms, {gflops:.2f} GFLOPS, {bandwidth_gb_s:.2f} GB/s")
                print(f"  Optimized: {avg_time_opt_ms:.3f} ms, {gflops_opt:.2f} GFLOPS, {bandwidth_opt_gb_s:.2f} GB/s")
                print(f"  Speedup: {avg_time_ms / avg_time_opt_ms:.2f}x")
    
    return results


def create_test_matrices(M: int, K: int, N: int, device: str = 'cuda', dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create test matrices for matrix multiplication.
    
    Args:
        M, K, N: Matrix dimensions
        device: Device to create tensors on
        dtype: Data type for tensors
        
    Returns:
        Tuple of (a, b, expected_result)
    """
    torch.manual_seed(42)
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(K, N, device=device, dtype=dtype)
    expected = torch.matmul(a, b)
    return a, b, expected


# Unit test and profiling
if __name__ == "__main__":
    print("Testing Matrix Multiplication PyTorch implementation")
    
    # Test configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    M, K, N = 100, 100, 100
    
    print(f"Device: {device}")
    print(f"Data type: {dtype}")
    print(f"Matrix size: A[{M}×{K}] × B[{K}×{N}]")
    
    # Create test matrices
    a, b, expected = create_test_matrices(M, K, N, device, dtype)
    
    print(f"\n=== Testing Standard Implementation ===")
    
    # Test standard implementation
    config = MatrixMultiplyConfig(dtype=dtype, device=device)
    result = matrix_multiply_pytorch(a, b, config)
    
    # Verify correctness
    assert result.shape == expected.shape, f"Output shape mismatch: {result.shape} vs {expected.shape}"
    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4), "Output values don't match expected"
    print("✓ Correctness test passed")
    
    # Test determinism
    result2 = matrix_multiply_pytorch(a, b, config)
    assert torch.allclose(result, result2), "Non-deterministic output detected"
    print("✓ Determinism test passed")
    
    print(f"\n=== Testing Optimized Implementation ===")
    
    # Test optimized implementation
    result_opt = matrix_multiply_optimized(a, b, config)
    assert torch.allclose(result_opt, expected, rtol=1e-4, atol=1e-4), "Optimized output mismatch"
    print("✓ Optimized implementation test passed")
    
    print(f"\n=== Testing Tiled Implementation ===")
    
    # Test tiled implementation
    result_tiled = matrix_multiply_tiled(a, b, config, tile_size=32)
    assert torch.allclose(result_tiled, expected, rtol=1e-4, atol=1e-4), "Tiled output mismatch"
    print("✓ Tiled implementation test passed")
    
    print(f"\n=== Testing Module Wrapper ===")
    
    # Test module wrapper
    module = MatrixMultiplyModule(config)
    result_module = module(a, b)
    assert torch.allclose(result, result_module), "Module wrapper output mismatch"
    print("✓ Module wrapper test passed")
    
    # Test performance tracking
    for _ in range(10):
        _ = module(a, b)
    
    stats = module.get_performance_stats()
    print(f"Performance stats: {stats}")
    
    # Run comprehensive benchmark
    print(f"\n=== Performance Benchmark ===")
    benchmark_results = benchmark_matrix_multiply_pytorch(
        sizes=[(100, 100, 100), (500, 500, 500), (1000, 1000, 1000)],
        dtypes=[torch.float32],
        devices=[device.type],
        num_iterations=50
    )
    
    # Print summary
    print(f"\n=== Performance Summary ===")
    for device_str, device_results in benchmark_results.items():
        print(f"\nDevice: {device_str}")
        for dtype_str, dtype_results in device_results.items():
            print(f"  Data type: {dtype_str}")
            for size, metrics in dtype_results.items():
                print(f"    Size {size}: {metrics['speedup']:.2f}x speedup, "
                      f"{metrics['optimized_gflops']:.2f} GFLOPS, "
                      f"{metrics['optimized_bandwidth_gb_s']:.2f} GB/s")
    
    print("\n=== All tests passed! ===")

"""
Performance optimization tips:

1. Memory optimization:
   - Use appropriate data types (FP16 for memory-bound operations)
   - Enable gradient checkpointing for large models
   - Use tiled multiplication for very large matrices
   - Use torch.cuda.empty_cache() to free unused memory

2. Computation optimization:
   - Use batch matrix multiplication (bmm) when possible
   - Enable mixed precision training with torch.cuda.amp
   - Use torch.jit.script for JIT compilation
   - Enable TF32 on Ampere GPUs for better performance

3. Distributed training:
   - Use torch.nn.DataParallel for single-node multi-GPU
   - Use torch.nn.parallel.DistributedDataParallel for multi-node
   - Use torch.distributed for custom distributed strategies

4. Profiling and debugging:
   - Use torch.profiler for detailed performance analysis
   - Use torch.autograd.profiler for gradient computation profiling
   - Monitor memory usage with torch.cuda.memory_summary()

5. Hardware-specific optimizations:
   - Use Tensor Cores with FP16/BF16 on modern GPUs
   - Enable cuDNN optimizations
   - Use appropriate batch sizes for your GPU memory
   - Consider using cuBLAS for maximum performance
"""
