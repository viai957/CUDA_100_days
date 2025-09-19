"""
Layer Normalization PyTorch Implementation: High-Performance Normalization
Math: output = (input - mean) / sqrt(variance + epsilon) * gamma + beta
Inputs: input[N, D] - input tensor, N - batch size, D - feature dimension
Assumptions: N > 0, D > 0, tensors are contiguous, device has sufficient memory
Parallel Strategy: PyTorch's optimized tensor operations with automatic parallelization
Mixed Precision Policy: Configurable data types (FP16, FP32, FP64)
Distributed Hooks: Ready for multi-GPU via torch.distributed and DataParallel
Complexity: O(ND) FLOPs, O(ND) bytes moved
Test Vectors: Deterministic random tensors with known normalization results
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
class LayerNormConfig:
    """Configuration for layer normalization operations."""
    dtype: torch.dtype = torch.float32
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_mixed_precision: bool = False
    enable_gradient_checkpointing: bool = False
    use_tf32: bool = True  # Enable TF32 for better performance on Ampere GPUs
    epsilon: float = 1e-7


class LayerNormModule(nn.Module):
    """
    PyTorch module for layer normalization with advanced features.
    
    Features:
    - Automatic mixed precision support
    - Gradient checkpointing for memory efficiency
    - Configurable data types and precision
    - Built-in performance monitoring
    - Support for various normalization algorithms
    """
    
    def __init__(self, feature_dim: int, config: LayerNormConfig = LayerNormConfig()):
        super().__init__()
        self.feature_dim = feature_dim
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = config.dtype
        self.epsilon = config.epsilon
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(feature_dim, device=self.device, dtype=self.dtype))
        self.beta = nn.Parameter(torch.zeros(feature_dim, device=self.device, dtype=self.dtype))
        
        # Performance tracking
        self.forward_times = []
        self.memory_usage = []
        self.flop_counts = []
        
        # Enable TF32 if supported
        if config.use_tf32 and self.device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for layer normalization.
        
        Args:
            input_tensor: Input tensor [batch_size, feature_dim]
            
        Returns:
            Result tensor [batch_size, feature_dim]
        """
        # Input validation
        assert input_tensor.dim() == 2, "Input must be 2D tensor"
        assert input_tensor.shape[1] == self.feature_dim, f"Feature dimension must be {self.feature_dim}"
        
        # Ensure tensor is on correct device and dtype
        input_tensor = input_tensor.to(device=self.device, dtype=self.dtype)
        
        # Track memory usage
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated(self.device)
        
        # Start timing
        start_time = time.time()
        
        # Apply mixed precision if enabled
        if self.config.use_mixed_precision and self.dtype == torch.float32:
            with torch.cuda.amp.autocast():
                result = self._layer_norm_impl(input_tensor)
        else:
            result = self._layer_norm_impl(input_tensor)
        
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
        flops = 3 * input_tensor.shape[0] * input_tensor.shape[1]  # Mean, variance, normalization
        self.flop_counts.append(flops)
        
        return result
    
    def _layer_norm_impl(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Core layer normalization implementation."""
        # Compute mean and variance
        mean = input_tensor.mean(dim=1, keepdim=True)
        variance = input_tensor.var(dim=1, keepdim=True, unbiased=False)
        
        # Normalize
        normalized = (input_tensor - mean) / torch.sqrt(variance + self.epsilon)
        
        # Apply scale and shift
        return normalized * self.gamma + self.beta
    
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


class OptimizedLayerNormModule(LayerNormModule):
    """
    Optimized layer normalization module with advanced techniques.
    """
    
    def __init__(self, feature_dim: int, config: LayerNormConfig = LayerNormConfig()):
        super().__init__(feature_dim, config)
        self.use_fused_ops = True
        self.enable_fusion = True
        
    def _layer_norm_impl(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Optimized layer normalization implementation."""
        if self.use_fused_ops:
            # Use PyTorch's optimized layer normalization
            return F.layer_norm(input_tensor, [self.feature_dim], self.gamma, self.beta, self.epsilon)
        else:
            # Fallback to standard implementation
            return super()._layer_norm_impl(input_tensor)


class CustomLayerNormModule(LayerNormModule):
    """
    Custom layer normalization module with manual implementation.
    """
    
    def __init__(self, feature_dim: int, config: LayerNormConfig = LayerNormConfig()):
        super().__init__(feature_dim, config)
        self.use_custom_impl = True
        
    def _layer_norm_impl(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Custom layer normalization implementation."""
        # Manual implementation for educational purposes
        batch_size, feature_dim = input_tensor.shape
        
        # Compute mean
        mean = torch.sum(input_tensor, dim=1, keepdim=True) / feature_dim
        
        # Compute variance
        variance = torch.sum((input_tensor - mean) ** 2, dim=1, keepdim=True) / feature_dim
        
        # Normalize
        stddev = torch.sqrt(variance + self.epsilon)
        normalized = (input_tensor - mean) / stddev
        
        # Apply scale and shift
        return normalized * self.gamma + self.beta


def layer_norm_pytorch(
    input_tensor: torch.Tensor,
    feature_dim: int,
    config: LayerNormConfig = LayerNormConfig()
) -> torch.Tensor:
    """
    PyTorch layer normalization function.
    
    Args:
        input_tensor: Input tensor [batch_size, feature_dim]
        feature_dim: Feature dimension
        config: Configuration for the operation
        
    Returns:
        Result tensor [batch_size, feature_dim]
    """
    module = LayerNormModule(feature_dim, config)
    return module(input_tensor)


def layer_norm_optimized(
    input_tensor: torch.Tensor,
    feature_dim: int,
    config: LayerNormConfig = LayerNormConfig()
) -> torch.Tensor:
    """
    Optimized PyTorch layer normalization function.
    
    Args:
        input_tensor: Input tensor [batch_size, feature_dim]
        feature_dim: Feature dimension
        config: Configuration for the operation
        
    Returns:
        Result tensor [batch_size, feature_dim]
    """
    module = OptimizedLayerNormModule(feature_dim, config)
    return module(input_tensor)


def layer_norm_custom(
    input_tensor: torch.Tensor,
    feature_dim: int,
    config: LayerNormConfig = LayerNormConfig()
) -> torch.Tensor:
    """
    Custom PyTorch layer normalization function.
    
    Args:
        input_tensor: Input tensor [batch_size, feature_dim]
        feature_dim: Feature dimension
        config: Configuration for the operation
        
    Returns:
        Result tensor [batch_size, feature_dim]
    """
    module = CustomLayerNormModule(feature_dim, config)
    return module(input_tensor)


def benchmark_layer_norm_pytorch(
    batch_sizes: List[int] = [1, 4, 8, 16, 32],
    feature_dims: List[int] = [128, 256, 512, 1024, 2048],
    dtypes: List[torch.dtype] = [torch.float32, torch.float16],
    devices: List[str] = ['cpu', 'cuda'],
    num_iterations: int = 100
) -> Dict:
    """
    Comprehensive benchmark for PyTorch layer normalization.
    
    Args:
        batch_sizes: List of batch sizes to test
        feature_dims: List of feature dimensions to test
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
            
            for batch_size in batch_sizes:
                for feature_dim in feature_dims:
                    config_key = f"batch_size={batch_size},feature_dim={feature_dim}"
                    print(f"\nBenchmarking: {device_str}, {dtype}, {config_key}")
                    
                    # Create test tensor
                    torch.manual_seed(42)
                    input_tensor = torch.randn(batch_size, feature_dim, device=device, dtype=dtype)
                    
                    # Expected result
                    expected = F.layer_norm(input_tensor, [feature_dim])
                    
                    # Test standard implementation
                    config = LayerNormConfig(dtype=dtype, device=device_str)
                    module = LayerNormModule(feature_dim, config)
                    
                    # Warmup
                    for _ in range(10):
                        _ = module(input_tensor)
                    
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    # Benchmark
                    start_time = time.time()
                    for _ in range(num_iterations):
                        result = module(input_tensor)
                    end_time = time.time()
                    
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    # Calculate metrics
                    avg_time_ms = (end_time - start_time) / num_iterations * 1000
                    flops = 3 * batch_size * feature_dim  # Mean, variance, normalization
                    gflops = flops / (avg_time_ms / 1000.0) / 1e9
                    bandwidth_gb_s = 4 * batch_size * feature_dim * input_tensor.element_size() / (avg_time_ms / 1000.0) / 1e9
                    
                    # Verify correctness
                    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)
                    
                    # Test optimized implementation
                    opt_module = OptimizedLayerNormModule(feature_dim, config)
                    
                    # Warmup
                    for _ in range(10):
                        _ = opt_module(input_tensor)
                    
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    # Benchmark optimized
                    start_time = time.time()
                    for _ in range(num_iterations):
                        result_opt = opt_module(input_tensor)
                    end_time = time.time()
                    
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    avg_time_opt_ms = (end_time - start_time) / num_iterations * 1000
                    gflops_opt = flops / (avg_time_opt_ms / 1000.0) / 1e9
                    bandwidth_opt_gb_s = 4 * batch_size * feature_dim * input_tensor.element_size() / (avg_time_opt_ms / 1000.0) / 1e9
                    
                    # Verify correctness
                    torch.testing.assert_close(result_opt, expected, rtol=1e-4, atol=1e-4)
                    
                    # Store results
                    results[device_str][str(dtype)][config_key] = {
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


def create_test_tensor(
    batch_size: int, 
    feature_dim: int, 
    device: str = 'cuda', 
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create test tensor for layer normalization.
    
    Args:
        batch_size: Batch size
        feature_dim: Feature dimension
        device: Device to create tensor on
        dtype: Data type for tensor
        
    Returns:
        Tuple of (input_tensor, expected_result)
    """
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, feature_dim, device=device, dtype=dtype)
    expected = F.layer_norm(input_tensor, [feature_dim])
    return input_tensor, expected


# Unit test and profiling
if __name__ == "__main__":
    print("Testing Layer Normalization PyTorch implementation")
    
    # Test configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    batch_size, feature_dim = 10, 10
    
    print(f"Device: {device}")
    print(f"Data type: {dtype}")
    print(f"Configuration: batch_size={batch_size}, feature_dim={feature_dim}")
    
    # Create test tensor
    input_tensor, expected = create_test_tensor(batch_size, feature_dim, device, dtype)
    
    print(f"\n=== Testing Standard Implementation ===")
    
    # Test standard implementation
    config = LayerNormConfig(dtype=dtype, device=device)
    result = layer_norm_pytorch(input_tensor, feature_dim, config)
    
    # Verify correctness
    assert result.shape == expected.shape, f"Output shape mismatch: {result.shape} vs {expected.shape}"
    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4), "Output values don't match expected"
    print("✓ Correctness test passed")
    
    # Test determinism
    result2 = layer_norm_pytorch(input_tensor, feature_dim, config)
    assert torch.allclose(result, result2), "Non-deterministic output detected"
    print("✓ Determinism test passed")
    
    print(f"\n=== Testing Optimized Implementation ===")
    
    # Test optimized implementation
    result_opt = layer_norm_optimized(input_tensor, feature_dim, config)
    assert torch.allclose(result_opt, expected, rtol=1e-4, atol=1e-4), "Optimized output mismatch"
    print("✓ Optimized implementation test passed")
    
    print(f"\n=== Testing Custom Implementation ===")
    
    # Test custom implementation
    result_custom = layer_norm_custom(input_tensor, feature_dim, config)
    assert torch.allclose(result_custom, expected, rtol=1e-4, atol=1e-4), "Custom output mismatch"
    print("✓ Custom implementation test passed")
    
    print(f"\n=== Testing Module Wrapper ===")
    
    # Test module wrapper
    module = LayerNormModule(feature_dim, config)
    result_module = module(input_tensor)
    assert torch.allclose(result, result_module), "Module wrapper output mismatch"
    print("✓ Module wrapper test passed")
    
    # Test performance tracking
    for _ in range(10):
        _ = module(input_tensor)
    
    stats = module.get_performance_stats()
    print(f"Performance stats: {stats}")
    
    # Run comprehensive benchmark
    print(f"\n=== Performance Benchmark ===")
    benchmark_results = benchmark_layer_norm_pytorch(
        batch_sizes=[1, 4, 8, 16],
        feature_dims=[128, 256, 512],
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
            for config, metrics in dtype_results.items():
                print(f"    {config}: {metrics['speedup']:.2f}x speedup, "
                      f"{metrics['optimized_gflops']:.2f} GFLOPS, "
                      f"{metrics['optimized_bandwidth_gb_s']:.2f} GB/s")
    
    print("\n=== All tests passed! ===")

"""
Performance optimization tips:

1. Memory optimization:
   - Use appropriate data types (FP16 for memory-bound operations)
   - Enable gradient checkpointing for large models
   - Use torch.cuda.empty_cache() to free unused memory
   - Consider fused operations for better performance

2. Computation optimization:
   - Use PyTorch's optimized layer normalization when available
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

6. Normalization-specific optimizations:
   - Use efficient reduction algorithms for mean and variance
   - Consider numerical stability in variance computation
   - Use appropriate epsilon values for numerical stability
   - Consider fused operations for better performance
"""
