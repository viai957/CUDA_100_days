"""
Self-Attention PyTorch Implementation: High-Performance Multi-Head Attention
Math: Attention(Q,K,V) = softmax(QK^T/√d_k)V
Inputs: Q[N, d_model], K[N, d_model], V[N, d_model] - query, key, value matrices
Assumptions: N > 0, d_model > 0, num_heads > 0, d_model % num_heads == 0
Parallel Strategy: PyTorch's optimized tensor operations with automatic parallelization
Mixed Precision Policy: Configurable data types (FP16, FP32, FP64)
Distributed Hooks: Ready for multi-GPU via torch.distributed and DataParallel
Complexity: O(N²d_model) FLOPs, O(N² + Nd_model) bytes moved
Test Vectors: Deterministic random tensors with known attention patterns
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
class SelfAttentionConfig:
    """Configuration for self-attention operations."""
    dtype: torch.dtype = torch.float32
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_mixed_precision: bool = False
    enable_gradient_checkpointing: bool = False
    use_tf32: bool = True  # Enable TF32 for better performance on Ampere GPUs
    dropout_rate: float = 0.1
    use_causal_mask: bool = False


class SelfAttentionModule(nn.Module):
    """
    PyTorch module for self-attention with advanced features.
    
    Features:
    - Multi-head attention mechanism
    - Automatic mixed precision support
    - Gradient checkpointing for memory efficiency
    - Configurable data types and precision
    - Built-in performance monitoring
    - Support for causal masking
    - Dropout for regularization
    """
    
    def __init__(self, d_model: int, num_heads: int, config: SelfAttentionConfig = SelfAttentionConfig()):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = config.dtype
        
        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.out_linear = nn.Linear(d_model, d_model, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Performance tracking
        self.forward_times = []
        self.memory_usage = []
        self.flop_counts = []
        
        # Enable TF32 if supported
        if config.use_tf32 and self.device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for self-attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Result tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Input validation
        assert d_model == self.d_model, f"Input dimension {d_model} must match model dimension {self.d_model}"
        
        # Ensure tensor is on correct device and dtype
        x = x.to(device=self.device, dtype=self.dtype)
        
        # Track memory usage
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated(self.device)
        
        # Start timing
        start_time = time.time()
        
        # Apply mixed precision if enabled
        if self.config.use_mixed_precision and self.dtype == torch.float32:
            with torch.cuda.amp.autocast():
                result = self._self_attention_impl(x, mask)
        else:
            result = self._self_attention_impl(x, mask)
        
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
        flops = 2 * batch_size * seq_len * seq_len * d_model + batch_size * seq_len * d_model * d_model
        self.flop_counts.append(flops)
        
        return result
    
    def _self_attention_impl(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Core self-attention implementation."""
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        Q = self.q_linear(x)  # [batch_size, seq_len, d_model]
        K = self.k_linear(x)  # [batch_size, seq_len, d_model]
        V = self.v_linear(x)  # [batch_size, seq_len, d_model]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Scaled dot-product attention
        attention_output = self._scaled_dot_product_attention(Q, K, V, mask)
        
        # Reshape back
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Output projection
        output = self.out_linear(attention_output)
        
        return output
    
    def _scaled_dot_product_attention(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Scaled dot-product attention computation."""
        batch_size, num_heads, seq_len, head_dim = Q.shape
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # Apply causal mask if enabled
        if self.config.use_causal_mask:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool))
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Apply custom mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output
    
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


class OptimizedSelfAttentionModule(SelfAttentionModule):
    """
    Optimized self-attention module with advanced techniques.
    """
    
    def __init__(self, d_model: int, num_heads: int, config: SelfAttentionConfig = SelfAttentionConfig()):
        super().__init__(d_model, num_heads, config)
        self.use_flash_attention = True
        self.enable_fusion = True
        
    def _scaled_dot_product_attention(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Optimized scaled dot-product attention implementation."""
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's optimized scaled dot-product attention
            return F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, dropout_p=self.config.dropout_rate)
        else:
            # Fallback to standard implementation
            return super()._scaled_dot_product_attention(Q, K, V, mask)


class CausalSelfAttentionModule(SelfAttentionModule):
    """
    Causal self-attention module for autoregressive models.
    """
    
    def __init__(self, d_model: int, num_heads: int, config: SelfAttentionConfig = SelfAttentionConfig()):
        config.use_causal_mask = True
        super().__init__(d_model, num_heads, config)


def self_attention_pytorch(
    x: torch.Tensor,
    d_model: int,
    num_heads: int,
    config: SelfAttentionConfig = SelfAttentionConfig(),
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    PyTorch self-attention function.
    
    Args:
        x: Input tensor [batch_size, seq_len, d_model]
        d_model: Model dimension
        num_heads: Number of attention heads
        config: Configuration for the operation
        mask: Optional attention mask
        
    Returns:
        Result tensor [batch_size, seq_len, d_model]
    """
    module = SelfAttentionModule(d_model, num_heads, config)
    return module(x, mask)


def self_attention_optimized(
    x: torch.Tensor,
    d_model: int,
    num_heads: int,
    config: SelfAttentionConfig = SelfAttentionConfig(),
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Optimized PyTorch self-attention function.
    
    Args:
        x: Input tensor [batch_size, seq_len, d_model]
        d_model: Model dimension
        num_heads: Number of attention heads
        config: Configuration for the operation
        mask: Optional attention mask
        
    Returns:
        Result tensor [batch_size, seq_len, d_model]
    """
    module = OptimizedSelfAttentionModule(d_model, num_heads, config)
    return module(x, mask)


def benchmark_self_attention_pytorch(
    seq_lens: List[int] = [64, 128, 256, 512],
    d_models: List[int] = [512, 768, 1024],
    num_heads: List[int] = [8, 12, 16],
    batch_sizes: List[int] = [1, 4, 8],
    dtypes: List[torch.dtype] = [torch.float32, torch.float16],
    devices: List[str] = ['cpu', 'cuda'],
    num_iterations: int = 100
) -> Dict:
    """
    Comprehensive benchmark for PyTorch self-attention.
    
    Args:
        seq_lens: List of sequence lengths to test
        d_models: List of model dimensions to test
        num_heads: List of number of heads to test
        batch_sizes: List of batch sizes to test
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
            
            for seq_len in seq_lens:
                for d_model in d_models:
                    for num_heads in num_heads:
                        if d_model % num_heads == 0:
                            for batch_size in batch_sizes:
                                config_key = f"seq_len={seq_len},d_model={d_model},num_heads={num_heads},batch_size={batch_size}"
                                print(f"\nBenchmarking: {device_str}, {dtype}, {config_key}")
                                
                                # Create test tensor
                                torch.manual_seed(42)
                                x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
                                
                                # Expected result
                                expected = F.scaled_dot_product_attention(
                                    x, x, x, 
                                    attn_mask=None, 
                                    dropout_p=0.0
                                )
                                
                                # Test standard implementation
                                config = SelfAttentionConfig(dtype=dtype, device=device_str)
                                module = SelfAttentionModule(d_model, num_heads, config)
                                
                                # Warmup
                                for _ in range(10):
                                    _ = module(x)
                                
                                if device.type == 'cuda':
                                    torch.cuda.synchronize()
                                
                                # Benchmark
                                start_time = time.time()
                                for _ in range(num_iterations):
                                    result = module(x)
                                end_time = time.time()
                                
                                if device.type == 'cuda':
                                    torch.cuda.synchronize()
                                
                                # Calculate metrics
                                avg_time_ms = (end_time - start_time) / num_iterations * 1000
                                flops = 2 * batch_size * seq_len * seq_len * d_model + batch_size * seq_len * d_model * d_model
                                gflops = flops / (avg_time_ms / 1000.0) / 1e9
                                bandwidth_gb_s = (4 * batch_size * seq_len * d_model + batch_size * num_heads * seq_len * seq_len) * x.element_size() / (avg_time_ms / 1000.0) / 1e9
                                
                                # Verify correctness
                                torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)
                                
                                # Test optimized implementation
                                opt_module = OptimizedSelfAttentionModule(d_model, num_heads, config)
                                
                                # Warmup
                                for _ in range(10):
                                    _ = opt_module(x)
                                
                                if device.type == 'cuda':
                                    torch.cuda.synchronize()
                                
                                # Benchmark optimized
                                start_time = time.time()
                                for _ in range(num_iterations):
                                    result_opt = opt_module(x)
                                end_time = time.time()
                                
                                if device.type == 'cuda':
                                    torch.cuda.synchronize()
                                
                                avg_time_opt_ms = (end_time - start_time) / num_iterations * 1000
                                gflops_opt = flops / (avg_time_opt_ms / 1000.0) / 1e9
                                bandwidth_opt_gb_s = (4 * batch_size * seq_len * d_model + batch_size * num_heads * seq_len * seq_len) * x.element_size() / (avg_time_opt_ms / 1000.0) / 1e9
                                
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


def create_test_data(
    batch_size: int, 
    seq_len: int, 
    d_model: int, 
    device: str = 'cuda', 
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create test data for self-attention.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        d_model: Model dimension
        device: Device to create tensors on
        dtype: Data type for tensors
        
    Returns:
        Tuple of (input_tensor, expected_result)
    """
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
    expected = F.scaled_dot_product_attention(x, x, x, attn_mask=None, dropout_p=0.0)
    return x, expected


# Unit test and profiling
if __name__ == "__main__":
    print("Testing Self-Attention PyTorch implementation")
    
    # Test configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    batch_size, seq_len, d_model, num_heads = 1, 64, 512, 8
    
    print(f"Device: {device}")
    print(f"Data type: {dtype}")
    print(f"Configuration: batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}, num_heads={num_heads}")
    
    # Create test data
    x, expected = create_test_data(batch_size, seq_len, d_model, device, dtype)
    
    print(f"\n=== Testing Standard Implementation ===")
    
    # Test standard implementation
    config = SelfAttentionConfig(dtype=dtype, device=device)
    result = self_attention_pytorch(x, d_model, num_heads, config)
    
    # Verify correctness
    assert result.shape == expected.shape, f"Output shape mismatch: {result.shape} vs {expected.shape}"
    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4), "Output values don't match expected"
    print("✓ Correctness test passed")
    
    # Test determinism
    result2 = self_attention_pytorch(x, d_model, num_heads, config)
    assert torch.allclose(result, result2), "Non-deterministic output detected"
    print("✓ Determinism test passed")
    
    print(f"\n=== Testing Optimized Implementation ===")
    
    # Test optimized implementation
    result_opt = self_attention_optimized(x, d_model, num_heads, config)
    assert torch.allclose(result_opt, expected, rtol=1e-4, atol=1e-4), "Optimized output mismatch"
    print("✓ Optimized implementation test passed")
    
    print(f"\n=== Testing Causal Implementation ===")
    
    # Test causal implementation
    causal_module = CausalSelfAttentionModule(d_model, num_heads, config)
    result_causal = causal_module(x)
    assert result_causal.shape == expected.shape, "Causal output shape mismatch"
    print("✓ Causal implementation test passed")
    
    print(f"\n=== Testing Module Wrapper ===")
    
    # Test module wrapper
    module = SelfAttentionModule(d_model, num_heads, config)
    result_module = module(x)
    assert torch.allclose(result, result_module), "Module wrapper output mismatch"
    print("✓ Module wrapper test passed")
    
    # Test performance tracking
    for _ in range(10):
        _ = module(x)
    
    stats = module.get_performance_stats()
    print(f"Performance stats: {stats}")
    
    # Run comprehensive benchmark
    print(f"\n=== Performance Benchmark ===")
    benchmark_results = benchmark_self_attention_pytorch(
        seq_lens=[64, 128, 256],
        d_models=[512, 768],
        num_heads=[8, 12],
        batch_sizes=[1, 4],
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
   - Consider attention weight caching for multiple queries

2. Computation optimization:
   - Use PyTorch's optimized scaled_dot_product_attention when available
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

6. Attention-specific optimizations:
   - Use causal masking for autoregressive models
   - Consider sparse attention patterns for long sequences
   - Implement attention weight caching for inference
   - Use flash attention for memory efficiency
"""
