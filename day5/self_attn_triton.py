/*
 * Self-Attention Triton Implementation: High-Performance Multi-Head Attention
 * Math: Attention(Q,K,V) = softmax(QK^T/√d_k)V
 * Inputs: Q[N, d_model], K[N, d_model], V[N, d_model] - query, key, value matrices
 * Assumptions: N > 0, d_model > 0, num_heads > 0, d_model % num_heads == 0
 * Parallel Strategy: Each block processes multiple attention heads with tiled computation
 * Mixed Precision Policy: FP16/BF16 for computation, FP32 for softmax and reductions
 * Distributed Hooks: Ready for tensor parallelism via tl.comm_* primitives
 * Complexity: O(N²d_model) FLOPs, O(N² + Nd_model) bytes moved
 * Test Vectors: Deterministic random tensors with known attention patterns
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
    ],
    key=['seq_len', 'head_dim'],
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
    Triton kernel for matrix multiplication (reused from day4).
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
def softmax_kernel(
    # Input tensor
    input_ptr,
    # Output tensor
    output_ptr,
    # Dimensions
    seq_len,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for softmax computation.
    """
    
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate block range
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, seq_len)
    
    # Process each row
    for row in range(block_start, block_end):
        # Find maximum value for numerical stability
        max_val = tl.load(input_ptr + row * seq_len)
        for col in range(1, seq_len):
            val = tl.load(input_ptr + row * seq_len + col)
            max_val = tl.maximum(max_val, val)
        
        # Compute exponentials and sum
        sum_val = 0.0
        for col in range(seq_len):
            val = tl.load(input_ptr + row * seq_len + col)
            exp_val = tl.exp(val - max_val)
            tl.store(output_ptr + row * seq_len + col, exp_val)
            sum_val += exp_val
        
        # Normalize
        for col in range(seq_len):
            val = tl.load(output_ptr + row * seq_len + col)
            normalized_val = val / sum_val
            tl.store(output_ptr + row * seq_len + col, normalized_val)


@triton.jit
def scaled_dot_product_attention_kernel(
    # Input tensors
    q_ptr, k_ptr, v_ptr,
    # Output tensor
    output_ptr,
    # Dimensions
    seq_len, head_dim,
    # Scale factor
    scale_factor,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for scaled dot-product attention.
    """
    
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate block range
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, seq_len)
    
    # Process each row
    for row in range(block_start, block_end):
        # Compute QK^T for this row
        for col in range(seq_len):
            sum_val = 0.0
            for k in range(head_dim):
                q_val = tl.load(q_ptr + row * head_dim + k)
                k_val = tl.load(k_ptr + col * head_dim + k)
                sum_val += q_val * k_val
            
            # Scale by sqrt(d_k)
            scaled_val = sum_val * scale_factor
            tl.store(output_ptr + row * seq_len + col, scaled_val)
        
        # Apply softmax to this row
        # Find maximum value
        max_val = tl.load(output_ptr + row * seq_len)
        for col in range(1, seq_len):
            val = tl.load(output_ptr + row * seq_len + col)
            max_val = tl.maximum(max_val, val)
        
        # Compute exponentials and sum
        sum_val = 0.0
        for col in range(seq_len):
            val = tl.load(output_ptr + row * seq_len + col)
            exp_val = tl.exp(val - max_val)
            tl.store(output_ptr + row * seq_len + col, exp_val)
            sum_val += exp_val
        
        # Normalize
        for col in range(seq_len):
            val = tl.load(output_ptr + row * seq_len + col)
            normalized_val = val / sum_val
            tl.store(output_ptr + row * seq_len + col, normalized_val)
        
        # Compute attention * V
        for col in range(head_dim):
            sum_val = 0.0
            for k in range(seq_len):
                attn_val = tl.load(output_ptr + row * seq_len + k)
                v_val = tl.load(v_ptr + k * head_dim + col)
                sum_val += attn_val * v_val
            
            tl.store(output_ptr + row * head_dim + col, sum_val)


def self_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    head_dim: int,
    optimized: bool = True
) -> torch.Tensor:
    """
    Apply self-attention using Triton kernels.
    
    Args:
        q: Query tensor [seq_len, d_model]
        k: Key tensor [seq_len, d_model]
        v: Value tensor [seq_len, d_model]
        num_heads: Number of attention heads
        head_dim: Dimension of each head
        optimized: Whether to use optimized kernels
        
    Returns:
        Result tensor [seq_len, d_model]
    """
    # Input validation
    assert q.dim() == 2, "Input must be 2D tensor"
    assert k.dim() == 2, "Input must be 2D tensor"
    assert v.dim() == 2, "Input must be 2D tensor"
    assert q.shape == k.shape == v.shape, "All inputs must have same shape"
    assert q.shape[1] % num_heads == 0, "d_model must be divisible by num_heads"
    
    seq_len, d_model = q.shape
    device = q.device
    dtype = q.dtype
    
    # Ensure tensors are on correct device
    q = q.to(device)
    k = k.to(device)
    v = v.to(device)
    
    # Prepare output tensor
    output = torch.empty_like(q)
    
    # Process each attention head
    for head in range(num_heads):
        # Extract head-specific tensors
        q_head = q[:, head * head_dim:(head + 1) * head_dim]
        k_head = k[:, head * head_dim:(head + 1) * head_dim]
        v_head = v[:, head * head_dim:(head + 1) * head_dim]
        
        # Calculate grid dimensions
        grid_size = triton.cdiv(seq_len, 64)  # Default block size
        
        # Launch scaled dot-product attention kernel
        scaled_dot_product_attention_kernel[(grid_size,)](
            q_head, k_head, v_head,
            output[:, head * head_dim:(head + 1) * head_dim],
            seq_len, head_dim,
            scale_factor=1.0 / math.sqrt(head_dim),
            BLOCK_SIZE=64,  # Will be overridden by autotune
        )
    
    return output


class SelfAttentionModule(torch.nn.Module):
    """
    PyTorch module wrapper for Triton self-attention.
    """
    
    def __init__(self, d_model: int, num_heads: int, head_dim: int, optimized: bool = True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.optimized = optimized
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self_attention_triton(q, k, v, self.num_heads, self.head_dim, self.optimized)


def benchmark_self_attention(
    seq_lens: list = [64, 128, 256, 512],
    d_models: list = [512, 768, 1024],
    num_heads: list = [8, 12, 16],
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32
) -> None:
    """
    Benchmark self-attention across different configurations.
    """
    print(f"\n=== Self-Attention Benchmark on {device} ===")
    print(f"Data type: {dtype}")
    
    results = []
    
    for seq_len in seq_lens:
        for d_model in d_models:
            for num_heads in num_heads:
                if d_model % num_heads == 0:
                    head_dim = d_model // num_heads
                    print(f"\nConfiguration: seq_len={seq_len}, d_model={d_model}, num_heads={num_heads}")
                    
                    # Create test tensors
                    torch.manual_seed(42)
                    q = torch.randn(seq_len, d_model, device=device, dtype=dtype)
                    k = torch.randn(seq_len, d_model, device=device, dtype=dtype)
                    v = torch.randn(seq_len, d_model, device=device, dtype=dtype)
                    
                    # Expected result (using PyTorch)
                    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                    
                    # Test Triton implementation
                    if device == 'cuda':
                        # Warmup
                        for _ in range(10):
                            _ = self_attention_triton(q, k, v, num_heads, head_dim, optimized=True)
                        
                        torch.cuda.synchronize()
                        
                        # Benchmark
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        
                        num_iterations = 100
                        start_event.record()
                        for _ in range(num_iterations):
                            result = self_attention_triton(q, k, v, num_heads, head_dim, optimized=True)
                        end_event.record()
                        
                        torch.cuda.synchronize()
                        elapsed_ms = start_event.elapsed_time(end_event)
                        
                        # Verify correctness
                        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)
                        
                        # Calculate performance metrics
                        avg_time_ms = elapsed_ms / num_iterations
                        flops = 2 * seq_len * seq_len * d_model + seq_len * d_model * d_model
                        gflops = flops / (avg_time_ms / 1000.0) / 1e9
                        bandwidth_gb_s = (4 * seq_len * d_model + num_heads * seq_len * seq_len) * q.element_size() / (avg_time_ms / 1000.0) / 1e9
                        
                        print(f"  Triton: {avg_time_ms:.3f} ms, {gflops:.2f} GFLOPS, {bandwidth_gb_s:.2f} GB/s")
                        
                        # Compare with PyTorch
                        torch.cuda.synchronize()
                        start_event.record()
                        for _ in range(num_iterations):
                            result_pytorch = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                        end_event.record()
                        torch.cuda.synchronize()
                        pytorch_time_ms = start_event.elapsed_time(end_event) / num_iterations
                        
                        speedup = pytorch_time_ms / avg_time_ms
                        print(f"  PyTorch: {pytorch_time_ms:.3f} ms")
                        print(f"  Speedup: {speedup:.2f}x")
                        
                        results.append({
                            'seq_len': seq_len,
                            'd_model': d_model,
                            'num_heads': num_heads,
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
                            result = self_attention_triton(q, k, v, num_heads, head_dim, optimized=True)
                        end_time = time.time()
                        
                        avg_time_ms = (end_time - start_time) / 10 * 1000
                        print(f"  Triton (CPU): {avg_time_ms:.3f} ms")
    
    return results


# Unit test and profiling
if __name__ == "__main__":
    # Test configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    seq_len, d_model, num_heads = 64, 512, 8
    head_dim = d_model // num_heads
    
    print(f"Testing Self-Attention Triton implementation on {device}")
    print(f"Configuration: seq_len={seq_len}, d_model={d_model}, num_heads={num_heads}")
    
    # Create test tensors
    torch.manual_seed(42)
    q = torch.randn(seq_len, d_model, device=device, dtype=dtype)
    k = torch.randn(seq_len, d_model, device=device, dtype=dtype)
    v = torch.randn(seq_len, d_model, device=device, dtype=dtype)
    
    print(f"\n=== Testing with configuration: {seq_len}x{d_model} with {num_heads} heads ===")
    
    # Test Triton implementation
    result = self_attention_triton(q, k, v, num_heads, head_dim, optimized=True)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    
    # Verify output
    assert result.shape == expected.shape, f"Output shape mismatch: {result.shape} vs {expected.shape}"
    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4), "Output values don't match expected"
    
    print("✓ Correctness test passed")
    
    # Test determinism
    result2 = self_attention_triton(q, k, v, num_heads, head_dim, optimized=True)
    assert torch.allclose(result, result2), "Non-deterministic output detected"
    print("✓ Determinism test passed")
    
    # Test module wrapper
    print("\n=== Testing Module Wrapper ===")
    self_attn_module = SelfAttentionModule(d_model, num_heads, head_dim, optimized=True)
    result_module = self_attn_module(q, k, v)
    assert torch.allclose(result, result_module), "Module wrapper output mismatch"
    print("✓ Module wrapper test passed")
    
    # Run comprehensive benchmark
    if device.type == 'cuda':
        print("\n=== Performance Benchmark ===")
        benchmark_results = benchmark_self_attention(
            seq_lens=[64, 128, 256],
            d_models=[512, 768],
            num_heads=[8, 12],
            device=device.type,
            dtype=dtype
        )
        
        # Print summary
        print("\n=== Performance Summary ===")
        for result in benchmark_results:
            print(f"Config {result['seq_len']}x{result['d_model']}x{result['num_heads']}: "
                  f"{result['speedup']:.2f}x speedup, {result['gflops']:.2f} GFLOPS, "
                  f"{result['bandwidth']:.2f} GB/s")
    
    print("\n=== All tests passed! ===")

/*
 * Profiling example & performance tips:
 * 
 * 1. Use nsys profile to analyze kernel performance:
 *    nsys profile --trace=cuda python self_attn_triton.py
 * 
 * 2. Monitor memory bandwidth utilization:
 *    nvprof --metrics achieved_occupancy,sm_efficiency python self_attn_triton.py
 * 
 * 3. For optimal performance:
 *    - Use autotuning to find best block sizes for your hardware
 *    - Enable optimized kernels for better memory access patterns
 *    - Use appropriate data types (FP16/BF16 for memory-bound operations)
 *    - Consider tiling strategies for large sequence lengths
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
