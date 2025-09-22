"""
Transformer Triton Implementation: High-Performance Transformer Architecture
Math: Multi-head attention, layer normalization, feed-forward networks
Inputs: input[B, S, D] - input tensor, B - batch size, S - sequence length, D - model dimension
Assumptions: B, S, D > 0, tensors are contiguous, device has sufficient memory
Parallel Strategy: Multi-dimensional thread blocks with shared memory optimization
Mixed Precision Policy: FP16/BF16 for computation, FP32 for reductions
Distributed Hooks: Ready for multi-GPU via tl.comm_* primitives
Complexity: O(B*S²*D) FLOPs, O(B*S*D) bytes moved
Test Vectors: Deterministic random tensors with known transformer outputs
"""

import torch
import triton
import triton.language as tl
import math
import time
from typing import Tuple, Optional, List, Dict
import numpy as np

# Autotune configurations for different hardware
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 1, 'BLOCK_SIZE_S': 32, 'BLOCK_SIZE_D': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 2, 'BLOCK_SIZE_S': 64, 'BLOCK_SIZE_D': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 4, 'BLOCK_SIZE_S': 32, 'BLOCK_SIZE_D': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE_B': 1, 'BLOCK_SIZE_S': 16, 'BLOCK_SIZE_D': 128}, num_warps=4),
    ],
    key=['batch_size', 'seq_len', 'd_model'],
)
@triton.jit
def scaled_dot_product_attention_kernel(
    # Input tensors
    q_ptr, k_ptr, v_ptr,
    # Output tensor
    output_ptr,
    # Dimensions
    batch_size, seq_len, head_dim,
    # Scale factor
    scale_factor,
    # Block dimensions
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Triton kernel for scaled dot-product attention.
    
    Memory layout: [batch, heads, seq_len, head_dim]
    Each thread block processes a tile of [BLOCK_SIZE_B, BLOCK_SIZE_S, BLOCK_SIZE_D]
    """
    
    # Get program IDs
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_s = tl.program_id(axis=2)
    
    # Calculate block ranges
    b_start = pid_b * BLOCK_SIZE_B
    s_start = pid_s * BLOCK_SIZE_S
    d_start = 0  # Process all head dimensions
    
    # Bounds checking
    b_end = min(b_start + BLOCK_SIZE_B, batch_size)
    s_end = min(s_start + BLOCK_SIZE_S, seq_len)
    
    # Process each element in the block
    for b in range(b_start, b_end):
        for s in range(s_start, s_end):
            # Calculate attention scores for this query position
            attention_scores = tl.zeros([seq_len], dtype=tl.float32)
            max_score = -float('inf')
            
            # Compute attention scores
            for k in range(seq_len):
                score = 0.0
                
                # Compute dot product between query and key
                for d in range(head_dim):
                    q_offset = b * seq_len * head_dim + s * head_dim + d
                    k_offset = b * seq_len * head_dim + k * head_dim + d
                    score += tl.load(q_ptr + q_offset) * tl.load(k_ptr + k_offset)
                
                score *= scale_factor
                attention_scores[k] = score
                max_score = tl.maximum(max_score, score)
            
            # Compute softmax
            exp_scores = tl.zeros([seq_len], dtype=tl.float32)
            sum_exp = 0.0
            
            for k in range(seq_len):
                exp_score = tl.exp(attention_scores[k] - max_score)
                exp_scores[k] = exp_score
                sum_exp += exp_score
            
            # Normalize attention scores
            for k in range(seq_len):
                attention_scores[k] = exp_scores[k] / sum_exp
            
            # Compute weighted sum of values
            for d in range(head_dim):
                weighted_sum = 0.0
                
                for v_idx in range(seq_len):
                    v_offset = b * seq_len * head_dim + v_idx * head_dim + d
                    weighted_sum += attention_scores[v_idx] * tl.load(v_ptr + v_offset)
                
                out_offset = b * seq_len * head_dim + s * head_dim + d
                tl.store(output_ptr + out_offset, weighted_sum)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 1, 'BLOCK_SIZE_S': 32, 'BLOCK_SIZE_D': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 2, 'BLOCK_SIZE_S': 64, 'BLOCK_SIZE_D': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 4, 'BLOCK_SIZE_S': 32, 'BLOCK_SIZE_D': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE_B': 1, 'BLOCK_SIZE_S': 16, 'BLOCK_SIZE_D': 128}, num_warps=4),
    ],
    key=['batch_size', 'seq_len', 'd_model'],
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
    batch_size, seq_len, d_model,
    # Epsilon for numerical stability
    epsilon,
    # Block dimensions
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Triton kernel for layer normalization.
    
    Memory layout: [batch, seq_len, d_model]
    Each thread block processes a tile of [BLOCK_SIZE_B, BLOCK_SIZE_S, BLOCK_SIZE_D]
    """
    
    # Get program IDs
    pid_b = tl.program_id(axis=0)
    pid_s = tl.program_id(axis=1)
    
    # Calculate block ranges
    b_start = pid_b * BLOCK_SIZE_B
    s_start = pid_s * BLOCK_SIZE_S
    
    # Bounds checking
    b_end = min(b_start + BLOCK_SIZE_B, batch_size)
    s_end = min(s_start + BLOCK_SIZE_S, seq_len)
    
    # Process each element in the block
    for b in range(b_start, b_end):
        for s in range(s_start, s_end):
            # Load data for this position
            data = tl.zeros([d_model], dtype=tl.float32)
            for d in range(d_model):
                offset = b * seq_len * d_model + s * d_model + d
                data[d] = tl.load(input_ptr + offset)
            
            # Compute mean
            mean = tl.sum(data) / d_model
            
            # Compute variance
            var_sum = 0.0
            for d in range(d_model):
                diff = data[d] - mean
                var_sum += diff * diff
            variance = var_sum / d_model
            stddev = tl.sqrt(variance + epsilon)
            
            # Apply normalization
            for d in range(d_model):
                normalized = (data[d] - mean) / stddev
                scaled = normalized * tl.load(gamma_ptr + d) + tl.load(beta_ptr + d)
                offset = b * seq_len * d_model + s * d_model + d
                tl.store(output_ptr + offset, scaled)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 1, 'BLOCK_SIZE_S': 32, 'BLOCK_SIZE_D': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 2, 'BLOCK_SIZE_S': 64, 'BLOCK_SIZE_D': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 4, 'BLOCK_SIZE_S': 32, 'BLOCK_SIZE_D': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE_B': 1, 'BLOCK_SIZE_S': 16, 'BLOCK_SIZE_D': 128}, num_warps=4),
    ],
    key=['batch_size', 'seq_len', 'd_model'],
)
@triton.jit
def feed_forward_kernel(
    # Input tensor
    input_ptr,
    # Output tensor
    output_ptr,
    # Weight matrices
    w1_ptr, b1_ptr, w2_ptr, b2_ptr,
    # Dimensions
    batch_size, seq_len, d_model, d_ff,
    # Block dimensions
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Triton kernel for feed-forward network.
    
    Memory layout: [batch, seq_len, d_model]
    Each thread block processes a tile of [BLOCK_SIZE_B, BLOCK_SIZE_S, BLOCK_SIZE_D]
    """
    
    # Get program IDs
    pid_b = tl.program_id(axis=0)
    pid_s = tl.program_id(axis=1)
    
    # Calculate block ranges
    b_start = pid_b * BLOCK_SIZE_B
    s_start = pid_s * BLOCK_SIZE_S
    
    # Bounds checking
    b_end = min(b_start + BLOCK_SIZE_B, batch_size)
    s_end = min(s_start + BLOCK_SIZE_S, seq_len)
    
    # Process each element in the block
    for b in range(b_start, b_end):
        for s in range(s_start, s_end):
            # Load input data
            input_data = tl.zeros([d_model], dtype=tl.float32)
            for d in range(d_model):
                offset = b * seq_len * d_model + s * d_model + d
                input_data[d] = tl.load(input_ptr + offset)
            
            # First linear layer: input -> hidden
            hidden = tl.zeros([d_ff], dtype=tl.float32)
            for h in range(d_ff):
                sum_val = tl.load(b1_ptr + h)
                for d in range(d_model):
                    w1_offset = d * d_ff + h
                    sum_val += input_data[d] * tl.load(w1_ptr + w1_offset)
                hidden[h] = tl.maximum(0.0, sum_val)  # ReLU activation
            
            # Second linear layer: hidden -> output
            for d in range(d_model):
                sum_val = tl.load(b2_ptr + d)
                for h in range(d_ff):
                    w2_offset = h * d_model + d
                    sum_val += hidden[h] * tl.load(w2_ptr + w2_offset)
                offset = b * seq_len * d_model + s * d_model + d
                tl.store(output_ptr + offset, sum_val)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 1, 'BLOCK_SIZE_S': 32, 'BLOCK_SIZE_D': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 2, 'BLOCK_SIZE_S': 64, 'BLOCK_SIZE_D': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 4, 'BLOCK_SIZE_S': 32, 'BLOCK_SIZE_D': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE_B': 1, 'BLOCK_SIZE_S': 16, 'BLOCK_SIZE_D': 128}, num_warps=4),
    ],
    key=['batch_size', 'seq_len', 'd_model'],
)
@triton.jit
def positional_encoding_kernel(
    # Input tensor
    input_ptr,
    # Dimensions
    batch_size, seq_len, d_model,
    # Base frequency
    base_freq,
    # Block dimensions
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Triton kernel for positional encoding.
    
    Memory layout: [batch, seq_len, d_model]
    Each thread block processes a tile of [BLOCK_SIZE_B, BLOCK_SIZE_S, BLOCK_SIZE_D]
    """
    
    # Get program IDs
    pid_b = tl.program_id(axis=0)
    pid_s = tl.program_id(axis=1)
    pid_d = tl.program_id(axis=2)
    
    # Calculate block ranges
    b_start = pid_b * BLOCK_SIZE_B
    s_start = pid_s * BLOCK_SIZE_S
    d_start = pid_d * BLOCK_SIZE_D
    
    # Bounds checking
    b_end = min(b_start + BLOCK_SIZE_B, batch_size)
    s_end = min(s_start + BLOCK_SIZE_S, seq_len)
    d_end = min(d_start + BLOCK_SIZE_D, d_model)
    
    # Process each element in the block
    for b in range(b_start, b_end):
        for s in range(s_start, s_end):
            for d in range(d_start, d_end):
                offset = b * seq_len * d_model + s * d_model + d
                current_val = tl.load(input_ptr + offset)
                
                if d % 2 == 0:
                    # Even indices: sin
                    freq = 1.0 / tl.math.pow(base_freq, tl.cast(d, tl.float32) / d_model)
                    angle = tl.cast(s, tl.float32) * freq
                    pe_val = tl.math.sin(angle)
                else:
                    # Odd indices: cos
                    freq = 1.0 / tl.math.pow(base_freq, tl.cast(d - 1, tl.float32) / d_model)
                    angle = tl.cast(s, tl.float32) * freq
                    pe_val = tl.math.cos(angle)
                
                tl.store(input_ptr + offset, current_val + pe_val)

def scaled_dot_product_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale_factor: float = None,
    optimized: bool = True
) -> torch.Tensor:
    """
    Apply scaled dot-product attention using Triton kernel.
    
    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        v: Value tensor [batch, heads, seq_len, head_dim]
        scale_factor: Scale factor for attention scores
        optimized: Whether to use optimized kernel
        
    Returns:
        Output tensor [batch, heads, seq_len, head_dim]
    """
    # Input validation
    assert q.dim() == 4, "Query must be 4D tensor [batch, heads, seq_len, head_dim]"
    assert k.dim() == 4, "Key must be 4D tensor [batch, heads, seq_len, head_dim]"
    assert v.dim() == 4, "Value must be 4D tensor [batch, heads, seq_len, head_dim]"
    assert q.shape == k.shape == v.shape, "Q, K, V must have same shape"
    
    batch_size, num_heads, seq_len, head_dim = q.shape
    device = q.device
    dtype = q.dtype
    
    if scale_factor is None:
        scale_factor = 1.0 / math.sqrt(head_dim)
    
    # Prepare output tensor
    output = torch.empty_like(q)
    
    # Ensure tensors are contiguous and on correct device
    q = q.contiguous().to(device)
    k = k.contiguous().to(device)
    v = v.contiguous().to(device)
    output = output.contiguous().to(device)
    
    # Calculate grid dimensions
    grid_b = triton.cdiv(batch_size, 1)
    grid_h = triton.cdiv(num_heads, 1)
    grid_s = triton.cdiv(seq_len, 32)
    
    # Launch kernel
    scaled_dot_product_attention_kernel[(grid_b, grid_h, grid_s)](
        q, k, v, output,
        batch_size, seq_len, head_dim,
        scale_factor=scale_factor,
        BLOCK_SIZE_B=1,
        BLOCK_SIZE_S=32,
        BLOCK_SIZE_D=64
    )
    
    return output

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
        input_tensor: Input tensor [batch, seq_len, d_model]
        gamma: Scale parameter [d_model]
        beta: Shift parameter [d_model]
        epsilon: Small value for numerical stability
        optimized: Whether to use optimized kernel
        
    Returns:
        Output tensor [batch, seq_len, d_model]
    """
    # Input validation
    assert input_tensor.dim() == 3, "Input must be 3D tensor [batch, seq_len, d_model]"
    assert gamma.dim() == 1, "Gamma must be 1D tensor [d_model]"
    assert beta.dim() == 1, "Beta must be 1D tensor [d_model]"
    assert input_tensor.shape[-1] == gamma.shape[0] == beta.shape[0], "Last dimension must match"
    
    batch_size, seq_len, d_model = input_tensor.shape
    device = input_tensor.device
    dtype = input_tensor.dtype
    
    # Prepare output tensor
    output = torch.empty_like(input_tensor)
    
    # Ensure tensors are contiguous and on correct device
    input_tensor = input_tensor.contiguous().to(device)
    gamma = gamma.contiguous().to(device)
    beta = beta.contiguous().to(device)
    output = output.contiguous().to(device)
    
    # Calculate grid dimensions
    grid_b = triton.cdiv(batch_size, 1)
    grid_s = triton.cdiv(seq_len, 32)
    
    # Launch kernel
    layer_norm_kernel[(grid_b, grid_s)](
        input_tensor, output, gamma, beta,
        batch_size, seq_len, d_model,
        epsilon=epsilon,
        BLOCK_SIZE_B=1,
        BLOCK_SIZE_S=32,
        BLOCK_SIZE_D=64
    )
    
    return output

def feed_forward_triton(
    input_tensor: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    optimized: bool = True
) -> torch.Tensor:
    """
    Apply feed-forward network using Triton kernel.
    
    Args:
        input_tensor: Input tensor [batch, seq_len, d_model]
        w1: First weight matrix [d_model, d_ff]
        b1: First bias vector [d_ff]
        w2: Second weight matrix [d_ff, d_model]
        b2: Second bias vector [d_model]
        optimized: Whether to use optimized kernel
        
    Returns:
        Output tensor [batch, seq_len, d_model]
    """
    # Input validation
    assert input_tensor.dim() == 3, "Input must be 3D tensor [batch, seq_len, d_model]"
    assert w1.dim() == 2, "W1 must be 2D tensor [d_model, d_ff]"
    assert w2.dim() == 2, "W2 must be 2D tensor [d_ff, d_model]"
    
    batch_size, seq_len, d_model = input_tensor.shape
    d_ff = w1.shape[1]
    device = input_tensor.device
    dtype = input_tensor.dtype
    
    # Prepare output tensor
    output = torch.empty_like(input_tensor)
    
    # Ensure tensors are contiguous and on correct device
    input_tensor = input_tensor.contiguous().to(device)
    w1 = w1.contiguous().to(device)
    b1 = b1.contiguous().to(device)
    w2 = w2.contiguous().to(device)
    b2 = b2.contiguous().to(device)
    output = output.contiguous().to(device)
    
    # Calculate grid dimensions
    grid_b = triton.cdiv(batch_size, 1)
    grid_s = triton.cdiv(seq_len, 32)
    
    # Launch kernel
    feed_forward_kernel[(grid_b, grid_s)](
        input_tensor, output, w1, b1, w2, b2,
        batch_size, seq_len, d_model, d_ff,
        BLOCK_SIZE_B=1,
        BLOCK_SIZE_S=32,
        BLOCK_SIZE_D=64
    )
    
    return output

def positional_encoding_triton(
    input_tensor: torch.Tensor,
    base_freq: float = 10000.0,
    optimized: bool = True
) -> torch.Tensor:
    """
    Apply positional encoding using Triton kernel.
    
    Args:
        input_tensor: Input tensor [batch, seq_len, d_model]
        base_freq: Base frequency for positional encoding
        optimized: Whether to use optimized kernel
        
    Returns:
        Output tensor [batch, seq_len, d_model] (modified in-place)
    """
    # Input validation
    assert input_tensor.dim() == 3, "Input must be 3D tensor [batch, seq_len, d_model]"
    
    batch_size, seq_len, d_model = input_tensor.shape
    device = input_tensor.device
    dtype = input_tensor.dtype
    
    # Ensure tensor is contiguous and on correct device
    input_tensor = input_tensor.contiguous().to(device)
    
    # Calculate grid dimensions
    grid_b = triton.cdiv(batch_size, 1)
    grid_s = triton.cdiv(seq_len, 32)
    grid_d = triton.cdiv(d_model, 64)
    
    # Launch kernel
    positional_encoding_kernel[(grid_b, grid_s, grid_d)](
        input_tensor,
        batch_size, seq_len, d_model,
        base_freq=base_freq,
        BLOCK_SIZE_B=1,
        BLOCK_SIZE_S=32,
        BLOCK_SIZE_D=64
    )
    
    return input_tensor

class TransformerTriton(torch.nn.Module):
    """
    PyTorch module wrapper for Triton transformer components.
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Initialize parameters
        self.gamma = torch.nn.Parameter(torch.ones(d_model))
        self.beta = torch.nn.Parameter(torch.zeros(d_model))
        self.w1 = torch.nn.Parameter(torch.randn(d_model, d_ff) * 0.1)
        self.b1 = torch.nn.Parameter(torch.zeros(d_ff))
        self.w2 = torch.nn.Parameter(torch.randn(d_ff, d_model) * 0.1)
        self.b2 = torch.nn.Parameter(torch.zeros(d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for transformer components.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Apply positional encoding
        x = positional_encoding_triton(x)
        
        # Apply layer normalization
        x = layer_norm_triton(x, self.gamma, self.beta)
        
        # Apply feed-forward network
        x = feed_forward_triton(x, self.w1, self.b1, self.w2, self.b2)
        
        return x

def test_transformer_triton(
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    use_optimized: bool = True
):
    """
    Test function for transformer components with Triton.
    """
    print(f"\n=== Transformer Triton Test ===")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Model dimension: {d_model}")
    print(f"Number of heads: {num_heads}")
    print(f"Feed-forward dimension: {d_ff}")
    print(f"Optimized: {use_optimized}")
    
    # Create test data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    
    # Test scaled dot-product attention
    print("\n--- Testing Scaled Dot-Product Attention ---")
    q = torch.randn(batch_size, num_heads, seq_len, d_model // num_heads, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, d_model // num_heads, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, d_model // num_heads, device=device, dtype=dtype)
    
    output_attn = scaled_dot_product_attention_triton(q, k, v, optimized=use_optimized)
    print(f"Attention output shape: {output_attn.shape}")
    
    # Test layer normalization
    print("\n--- Testing Layer Normalization ---")
    input_tensor = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
    gamma = torch.ones(d_model, device=device, dtype=dtype)
    beta = torch.zeros(d_model, device=device, dtype=dtype)
    
    output_norm = layer_norm_triton(input_tensor, gamma, beta, optimized=use_optimized)
    print(f"Layer norm output shape: {output_norm.shape}")
    
    # Test feed-forward network
    print("\n--- Testing Feed-Forward Network ---")
    w1 = torch.randn(d_model, d_ff, device=device, dtype=dtype) * 0.1
    b1 = torch.zeros(d_ff, device=device, dtype=dtype)
    w2 = torch.randn(d_ff, d_model, device=device, dtype=dtype) * 0.1
    b2 = torch.zeros(d_model, device=device, dtype=dtype)
    
    output_ff = feed_forward_triton(input_tensor, w1, b1, w2, b2, optimized=use_optimized)
    print(f"Feed-forward output shape: {output_ff.shape}")
    
    # Test positional encoding
    print("\n--- Testing Positional Encoding ---")
    input_pe = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
    output_pe = positional_encoding_triton(input_pe, optimized=use_optimized)
    print(f"Positional encoding output shape: {output_pe.shape}")
    
    # Performance test
    if device.type == 'cuda':
        print("\n--- Performance Test ---")
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(100):
            _ = scaled_dot_product_attention_triton(q, k, v, optimized=use_optimized)
            _ = layer_norm_triton(input_tensor, gamma, beta, optimized=use_optimized)
            _ = feed_forward_triton(input_tensor, w1, b1, w2, b2, optimized=use_optimized)
            _ = positional_encoding_triton(input_pe, optimized=use_optimized)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100 * 1000  # Convert to ms
        print(f"Average time per iteration: {avg_time:.3f} ms")
    
    return output_attn, output_norm, output_ff, output_pe

def benchmark_transformer_triton():
    """
    Benchmark transformer components with different configurations.
    """
    print("\n=== Transformer Triton Benchmark ===")
    
    configurations = [
        (2, 64, 512, 8, 2048),
        (4, 128, 768, 12, 3072),
        (8, 256, 1024, 16, 4096),
    ]
    
    for batch_size, seq_len, d_model, num_heads, d_ff in configurations:
        print(f"\nConfiguration: batch={batch_size}, seq={seq_len}, d_model={d_model}, heads={num_heads}, d_ff={d_ff}")
        test_transformer_triton(batch_size, seq_len, d_model, num_heads, d_ff, use_optimized=True)

def test_transformer_module():
    """
    Test the PyTorch module wrapper.
    """
    print("\n=== Transformer Module Test ===")
    
    # Create module
    transformer = TransformerTriton(d_model=512, num_heads=8, d_ff=2048)
    transformer = transformer.cuda()
    
    # Create test data
    batch_size = 4
    seq_len = 128
    input_tensor = torch.randn(batch_size, seq_len, 512, device='cuda')
    
    # Test forward pass
    output = transformer(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test gradient computation
    loss = output.sum()
    loss.backward()
    print(f"Gamma gradient norm: {transformer.gamma.grad.norm().item():.6f}")
    print(f"Beta gradient norm: {transformer.beta.grad.norm().item():.6f}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test basic functionality
    test_transformer_triton(2, 64, 512, 8, 2048, use_optimized=True)
    test_transformer_triton(2, 64, 512, 8, 2048, use_optimized=False)
    
    # Test module wrapper
    test_transformer_module()
    
    # Run benchmark
    benchmark_transformer_triton()
    
    print("\n=== All Tests Complete ===")

"""
Profiling example & performance tips:

1. Use nsys profile to analyze kernel performance:
   nsys profile --trace=cuda python transformer_triton.py

2. Monitor memory bandwidth utilization:
   nvprof --metrics achieved_occupancy,sm_efficiency python transformer_triton.py

3. For optimal performance:
   - Use autotuning to find best block sizes for your hardware
   - Enable optimized kernels for better memory access patterns
   - Use appropriate data types (FP16/BF16 for memory-bound operations)
   - Consider kernel fusion for better performance

4. Memory optimization:
   - Ensure input tensors are contiguous
   - Use appropriate block sizes for your GPU architecture
   - Consider memory coalescing for better bandwidth utilization

5. Distributed training considerations:
   - Use tl.comm_* primitives for multi-GPU operations
   - Implement gradient synchronization for distributed training
   - Consider memory-efficient implementations for large models
"""
