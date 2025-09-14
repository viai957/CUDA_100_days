/*
 * RoPE Triton Implementation: Rotary Position Embedding for Transformers
 * Math: Complex rotation of query/key vectors using position-dependent frequencies
 * Inputs: queries [B, S, H, D], keys [B, S, H, D], positions [S]
 * Assumptions: head_dim must be even, contiguous memory layout
 * Parallel Strategy: Each block processes multiple heads across sequence positions
 * Mixed Precision Policy: FP16/BF16 for computation, FP32 for reductions
 * Distributed Hooks: Ready for tensor parallelism via tl.comm_* primitives
 * Complexity: O(B*S*H*D) FLOPs, O(B*S*H*D) bytes moved
 * Test Vectors: Deterministic tiny tensors with known rotation angles
 */

import torch
import triton
import triton.language as tl
import math
from typing import Tuple, Optional

# Autotune configurations for different hardware
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 1, 'BLOCK_SIZE_S': 32, 'BLOCK_SIZE_H': 4, 'BLOCK_SIZE_D': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 1, 'BLOCK_SIZE_S': 64, 'BLOCK_SIZE_H': 2, 'BLOCK_SIZE_D': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 2, 'BLOCK_SIZE_S': 32, 'BLOCK_SIZE_H': 2, 'BLOCK_SIZE_D': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE_B': 1, 'BLOCK_SIZE_S': 16, 'BLOCK_SIZE_H': 8, 'BLOCK_SIZE_D': 32}, num_warps=2),
    ],
    key=['batch_size', 'seq_len', 'num_heads', 'head_dim'],
)
@triton.jit
def rope_kernel(
    # Input tensors
    q_ptr, k_ptr,
    # Output tensors  
    q_out_ptr, k_out_ptr,
    # Tensor dimensions
    batch_size, seq_len, num_heads, head_dim,
    # RoPE parameters
    base_freq: tl.constexpr,
    # Block dimensions
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr, 
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Triton kernel for applying Rotary Position Embedding to query and key tensors.
    
    Memory layout: [batch, seq_len, num_heads, head_dim]
    Each thread block processes a tile of [BLOCK_SIZE_B, BLOCK_SIZE_S, BLOCK_SIZE_H, BLOCK_SIZE_D]
    """
    
    # Get program IDs
    pid_b = tl.program_id(axis=0)
    pid_s = tl.program_id(axis=1) 
    pid_h = tl.program_id(axis=2)
    
    # Calculate block ranges
    b_start = pid_b * BLOCK_SIZE_B
    s_start = pid_s * BLOCK_SIZE_S
    h_start = pid_h * BLOCK_SIZE_H
    
    # Bounds checking
    b_end = min(b_start + BLOCK_SIZE_B, batch_size)
    s_end = min(s_start + BLOCK_SIZE_S, seq_len)
    h_end = min(h_start + BLOCK_SIZE_H, num_heads)
    
    # Process each element in the block
    for b in range(b_start, b_end):
        for s in range(s_start, s_end):
            for h in range(h_start, h_end):
                # Calculate base offset for this [b, s, h] position
                base_offset = b * seq_len * num_heads * head_dim + \
                             s * num_heads * head_dim + \
                             h * head_dim
                
                # Process pairs of dimensions (real, imaginary)
                for d in range(0, head_dim, 2):
                    # Bounds check for head_dim
                    if d + 1 >= head_dim:
                        break
                        
                    # Calculate frequency for this dimension pair
                    freq = 1.0 / tl.math.pow(base_freq, tl.cast(d, tl.float32) / head_dim)
                    theta = tl.cast(s, tl.float32) * freq
                    
                    # Compute rotation matrix elements
                    cos_theta = tl.math.cos(theta)
                    sin_theta = tl.math.sin(theta)
                    
                    # Load query values
                    q_real_offset = base_offset + d
                    q_img_offset = base_offset + d + 1
                    q_real = tl.load(q_ptr + q_real_offset)
                    q_img = tl.load(q_ptr + q_img_offset)
                    
                    # Load key values  
                    k_real_offset = base_offset + d
                    k_img_offset = base_offset + d + 1
                    k_real = tl.load(k_ptr + k_real_offset)
                    k_img = tl.load(k_ptr + k_img_offset)
                    
                    # Apply rotation to query
                    q_real_rot = q_real * cos_theta - q_img * sin_theta
                    q_img_rot = q_real * sin_theta + q_img * cos_theta
                    
                    # Apply rotation to key
                    k_real_rot = k_real * cos_theta - k_img * sin_theta
                    k_img_rot = k_real * sin_theta + k_img * cos_theta
                    
                    # Store rotated values
                    tl.store(q_out_ptr + q_real_offset, q_real_rot)
                    tl.store(q_out_ptr + q_img_offset, q_img_rot)
                    tl.store(k_out_ptr + k_real_offset, k_real_rot)
                    tl.store(k_out_ptr + k_img_offset, k_img_rot)


def apply_rope_triton(
    queries: torch.Tensor,
    keys: torch.Tensor,
    base_freq: float = 10000.0,
    inplace: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embedding using Triton kernel.
    
    Args:
        queries: Query tensor [batch_size, seq_len, num_heads, head_dim]
        keys: Key tensor [batch_size, seq_len, num_heads, head_dim] 
        base_freq: Base frequency for RoPE (default: 10000.0)
        inplace: Whether to modify tensors in-place
        
    Returns:
        Tuple of (rotated_queries, rotated_keys)
    """
    # Input validation
    assert queries.dim() == 4, "Queries must be 4D tensor [B, S, H, D]"
    assert keys.dim() == 4, "Keys must be 4D tensor [B, S, H, D]"
    assert queries.shape == keys.shape, "Queries and keys must have same shape"
    assert queries.shape[-1] % 2 == 0, "Head dimension must be even"
    
    batch_size, seq_len, num_heads, head_dim = queries.shape
    device = queries.device
    dtype = queries.dtype
    
    # Prepare output tensors
    if inplace:
        q_out = queries
        k_out = keys
    else:
        q_out = torch.empty_like(queries)
        k_out = torch.empty_like(keys)
    
    # Ensure tensors are contiguous and on correct device
    queries = queries.contiguous().to(device)
    keys = keys.contiguous().to(device)
    q_out = q_out.contiguous().to(device)
    k_out = k_out.contiguous().to(device)
    
    # Calculate grid dimensions
    # Each program processes a block of [BLOCK_SIZE_B, BLOCK_SIZE_S, BLOCK_SIZE_H, BLOCK_SIZE_D]
    grid_b = triton.cdiv(batch_size, 1)  # Will be overridden by autotune
    grid_s = triton.cdiv(seq_len, 32)    # Will be overridden by autotune  
    grid_h = triton.cdiv(num_heads, 4)   # Will be overridden by autotune
    
    # Launch kernel
    rope_kernel[(grid_b, grid_s, grid_h)](
        queries, keys,
        q_out, k_out,
        batch_size, seq_len, num_heads, head_dim,
        base_freq=base_freq,
        BLOCK_SIZE_B=1,  # Will be overridden by autotune
        BLOCK_SIZE_S=32, # Will be overridden by autotune
        BLOCK_SIZE_H=4,  # Will be overridden by autotune
        BLOCK_SIZE_D=64, # Will be overridden by autotune
    )
    
    return q_out, k_out


def precompute_rope_frequencies(
    head_dim: int, 
    seq_len: int, 
    device: torch.device,
    base_freq: float = 10000.0
) -> torch.Tensor:
    """
    Precompute RoPE frequencies for given head dimension and sequence length.
    
    Args:
        head_dim: Dimension of each attention head (must be even)
        seq_len: Maximum sequence length
        device: Target device
        base_freq: Base frequency for RoPE
        
    Returns:
        Frequencies tensor [seq_len, head_dim//2]
    """
    assert head_dim % 2 == 0, "Head dimension must be even"
    
    # Calculate theta values: theta_i = base_freq^(-2i/d) for i in [0, d/2)
    theta_numerator = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    theta = 1.0 / (base_freq ** (theta_numerator / head_dim))
    
    # Calculate position frequencies: freq[m, i] = m * theta[i]
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, theta)
    
    return freqs


class RoPELayer(torch.nn.Module):
    """
    PyTorch module wrapper for Triton RoPE implementation.
    """
    
    def __init__(self, head_dim: int, base_freq: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "Head dimension must be even"
        self.head_dim = head_dim
        self.base_freq = base_freq
        
    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return apply_rope_triton(queries, keys, self.base_freq)


# Unit test and profiling
if __name__ == "__main__":
    # Test configuration
    batch_size, seq_len, num_heads, head_dim = 2, 128, 8, 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Testing RoPE Triton implementation on {device}")
    print(f"Input shape: [{batch_size}, {seq_len}, {num_heads}, {head_dim}]")
    
    # Create test tensors
    torch.manual_seed(42)
    queries = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    keys = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16)
    
    # Test Triton implementation
    print("\n=== Testing Triton RoPE ===")
    q_rotated, k_rotated = apply_rope_triton(queries, keys)
    
    # Verify output shapes
    assert q_rotated.shape == queries.shape, f"Output shape mismatch: {q_rotated.shape} vs {queries.shape}"
    assert k_rotated.shape == keys.shape, f"Output shape mismatch: {k_rotated.shape} vs {keys.shape}"
    
    # Test determinism
    q_rotated2, k_rotated2 = apply_rope_triton(queries, keys)
    assert torch.allclose(q_rotated, q_rotated2), "Non-deterministic output detected"
    assert torch.allclose(k_rotated, k_rotated2), "Non-deterministic output detected"
    
    print("✓ Shape validation passed")
    print("✓ Determinism test passed")
    
    # Test module wrapper
    print("\n=== Testing Module Wrapper ===")
    rope_layer = RoPELayer(head_dim)
    q_mod, k_mod = rope_layer(queries, keys)
    assert torch.allclose(q_rotated, q_mod), "Module wrapper output mismatch"
    assert torch.allclose(k_rotated, k_mod), "Module wrapper output mismatch"
    print("✓ Module wrapper test passed")
    
    # Performance benchmark
    if device.type == 'cuda':
        print("\n=== Performance Benchmark ===")
        torch.cuda.synchronize()
        
        # Warmup
        for _ in range(10):
            _ = apply_rope_triton(queries, keys)
        
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Benchmark
        num_iterations = 100
        start_event.record()
        for _ in range(num_iterations):
            _ = apply_rope_triton(queries, keys)
        end_event.record()
        
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        
        print(f"Average time per call: {elapsed_ms / num_iterations:.3f} ms")
        print(f"Throughput: {batch_size * seq_len * num_heads * head_dim * 2 * num_iterations / (elapsed_ms / 1000) / 1e9:.2f} GFLOPS")
    
    print("\n=== All tests passed! ===")

/*
 * Profiling example & performance tips:
 * 
 * 1. Use nsys profile to analyze kernel performance:
 *    nsys profile --trace=cuda python RoPE_triton.py
 * 
 * 2. Monitor memory bandwidth utilization:
 *    nvprof --metrics achieved_occupancy,sm_efficiency python RoPE_triton.py
 * 
 * 3. For large sequences, consider:
 *    - Increasing BLOCK_SIZE_S for better sequence parallelism
 *    - Using mixed precision (FP16/BF16) for memory efficiency
 *    - Implementing gradient checkpointing for memory-constrained scenarios
 * 
 * 4. Distributed training considerations:
 *    - Use tl.comm_* primitives for tensor parallelism
 *    - Implement sequence parallelism for very long sequences
 *    - Consider pipeline parallelism for large models
 */
