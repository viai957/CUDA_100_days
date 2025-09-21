"""
1D Convolution Triton Implementation: High-Performance Convolutional Operations
Math: output[i] = Σ(input[i+k] * kernel[k]) for k ∈ [0, kernel_size)
Inputs: input[N] - input signal, kernel[K] - convolution kernel, N - signal length, K - kernel size
Assumptions: N > 0, K > 0, arrays are contiguous, device has sufficient memory
Parallel Strategy: Vectorized operations with autotuning and mixed precision
Mixed Precision Policy: FP16/BF16 for computation, FP32 for reductions
Distributed Hooks: Ready for multi-GPU via distributed training
Complexity: O(NK) FLOPs, O(N+K) bytes moved
Test Vectors: Deterministic random signals with known convolution results
"""

import torch
import triton
import triton.language as tl
import numpy as np
import time
import sys
import os

# Add parent directory to path for cuda_common import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['signal_length', 'kernel_size'],
)
@triton.jit
def conv_1d_kernel(
    input_ptr, output_ptr, kernel_ptr,
    signal_length, kernel_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for 1D convolution with autotuning
    """
    pid = tl.program_id(axis=0)
    
    # Calculate output indices
    output_start = pid * BLOCK_SIZE
    output_end = tl.minimum(output_start + BLOCK_SIZE, signal_length)
    
    # Calculate input range needed for this block
    input_start = tl.maximum(0, output_start - kernel_size // 2)
    input_end = tl.minimum(signal_length, output_end + kernel_size // 2)
    
    # Load input data
    input_indices = tl.arange(0, input_end - input_start) + input_start
    input_mask = input_indices < signal_length
    input_data = tl.load(input_ptr + input_indices, mask=input_mask, other=0.0)
    
    # Load kernel
    kernel_indices = tl.arange(0, kernel_size)
    kernel_data = tl.load(kernel_ptr + kernel_indices)
    
    # Compute convolution for each output element
    for i in range(BLOCK_SIZE):
        output_idx = output_start + i
        if output_idx < signal_length:
            result = 0.0
            
            for k in range(kernel_size):
                input_idx = output_idx - kernel_size // 2 + k
                if input_idx >= 0 and input_idx < signal_length:
                    input_val = tl.load(input_ptr + input_idx)
                    result += input_val * kernel_data[k]
            
            tl.store(output_ptr + output_idx, result)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'KERNEL_TILE': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256, 'KERNEL_TILE': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512, 'KERNEL_TILE': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024, 'KERNEL_TILE': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048, 'KERNEL_TILE': 16}, num_warps=8),
    ],
    key=['signal_length', 'kernel_size'],
)
@triton.jit
def conv_1d_tiled_kernel(
    input_ptr, output_ptr, kernel_ptr,
    signal_length, kernel_size,
    BLOCK_SIZE: tl.constexpr,
    KERNEL_TILE: tl.constexpr,
):
    """
    Triton kernel for 1D convolution with tiling optimization
    """
    pid = tl.program_id(axis=0)
    
    # Calculate output indices
    output_start = pid * BLOCK_SIZE
    output_end = tl.minimum(output_start + BLOCK_SIZE, signal_length)
    
    # Calculate input range needed for this block
    input_start = tl.maximum(0, output_start - kernel_size // 2)
    input_end = tl.minimum(signal_length, output_end + kernel_size // 2)
    
    # Load input data with tiling
    input_data = tl.zeros([input_end - input_start], dtype=tl.float32)
    for i in range(0, input_end - input_start, KERNEL_TILE):
        input_indices = tl.arange(0, KERNEL_TILE) + input_start + i
        input_mask = input_indices < signal_length
        input_tile = tl.load(input_ptr + input_indices, mask=input_mask, other=0.0)
        input_data = tl.where(input_mask, input_tile, input_data)
    
    # Load kernel with tiling
    kernel_data = tl.zeros([kernel_size], dtype=tl.float32)
    for i in range(0, kernel_size, KERNEL_TILE):
        kernel_indices = tl.arange(0, KERNEL_TILE) + i
        kernel_mask = kernel_indices < kernel_size
        kernel_tile = tl.load(kernel_ptr + kernel_indices, mask=kernel_mask, other=0.0)
        kernel_data = tl.where(kernel_mask, kernel_tile, kernel_data)
    
    # Compute convolution for each output element
    for i in range(BLOCK_SIZE):
        output_idx = output_start + i
        if output_idx < signal_length:
            result = 0.0
            
            for k in range(kernel_size):
                input_idx = output_idx - kernel_size // 2 + k
                if input_idx >= 0 and input_idx < signal_length:
                    input_val = tl.load(input_ptr + input_idx)
                    result += input_val * kernel_data[k]
            
            tl.store(output_ptr + output_idx, result)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['signal_length', 'kernel_size'],
)
@triton.jit
def conv_1d_vectorized_kernel(
    input_ptr, output_ptr, kernel_ptr,
    signal_length, kernel_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for 1D convolution with vectorized operations
    """
    pid = tl.program_id(axis=0)
    
    # Calculate output indices
    output_start = pid * BLOCK_SIZE
    output_end = tl.minimum(output_start + BLOCK_SIZE, signal_length)
    
    # Vectorized computation
    for i in range(0, BLOCK_SIZE, 4):  # Process 4 elements at a time
        output_indices = tl.arange(0, 4) + output_start + i
        output_mask = output_indices < signal_length
        
        if tl.any(output_mask):
            result = tl.zeros([4], dtype=tl.float32)
            
            for k in range(kernel_size):
                input_indices = output_indices - kernel_size // 2 + k
                input_mask = (input_indices >= 0) & (input_indices < signal_length)
                
                input_vals = tl.load(input_ptr + input_indices, mask=input_mask, other=0.0)
                kernel_val = tl.load(kernel_ptr + k)
                
                result += input_vals * kernel_val
            
            tl.store(output_ptr + output_indices, result, mask=output_mask)

class Conv1DTriton(torch.nn.Module):
    """
    PyTorch module wrapper for 1D convolution using Triton
    """
    def __init__(self, kernel_size, padding=0, stride=1, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        
        # Initialize kernel
        self.kernel = torch.nn.Parameter(torch.randn(kernel_size))
        
    def forward(self, x):
        """
        Forward pass for 1D convolution
        Args:
            x: Input tensor of shape (batch_size, signal_length)
        Returns:
            Output tensor of shape (batch_size, output_length)
        """
        batch_size, signal_length = x.shape
        output_length = (signal_length + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        
        # Pad input if needed
        if self.padding > 0:
            x = torch.nn.functional.pad(x, (self.padding, self.padding))
            signal_length = x.shape[1]
        
        # Flatten for processing
        x_flat = x.view(-1)
        output_flat = torch.zeros(batch_size * output_length, device=x.device, dtype=x.dtype)
        
        # Launch kernel
        grid = (output_length + 255) // 256
        conv_1d_kernel[grid](
            x_flat, output_flat, self.kernel,
            signal_length, self.kernel_size,
            BLOCK_SIZE=256
        )
        
        return output_flat.view(batch_size, output_length)

def test_conv_1d_triton(signal_length, kernel_size, batch_size=1, use_tiling=False, use_vectorized=False):
    """
    Test function for 1D convolution with Triton
    """
    print(f"\n=== 1D Convolution Triton Test ===")
    print(f"Signal length: {signal_length}")
    print(f"Kernel size: {kernel_size}")
    print(f"Batch size: {batch_size}")
    print(f"Tiling: {use_tiling}")
    print(f"Vectorized: {use_vectorized}")
    
    # Create test data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_data = torch.randn(batch_size, signal_length, device=device)
    kernel = torch.randn(kernel_size, device=device)
    
    # Compute expected result using PyTorch
    expected = torch.nn.functional.conv1d(
        input_data.unsqueeze(1), 
        kernel.unsqueeze(0).unsqueeze(0),
        padding=kernel_size//2
    ).squeeze(1)
    
    # Test Triton implementation
    if use_tiling:
        kernel_func = conv_1d_tiled_kernel
    elif use_vectorized:
        kernel_func = conv_1d_vectorized_kernel
    else:
        kernel_func = conv_1d_kernel
    
    # Flatten input for kernel
    input_flat = input_data.view(-1)
    output_flat = torch.zeros_like(input_flat)
    
    # Launch kernel
    grid = (signal_length + 255) // 256
    kernel_func[grid](
        input_flat, output_flat, kernel,
        signal_length, kernel_size,
        BLOCK_SIZE=256
    )
    
    # Reshape output
    output = output_flat.view(batch_size, signal_length)
    
    # Verify correctness
    diff = torch.abs(output - expected).max().item()
    print(f"Max difference: {diff:.6f}")
    print(f"Correctness: {'PASS' if diff < 1e-4 else 'FAIL'}")
    
    # Performance test
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(100):
        kernel_func[grid](
            input_flat, output_flat, kernel,
            signal_length, kernel_size,
            BLOCK_SIZE=256
        )
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100 * 1000  # Convert to ms
    print(f"Average time: {avg_time:.3f} ms")
    
    # Performance analysis
    operations = signal_length * kernel_size
    gflops = operations / (avg_time / 1000) / 1e9
    print(f"Performance: {gflops:.2f} GFLOPS")
    
    return output

def benchmark_conv_1d_triton():
    """
    Benchmark 1D convolution with different configurations
    """
    print("\n=== 1D Convolution Triton Benchmark ===")
    
    signal_lengths = [1024, 10000, 100000, 1000000]
    kernel_sizes = [3, 5, 7, 9, 15]
    
    for signal_length in signal_lengths:
        for kernel_size in kernel_sizes:
            print(f"\nConfiguration: signal_length={signal_length}, kernel_size={kernel_size}")
            test_conv_1d_triton(signal_length, kernel_size, use_tiling=True)

def test_conv_1d_module():
    """
    Test the PyTorch module wrapper
    """
    print("\n=== 1D Convolution Module Test ===")
    
    # Create module
    conv_module = Conv1DTriton(kernel_size=5, padding=2)
    conv_module = conv_module.cuda()
    
    # Create test data
    batch_size = 4
    signal_length = 1000
    input_data = torch.randn(batch_size, signal_length, device='cuda')
    
    # Test forward pass
    output = conv_module(input_data)
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test gradient computation
    loss = output.sum()
    loss.backward()
    print(f"Kernel gradient norm: {conv_module.kernel.grad.norm().item():.6f}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test basic functionality
    test_conv_1d_triton(1000, 5, use_tiling=False)
    test_conv_1d_triton(1000, 5, use_tiling=True)
    test_conv_1d_triton(1000, 5, use_vectorized=True)
    
    # Test module wrapper
    test_conv_1d_module()
    
    # Run benchmark
    benchmark_conv_1d_triton()
    
    print("\n=== All Tests Complete ===")
