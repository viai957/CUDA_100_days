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
    use_tf32: bool = True 

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

        # Calculate FLOPs
        M, K = a.shape
        K, N = b.shape
        flop_count = 2 * M * N * K
        self.flop_counts.append(flop_count)

        # Track memory usage
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            memory_after = torch.cuda.memory_allocated(self.device)
            self.memory_usage.append(memory_after - memory_before)

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

    def __init__(self, config: MatrixMultiplyConfig = MatrixMuliplyConfig()):
        super().__init__(config)
        self.use_bmm = True # Use batch matrix multiplication when possible
        self.enable_fusion = True 

    def _matrix_multiply_impl(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Optimized matrix multiplication implementation."""
        if self.use_bmm and a.dim() == 2 and b.dim() == 2:
            # Use batch matrix multiplication for better performance
            a_batch = a.unsqueeze(0) # [1, M, K]
            b_batch = b.unsqueeze(0) # [1, K, N]
            result_batch = torch.bmm(a_batch, b_batch) # [ 1, M, N]
            return result_batch.squeeze(0) # [M, N]
        else:
            # Fallback to standard matrix multiplication
            return torch.matmul(a, b)

class TiledMatrixMultiplyModule(MatrixMultiplyModule):
    """
    Tiled matrix multiplication module with advanced techniques.
    """
    def __init__(self, config: MatrixMultiplyConfig = MatricMultiplyConfig(), tile_size: int = 1024):
        """Initialize the tiled matrix multiplication module."""
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
                    partial_res = torch.matmul(a_tile, b_tile)

                    # Accumulate result
                    result[i:i+self.tile_size, j:j+self.tile_size] += partial_res

        return result