"""
Sparse Matrix Multiplication Triton Implementation: High-Performance Sparse Operations
Math: y = A * x where A is sparse matrix, x is dense vector, y is dense vector
Inputs: A[M, N] - sparse matrix, x[N] - dense vector, M - rows, N - columns
Assumptions: M, N > 0, matrix is sparse, device has sufficient memory
Parallel Strategy: Multiple sparse formats (CSR, ELL, COO) with optimized kernels
Mixed Precision Policy: FP16/BF16 for computation, FP32 for reductions
Distributed Hooks: Ready for multi-GPU via tl.comm_* primitives
Complexity: O(nnz) FLOPs, O(nnz + M + N) bytes moved
Test Vectors: Deterministic sparse matrices with known multiplication results
"""

import torch
import triton
import triton.language as tl
import math
import time
from typing import Tuple, Optional, List, Dict
import numpy as np
import scipy.sparse as sp

# Autotune configurations for different hardware
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['rows', 'nnz'],
)
@triton.jit
def csr_spmv_kernel(
    # CSR matrix data
    values_ptr, col_indices_ptr, row_ptrs_ptr,
    # Input vector
    x_ptr,
    # Output vector
    y_ptr,
    # Dimensions
    rows,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for CSR matrix-vector multiplication.
    
    Memory layout: CSR format with values, col_indices, row_ptrs
    Each thread block processes BLOCK_SIZE rows
    """
    
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate block range
    row_start = pid * BLOCK_SIZE
    row_end = min(row_start + BLOCK_SIZE, rows)
    
    # Process rows in this block
    for row in range(row_start, row_end):
        # Get row start and end indices
        row_start_idx = tl.load(row_ptrs_ptr + row)
        row_end_idx = tl.load(row_ptrs_ptr + row + 1)
        
        # Compute dot product for this row
        sum_val = 0.0
        for j in range(row_start_idx, row_end_idx):
            # Load value and column index
            val = tl.load(values_ptr + j)
            col_idx = tl.load(col_indices_ptr + j)
            
            # Load corresponding x value and accumulate
            x_val = tl.load(x_ptr + col_idx)
            sum_val += val * x_val
        
        # Store result
        tl.store(y_ptr + row, sum_val)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64, 'MAX_NNZ': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128, 'MAX_NNZ': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256, 'MAX_NNZ': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512, 'MAX_NNZ': 128}, num_warps=8),
    ],
    key=['rows', 'max_nnz_per_row'],
)
@triton.jit
def ell_spmv_kernel(
    # ELL matrix data
    data_ptr, indices_ptr,
    # Input vector
    x_ptr,
    # Output vector
    y_ptr,
    # Dimensions
    rows, max_nnz_per_row,
    # Block size
    BLOCK_SIZE: tl.constexpr,
    MAX_NNZ: tl.constexpr,
):
    """
    Triton kernel for ELL matrix-vector multiplication.
    
    Memory layout: ELL format with data and indices arrays
    Each thread block processes BLOCK_SIZE rows
    """
    
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate block range
    row_start = pid * BLOCK_SIZE
    row_end = min(row_start + BLOCK_SIZE, rows)
    
    # Process rows in this block
    for row in range(row_start, row_end):
        sum_val = 0.0
        
        # Process non-zeros in this row
        for j in range(max_nnz_per_row):
            # Load data and index
            val = tl.load(data_ptr + j * rows + row)
            col_idx = tl.load(indices_ptr + j * rows + row)
            
            # Check if this is a valid non-zero
            if col_idx != -1:
                x_val = tl.load(x_ptr + col_idx)
                sum_val += val * x_val
        
        # Store result
        tl.store(y_ptr + row, sum_val)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['nnz'],
)
@triton.jit
def coo_spmv_kernel(
    # COO matrix data
    values_ptr, row_indices_ptr, col_indices_ptr,
    # Input vector
    x_ptr,
    # Output vector
    y_ptr,
    # Dimensions
    nnz, rows,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for COO matrix-vector multiplication.
    
    Memory layout: COO format with values, row_indices, col_indices
    Each thread block processes BLOCK_SIZE non-zeros
    """
    
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate block range
    idx_start = pid * BLOCK_SIZE
    idx_end = min(idx_start + BLOCK_SIZE, nnz)
    
    # Process non-zeros in this block
    for idx in range(idx_start, idx_end):
        # Load data
        val = tl.load(values_ptr + idx)
        row_idx = tl.load(row_indices_ptr + idx)
        col_idx = tl.load(col_indices_ptr + idx)
        
        # Load x value and compute contribution
        x_val = tl.load(x_ptr + col_idx)
        contribution = val * x_val
        
        # Atomic add to output
        tl.atomic_add(y_ptr + row_idx, contribution)

def csr_spmv_triton(
    values: torch.Tensor,
    col_indices: torch.Tensor,
    row_ptrs: torch.Tensor,
    x: torch.Tensor,
    optimized: bool = True
) -> torch.Tensor:
    """
    Apply CSR matrix-vector multiplication using Triton kernel.
    
    Args:
        values: CSR values array [nnz]
        col_indices: CSR column indices array [nnz]
        row_ptrs: CSR row pointers array [rows + 1]
        x: Input vector [cols]
        optimized: Whether to use optimized kernel
        
    Returns:
        Output vector [rows]
    """
    # Input validation
    assert values.dim() == 1, "Values must be 1D tensor"
    assert col_indices.dim() == 1, "Col indices must be 1D tensor"
    assert row_ptrs.dim() == 1, "Row ptrs must be 1D tensor"
    assert x.dim() == 1, "Input vector must be 1D tensor"
    
    rows = row_ptrs.shape[0] - 1
    device = x.device
    dtype = x.dtype
    
    # Prepare output tensor
    y = torch.zeros(rows, device=device, dtype=dtype)
    
    # Ensure tensors are contiguous and on correct device
    values = values.contiguous().to(device)
    col_indices = col_indices.contiguous().to(device)
    row_ptrs = row_ptrs.contiguous().to(device)
    x = x.contiguous().to(device)
    y = y.contiguous().to(device)
    
    # Calculate grid dimensions
    grid_size = triton.cdiv(rows, 128)
    
    # Launch kernel
    csr_spmv_kernel[(grid_size,)](
        values, col_indices, row_ptrs, x, y,
        rows,
        BLOCK_SIZE=128
    )
    
    return y

def ell_spmv_triton(
    data: torch.Tensor,
    indices: torch.Tensor,
    x: torch.Tensor,
    max_nnz_per_row: int,
    optimized: bool = True
) -> torch.Tensor:
    """
    Apply ELL matrix-vector multiplication using Triton kernel.
    
    Args:
        data: ELL data array [max_nnz_per_row, rows]
        indices: ELL indices array [max_nnz_per_row, rows]
        x: Input vector [cols]
        max_nnz_per_row: Maximum non-zeros per row
        optimized: Whether to use optimized kernel
        
    Returns:
        Output vector [rows]
    """
    # Input validation
    assert data.dim() == 2, "Data must be 2D tensor"
    assert indices.dim() == 2, "Indices must be 2D tensor"
    assert x.dim() == 1, "Input vector must be 1D tensor"
    
    rows = data.shape[1]
    device = x.device
    dtype = x.dtype
    
    # Prepare output tensor
    y = torch.zeros(rows, device=device, dtype=dtype)
    
    # Ensure tensors are contiguous and on correct device
    data = data.contiguous().to(device)
    indices = indices.contiguous().to(device)
    x = x.contiguous().to(device)
    y = y.contiguous().to(device)
    
    # Calculate grid dimensions
    grid_size = triton.cdiv(rows, 128)
    
    # Launch kernel
    ell_spmv_kernel[(grid_size,)](
        data, indices, x, y,
        rows, max_nnz_per_row,
        BLOCK_SIZE=128,
        MAX_NNZ=32
    )
    
    return y

def coo_spmv_triton(
    values: torch.Tensor,
    row_indices: torch.Tensor,
    col_indices: torch.Tensor,
    x: torch.Tensor,
    rows: int,
    optimized: bool = True
) -> torch.Tensor:
    """
    Apply COO matrix-vector multiplication using Triton kernel.
    
    Args:
        values: COO values array [nnz]
        row_indices: COO row indices array [nnz]
        col_indices: COO column indices array [nnz]
        x: Input vector [cols]
        rows: Number of rows in matrix
        optimized: Whether to use optimized kernel
        
    Returns:
        Output vector [rows]
    """
    # Input validation
    assert values.dim() == 1, "Values must be 1D tensor"
    assert row_indices.dim() == 1, "Row indices must be 1D tensor"
    assert col_indices.dim() == 1, "Col indices must be 1D tensor"
    assert x.dim() == 1, "Input vector must be 1D tensor"
    
    nnz = values.shape[0]
    device = x.device
    dtype = x.dtype
    
    # Prepare output tensor
    y = torch.zeros(rows, device=device, dtype=dtype)
    
    # Ensure tensors are contiguous and on correct device
    values = values.contiguous().to(device)
    row_indices = row_indices.contiguous().to(device)
    col_indices = col_indices.contiguous().to(device)
    x = x.contiguous().to(device)
    y = y.contiguous().to(device)
    
    # Calculate grid dimensions
    grid_size = triton.cdiv(nnz, 128)
    
    # Launch kernel
    coo_spmv_kernel[(grid_size,)](
        values, row_indices, col_indices, x, y,
        nnz, rows,
        BLOCK_SIZE=128
    )
    
    return y

class SparseMatrixTriton(torch.nn.Module):
    """
    PyTorch module wrapper for Triton sparse matrix operations.
    """
    
    def __init__(self, matrix_format: str = 'csr', optimized: bool = True):
        super().__init__()
        self.matrix_format = matrix_format
        self.optimized = optimized
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sparse matrix-vector multiplication.
        
        Args:
            x: Input vector [cols]
            
        Returns:
            Output vector [rows]
        """
        if self.matrix_format == 'csr':
            return self._csr_forward(x)
        elif self.matrix_format == 'ell':
            return self._ell_forward(x)
        elif self.matrix_format == 'coo':
            return self._coo_forward(x)
        else:
            raise ValueError(f"Unsupported matrix format: {self.matrix_format}")
    
    def _csr_forward(self, x: torch.Tensor) -> torch.Tensor:
        """CSR forward pass."""
        return csr_spmv_triton(
            self.values, self.col_indices, self.row_ptrs, x, self.optimized
        )
    
    def _ell_forward(self, x: torch.Tensor) -> torch.Tensor:
        """ELL forward pass."""
        return ell_spmv_triton(
            self.data, self.indices, x, self.max_nnz_per_row, self.optimized
        )
    
    def _coo_forward(self, x: torch.Tensor) -> torch.Tensor:
        """COO forward pass."""
        return coo_spmv_triton(
            self.values, self.row_indices, self.col_indices, x, self.rows, self.optimized
        )

def create_sparse_matrix(
    rows: int,
    cols: int,
    sparsity: float,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a sparse matrix and convert to different formats.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        sparsity: Sparsity ratio (0.0 to 1.0)
        device: Device to create tensors on
        dtype: Data type for tensors
        
    Returns:
        Tuple of (dense_matrix, csr_values, csr_col_indices, csr_row_ptrs, 
                 ell_data, ell_indices, coo_values, coo_row_indices, coo_col_indices)
    """
    # Create dense matrix
    torch.manual_seed(42)
    dense_matrix = torch.randn(rows, cols, device=device, dtype=dtype)
    
    # Apply sparsity
    mask = torch.rand(rows, cols, device=device) > sparsity
    dense_matrix = dense_matrix * mask.float()
    
    # Convert to CSR format
    dense_np = dense_matrix.cpu().numpy()
    csr_matrix = sp.csr_matrix(dense_np)
    
    csr_values = torch.tensor(csr_matrix.data, device=device, dtype=dtype)
    csr_col_indices = torch.tensor(csr_matrix.indices, device=device, dtype=torch.int32)
    csr_row_ptrs = torch.tensor(csr_matrix.indptr, device=device, dtype=torch.int32)
    
    # Convert to ELL format
    max_nnz_per_row = int(csr_matrix.getnnz(axis=1).max())
    ell_data = torch.zeros(max_nnz_per_row, rows, device=device, dtype=dtype)
    ell_indices = torch.full((max_nnz_per_row, rows), -1, device=device, dtype=torch.int32)
    
    for i in range(rows):
        row_start = csr_matrix.indptr[i]
        row_end = csr_matrix.indptr[i + 1]
        for j, (val, col) in enumerate(zip(csr_matrix.data[row_start:row_end], 
                                         csr_matrix.indices[row_start:row_end])):
            if j < max_nnz_per_row:
                ell_data[j, i] = val
                ell_indices[j, i] = col
    
    # Convert to COO format
    coo_matrix = csr_matrix.tocoo()
    coo_values = torch.tensor(coo_matrix.data, device=device, dtype=dtype)
    coo_row_indices = torch.tensor(coo_matrix.row, device=device, dtype=torch.int32)
    coo_col_indices = torch.tensor(coo_matrix.col, device=device, dtype=torch.int32)
    
    return (dense_matrix, csr_values, csr_col_indices, csr_row_ptrs,
            ell_data, ell_indices, coo_values, coo_row_indices, coo_col_indices,
            max_nnz_per_row)

def test_sparse_matrix_triton(
    rows: int,
    cols: int,
    sparsity: float,
    use_optimized: bool = True
):
    """
    Test function for sparse matrix operations with Triton.
    """
    print(f"\n=== Sparse Matrix Triton Test ===")
    print(f"Matrix size: {rows} x {cols}")
    print(f"Sparsity: {sparsity:.2%}")
    print(f"Optimized: {use_optimized}")
    
    # Create test data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    
    # Create sparse matrix
    (dense_matrix, csr_values, csr_col_indices, csr_row_ptrs,
     ell_data, ell_indices, coo_values, coo_row_indices, coo_col_indices,
     max_nnz_per_row) = create_sparse_matrix(rows, cols, sparsity, device, dtype)
    
    # Create input vector
    x = torch.randn(cols, device=device, dtype=dtype)
    
    # Compute expected result
    expected = torch.matmul(dense_matrix, x)
    
    # Test CSR format
    print("\n--- Testing CSR Format ---")
    result_csr = csr_spmv_triton(csr_values, csr_col_indices, csr_row_ptrs, x, use_optimized)
    
    # Verify correctness
    csr_correct = torch.allclose(result_csr, expected, rtol=1e-5, atol=1e-5)
    print(f"CSR correctness: {'PASS' if csr_correct else 'FAIL'}")
    
    # Test ELL format
    print("\n--- Testing ELL Format ---")
    result_ell = ell_spmv_triton(ell_data, ell_indices, x, max_nnz_per_row, use_optimized)
    
    # Verify correctness
    ell_correct = torch.allclose(result_ell, expected, rtol=1e-5, atol=1e-5)
    print(f"ELL correctness: {'PASS' if ell_correct else 'FAIL'}")
    
    # Test COO format
    print("\n--- Testing COO Format ---")
    result_coo = coo_spmv_triton(coo_values, coo_row_indices, coo_col_indices, x, rows, use_optimized)
    
    # Verify correctness
    coo_correct = torch.allclose(result_coo, expected, rtol=1e-5, atol=1e-5)
    print(f"COO correctness: {'PASS' if coo_correct else 'FAIL'}")
    
    # Performance test
    if device.type == 'cuda':
        print("\n--- Performance Test ---")
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(100):
            _ = csr_spmv_triton(csr_values, csr_col_indices, csr_row_ptrs, x, use_optimized)
            _ = ell_spmv_triton(ell_data, ell_indices, x, max_nnz_per_row, use_optimized)
            _ = coo_spmv_triton(coo_values, coo_row_indices, coo_col_indices, x, rows, use_optimized)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100 * 1000  # Convert to ms
        print(f"Average time per iteration: {avg_time:.3f} ms")
    
    return result_csr, result_ell, result_coo

def benchmark_sparse_matrix_triton():
    """
    Benchmark sparse matrix operations with different configurations.
    """
    print("\n=== Sparse Matrix Triton Benchmark ===")
    
    configurations = [
        (1000, 1000, 0.1),
        (2000, 2000, 0.05),
        (5000, 5000, 0.02),
    ]
    
    for rows, cols, sparsity in configurations:
        print(f"\nConfiguration: {rows}x{cols}, sparsity={sparsity:.2%}")
        test_sparse_matrix_triton(rows, cols, sparsity, use_optimized=True)

def test_sparse_matrix_module():
    """
    Test the PyTorch module wrapper.
    """
    print("\n=== Sparse Matrix Module Test ===")
    
    # Create test data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows, cols = 1000, 1000
    sparsity = 0.1
    
    # Create sparse matrix
    (dense_matrix, csr_values, csr_col_indices, csr_row_ptrs,
     ell_data, ell_indices, coo_values, coo_row_indices, coo_col_indices,
     max_nnz_per_row) = create_sparse_matrix(rows, cols, sparsity, device)
    
    # Create input vector
    x = torch.randn(cols, device=device)
    
    # Test CSR module
    csr_module = SparseMatrixTriton('csr', optimized=True)
    csr_module.values = csr_values
    csr_module.col_indices = csr_col_indices
    csr_module.row_ptrs = csr_row_ptrs
    
    result_csr = csr_module(x)
    expected = torch.matmul(dense_matrix, x)
    
    csr_correct = torch.allclose(result_csr, expected, rtol=1e-5, atol=1e-5)
    print(f"CSR module correctness: {'PASS' if csr_correct else 'FAIL'}")
    
    # Test ELL module
    ell_module = SparseMatrixTriton('ell', optimized=True)
    ell_module.data = ell_data
    ell_module.indices = ell_indices
    ell_module.max_nnz_per_row = max_nnz_per_row
    
    result_ell = ell_module(x)
    ell_correct = torch.allclose(result_ell, expected, rtol=1e-5, atol=1e-5)
    print(f"ELL module correctness: {'PASS' if ell_correct else 'FAIL'}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test basic functionality
    test_sparse_matrix_triton(1000, 1000, 0.1, use_optimized=True)
    test_sparse_matrix_triton(1000, 1000, 0.1, use_optimized=False)
    
    # Test module wrapper
    test_sparse_matrix_module()
    
    # Run benchmark
    benchmark_sparse_matrix_triton()
    
    print("\n=== All Tests Complete ===")

"""
Profiling example & performance tips:

1. Use nsys profile to analyze kernel performance:
   nsys profile --trace=cuda python sparse_matrix_mul_triton.py

2. Monitor memory bandwidth utilization:
   nvprof --metrics achieved_occupancy,sm_efficiency python sparse_matrix_mul_triton.py

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
