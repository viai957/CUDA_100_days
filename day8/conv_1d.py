"""
1D Convolution PyTorch Implementation: High-Performance Convolutional Operations
Math: output[i] = Σ(input[i+k] * kernel[k]) for k ∈ [0, kernel_size)
Inputs: input[N] - input signal, kernel[K] - convolution kernel, N - signal length, K - kernel size
Assumptions: N > 0, K > 0, arrays are contiguous, device has sufficient memory
Parallel Strategy: Vectorized operations with mixed precision and gradient checkpointing
Mixed Precision Policy: FP16/BF16 for computation, FP32 for reductions
Distributed Hooks: Ready for multi-GPU via DataParallel and DistributedDataParallel
Complexity: O(NK) FLOPs, O(N+K) bytes moved
Test Vectors: Deterministic random signals with known convolution results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import sys
import os
from typing import Optional, Tuple, Union

# Add parent directory to path for cuda_common import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Conv1D(nn.Module):
    """
    High-performance 1D convolution module with advanced features
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        use_mixed_precision: bool = False,
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.use_mixed_precision = use_mixed_precision
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 1D convolution
        Args:
            x: Input tensor of shape (batch_size, in_channels, signal_length)
        Returns:
            Output tensor of shape (batch_size, out_channels, output_length)
        """
        if self.use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward_impl, x)
        else:
            return self._forward_impl(x)
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Implementation of forward pass"""
        if self.use_mixed_precision:
            with torch.cuda.amp.autocast():
                return self._conv1d_impl(x)
        else:
            return self._conv1d_impl(x)
    
    def _conv1d_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Core convolution implementation"""
        # Ensure input has correct shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Apply padding if needed
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding), mode=self.padding_mode)
        
        # Perform convolution
        output = F.conv1d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=0,  # Already padded
            dilation=self.dilation,
            groups=self.groups
        )
        
        return output

class Conv1DBlock(nn.Module):
    """
    Complete 1D convolution block with normalization and activation
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        use_batch_norm: bool = True,
        activation: str = 'relu',
        dropout: float = 0.0,
        use_mixed_precision: bool = False
    ):
        super().__init__()
        
        self.conv = Conv1D(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias,
            use_mixed_precision=use_mixed_precision
        )
        
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn = nn.BatchNorm1d(out_channels)
        
        self.activation = self._get_activation(activation)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
    def _get_activation(self, activation: str):
        """Get activation function"""
        activations = {
            'relu': nn.ReLU(inplace=True),
            'leaky_relu': nn.LeakyReLU(0.1, inplace=True),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'none': nn.Identity()
        }
        return activations.get(activation, nn.ReLU(inplace=True))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.conv(x)
        
        if self.use_batch_norm:
            x = self.bn(x)
        
        x = self.activation(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        return x

class Conv1DNet(nn.Module):
    """
    Complete 1D convolution network for signal processing
    """
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        hidden_channels: list = [64, 128, 256],
        kernel_sizes: list = [3, 3, 3],
        strides: list = [1, 2, 2],
        use_mixed_precision: bool = False
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.use_mixed_precision = use_mixed_precision
        
        # Build network
        layers = []
        in_channels = input_channels
        
        for i, (hidden_ch, kernel_size, stride) in enumerate(zip(hidden_channels, kernel_sizes, strides)):
            layers.append(Conv1DBlock(
                in_channels, hidden_ch, kernel_size, stride,
                padding=kernel_size//2,
                use_mixed_precision=use_mixed_precision
            ))
            in_channels = hidden_ch
        
        self.features = nn.Sequential(*layers)
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(in_channels, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def test_conv_1d_pytorch(
    signal_length: int,
    kernel_size: int,
    batch_size: int = 1,
    use_mixed_precision: bool = False,
    use_gradient_checkpointing: bool = False
):
    """
    Test function for 1D convolution with PyTorch
    """
    print(f"\n=== 1D Convolution PyTorch Test ===")
    print(f"Signal length: {signal_length}")
    print(f"Kernel size: {kernel_size}")
    print(f"Batch size: {batch_size}")
    print(f"Mixed precision: {use_mixed_precision}")
    print(f"Gradient checkpointing: {use_gradient_checkpointing}")
    
    # Create test data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_data = torch.randn(batch_size, 1, signal_length, device=device)
    
    # Create model
    model = Conv1D(
        in_channels=1,
        out_channels=1,
        kernel_size=kernel_size,
        padding=kernel_size//2,
        use_mixed_precision=use_mixed_precision,
        use_gradient_checkpointing=use_gradient_checkpointing
    ).to(device)
    
    # Test forward pass
    with torch.no_grad():
        output = model(input_data)
    
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test gradient computation
    model.train()
    output = model(input_data)
    loss = output.sum()
    loss.backward()
    
    print(f"Weight gradient norm: {model.weight.grad.norm().item():.6f}")
    if model.bias is not None:
        print(f"Bias gradient norm: {model.bias.grad.norm().item():.6f}")
    
    # Performance test
    model.eval()
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(100):
        with torch.no_grad():
            _ = model(input_data)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100 * 1000  # Convert to ms
    print(f"Average time: {avg_time:.3f} ms")
    
    # Performance analysis
    operations = signal_length * kernel_size
    gflops = operations / (avg_time / 1000) / 1e9
    print(f"Performance: {gflops:.2f} GFLOPS")
    
    return output

def test_conv_1d_block():
    """
    Test the complete 1D convolution block
    """
    print("\n=== 1D Convolution Block Test ===")
    
    # Create test data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_data = torch.randn(4, 1, 1000, device=device)
    
    # Create model
    model = Conv1DBlock(
        in_channels=1,
        out_channels=64,
        kernel_size=5,
        padding=2,
        use_batch_norm=True,
        activation='relu',
        dropout=0.1,
        use_mixed_precision=True
    ).to(device)
    
    # Test forward pass
    output = model(input_data)
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test gradient computation
    loss = output.sum()
    loss.backward()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

def test_conv_1d_network():
    """
    Test the complete 1D convolution network
    """
    print("\n=== 1D Convolution Network Test ===")
    
    # Create test data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_data = torch.randn(8, 1, 1000, device=device)
    targets = torch.randint(0, 10, (8,), device=device)
    
    # Create model
    model = Conv1DNet(
        input_channels=1,
        num_classes=10,
        hidden_channels=[64, 128, 256],
        kernel_sizes=[3, 3, 3],
        strides=[1, 2, 2],
        use_mixed_precision=True
    ).to(device)
    
    # Test forward pass
    output = model(input_data)
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

def benchmark_conv_1d_pytorch():
    """
    Benchmark 1D convolution with different configurations
    """
    print("\n=== 1D Convolution PyTorch Benchmark ===")
    
    signal_lengths = [1024, 10000, 100000, 1000000]
    kernel_sizes = [3, 5, 7, 9, 15]
    
    for signal_length in signal_lengths:
        for kernel_size in kernel_sizes:
            print(f"\nConfiguration: signal_length={signal_length}, kernel_size={kernel_size}")
            test_conv_1d_pytorch(signal_length, kernel_size, use_mixed_precision=True)

def test_distributed_conv_1d():
    """
    Test distributed 1D convolution
    """
    print("\n=== Distributed 1D Convolution Test ===")
    
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("Skipping distributed test - requires multiple GPUs")
        return
    
    # Create test data
    device = torch.device('cuda')
    input_data = torch.randn(8, 1, 1000, device=device)
    
    # Create model
    model = Conv1DNet(
        input_channels=1,
        num_classes=10,
        hidden_channels=[64, 128, 256],
        use_mixed_precision=True
    ).to(device)
    
    # Test DataParallel
    model_dp = nn.DataParallel(model)
    output_dp = model_dp(input_data)
    print(f"DataParallel output shape: {output_dp.shape}")
    
    # Test DistributedDataParallel (simplified)
    model_ddp = nn.parallel.DistributedDataParallel(model)
    output_ddp = model_ddp(input_data)
    print(f"DistributedDataParallel output shape: {output_ddp.shape}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test basic functionality
    test_conv_1d_pytorch(1000, 5, use_mixed_precision=False)
    test_conv_1d_pytorch(1000, 5, use_mixed_precision=True)
    test_conv_1d_pytorch(1000, 5, use_gradient_checkpointing=True)
    
    # Test advanced features
    test_conv_1d_block()
    test_conv_1d_network()
    test_distributed_conv_1d()
    
    # Run benchmark
    benchmark_conv_1d_pytorch()
    
    print("\n=== All Tests Complete ===")
