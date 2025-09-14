## GPU 100 days Learning Journey
##### This document serves as a log of the progress and knowledge I gained while working on GPU programming and studying the PMPP (Parallel Programming and Optimization) book.

Mentor: https://github.com/hkproj/
Bro in the 100 days challenge: https://github.com/1y33/100Days

> Day 1

Files: rope.py, RoPE.cu, RoPE_triton.py
Summary:
Implemented Rotary Position Embedding (RoPE) in three different approaches:

**PyTorch Implementation (rope.py):**
- Complex number-based rotation using torch.polar and torch.view_as_complex
- Precomputed frequency tensors for efficient batch processing
- Supports arbitrary sequence lengths and head dimensions

**CUDA Implementation (RoPE.cu):**
- Direct GPU kernel with manual memory management
- Optimized for coalesced memory access patterns
- Block-based parallelism across batch, sequence, and head dimensions
- Fused rotation operations for both query and key tensors

**Triton Implementation (RoPE_triton.py):**
- Production-grade kernel with autotuning for different hardware configurations
- Advanced memory layout optimization with configurable block sizes
- Mixed precision support (FP16/BF16) with FP32 reductions
- Comprehensive error handling and bounds checking
- Module wrapper for seamless PyTorch integration
- Built-in performance benchmarking and unit testing

**Key Learnings:**
- RoPE position-dependent rotations to query and key vectors
- Complex number representation enables efficient 2D rotations
- Memory layout optimization critical for GPU performance
- Triton provides higher-level abstractions while maintaining CUDA-level performance
- Autotuning essential for optimal performance across different hardware

**Mathematical Foundation:**
- Frequency calculation: θ_i = 10000^(-2i/d) for i ∈ [0, d/2)
- Position encoding: freq[m,i] = m * θ_i
- Rotation matrix: R(θ) = [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
- Applied to consecutive pairs of vector dimensions

Reading:

> Day 2
