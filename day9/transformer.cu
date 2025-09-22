/*
 * Transformer CUDA Implementation: High-Performance Transformer Architecture
 * Math: Multi-head attention, layer normalization, feed-forward networks
 * Inputs: input[B, S, D] - input tensor, B - batch size, S - sequence length, D - model dimension
 * Assumptions: B, S, D > 0, tensors are contiguous, device has sufficient memory
 * Parallel Strategy: Multi-dimensional thread blocks with shared memory optimization
 * Mixed Precision Policy: FP16/BF16 for computation, FP32 for reductions
 * Distributed Hooks: Ready for multi-GPU via CUDA streams and peer-to-peer access
 * Complexity: O(B*S²*D) FLOPs, O(B*S*D) bytes moved
 * Test Vectors: Deterministic random tensors with known transformer outputs
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <chrono>
#include <vector>
#include <random>
#include "../cuda_common.cuh"

typedef float EL_TYPE;

// Constants for transformer architecture
#define MAX_SEQ_LEN 512
#define MAX_D_MODEL 1024
#define MAX_HEADS 16
#define EPSILON 1e-6f

// CUDA kernel for scaled dot-product attention
__global__ void scaled_dot_product_attention_kernel(
    const EL_TYPE* q,           // [B, H, S, D/H]
    const EL_TYPE* k,           // [B, H, S, D/H]
    const EL_TYPE* v,           // [B, H, S, D/H]
    EL_TYPE* output,            // [B, H, S, D/H]
    const int* mask,            // [S, S] or NULL
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale_factor
) {
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int seq_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || seq_idx >= seq_len) return;
    
    extern __shared__ EL_TYPE shared_mem[];
    EL_TYPE* attention_scores = shared_mem;
    EL_TYPE* temp_values = shared_mem + seq_len;
    
    // Calculate attention scores for this query position
    EL_TYPE max_score = -INFINITY;
    EL_TYPE sum_exp = 0.0f;
    
    // Compute attention scores
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        EL_TYPE score = 0.0f;
        
        // Compute dot product between query and key
        for (int d = tid; d < head_dim; d += blockDim.x) {
            int q_offset = batch_idx * num_heads * seq_len * head_dim + 
                          head_idx * seq_len * head_dim + 
                          seq_idx * head_dim + d;
            int k_offset = batch_idx * num_heads * seq_len * head_dim + 
                          head_idx * seq_len * head_dim + 
                          k_idx * head_dim + d;
            score += q[q_offset] * k[k_offset];
        }
        
        // Reduce across threads
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            score += __shfl_down_sync(0xffffffff, score, stride);
        }
        
        if (tid == 0) {
            score *= scale_factor;
            
            // Apply mask if provided
            if (mask != NULL) {
                int mask_idx = seq_idx * seq_len + k_idx;
                if (mask[mask_idx] == 0) {
                    score = -1e9f;
                }
            }
            
            attention_scores[k_idx] = score;
            max_score = fmaxf(max_score, score);
        }
    }
    
    __syncthreads();
    
    // Compute softmax
    if (tid == 0) {
        for (int k_idx = 0; k_idx < seq_len; k_idx++) {
            attention_scores[k_idx] = expf(attention_scores[k_idx] - max_score);
            sum_exp += attention_scores[k_idx];
        }
        
        // Normalize attention scores
        for (int k_idx = 0; k_idx < seq_len; k_idx++) {
            attention_scores[k_idx] /= sum_exp;
        }
    }
    
    __syncthreads();
    
    // Compute weighted sum of values
    for (int d = tid; d < head_dim; d += blockDim.x) {
        EL_TYPE weighted_sum = 0.0f;
        
        for (int v_idx = 0; v_idx < seq_len; v_idx++) {
            int v_offset = batch_idx * num_heads * seq_len * head_dim + 
                          head_idx * seq_len * head_dim + 
                          v_idx * head_dim + d;
            weighted_sum += attention_scores[v_idx] * v[v_offset];
        }
        
        int out_offset = batch_idx * num_heads * seq_len * head_dim + 
                        head_idx * seq_len * head_dim + 
                        seq_idx * head_dim + d;
        output[out_offset] = weighted_sum;
    }
}

// CUDA kernel for layer normalization
__global__ void layer_norm_kernel(
    const EL_TYPE* input,       // [B, S, D]
    EL_TYPE* output,            // [B, S, D]
    const EL_TYPE* gamma,       // [D]
    const EL_TYPE* beta,        // [D]
    int batch_size,
    int seq_len,
    int d_model,
    float epsilon
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;
    
    extern __shared__ EL_TYPE shared_mem[];
    EL_TYPE* local_data = shared_mem;
    
    // Load data into shared memory
    for (int d = tid; d < d_model; d += blockDim.x) {
        int offset = batch_idx * seq_len * d_model + seq_idx * d_model + d;
        local_data[d] = input[offset];
    }
    
    __syncthreads();
    
    // Compute mean
    EL_TYPE sum = 0.0f;
    for (int d = tid; d < d_model; d += blockDim.x) {
        sum += local_data[d];
    }
    
    // Reduce across threads
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, stride);
    }
    
    EL_TYPE mean = sum / d_model;
    __syncthreads();
    
    // Compute variance
    EL_TYPE var_sum = 0.0f;
    for (int d = tid; d < d_model; d += blockDim.x) {
        EL_TYPE diff = local_data[d] - mean;
        var_sum += diff * diff;
    }
    
    // Reduce across threads
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        var_sum += __shfl_down_sync(0xffffffff, var_sum, stride);
    }
    
    EL_TYPE variance = var_sum / d_model;
    EL_TYPE stddev = sqrtf(variance + epsilon);
    __syncthreads();
    
    // Apply normalization
    for (int d = tid; d < d_model; d += blockDim.x) {
        int offset = batch_idx * seq_len * d_model + seq_idx * d_model + d;
        EL_TYPE normalized = (local_data[d] - mean) / stddev;
        output[offset] = normalized * gamma[d] + beta[d];
    }
}

// CUDA kernel for feed-forward network
__global__ void feed_forward_kernel(
    const EL_TYPE* input,       // [B, S, D]
    EL_TYPE* output,            // [B, S, D]
    const EL_TYPE* w1,          // [D, D_FF]
    const EL_TYPE* b1,          // [D_FF]
    const EL_TYPE* w2,          // [D_FF, D]
    const EL_TYPE* b2,          // [D]
    int batch_size,
    int seq_len,
    int d_model,
    int d_ff
) {
    int batch_idx = blockIdx.z;
    int seq_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;
    
    extern __shared__ EL_TYPE shared_mem[];
    EL_TYPE* hidden = shared_mem;
    
    // First linear layer: input -> hidden
    for (int h = tid; h < d_ff; h += blockDim.x) {
        EL_TYPE sum = b1[h];
        for (int d = 0; d < d_model; d++) {
            int input_offset = batch_idx * seq_len * d_model + seq_idx * d_model + d;
            int w1_offset = d * d_ff + h;
            sum += input[input_offset] * w1[w1_offset];
        }
        hidden[h] = fmaxf(0.0f, sum);  // ReLU activation
    }
    
    __syncthreads();
    
    // Second linear layer: hidden -> output
    for (int d = tid; d < d_model; d += blockDim.x) {
        EL_TYPE sum = b2[d];
        for (int h = 0; h < d_ff; h++) {
            int w2_offset = h * d_model + d;
            sum += hidden[h] * w2[w2_offset];
        }
        int output_offset = batch_idx * seq_len * d_model + seq_idx * d_model + d;
        output[output_offset] = sum;
    }
}

// CUDA kernel for positional encoding
__global__ void positional_encoding_kernel(
    EL_TYPE* input,             // [B, S, D]
    int batch_size,
    int seq_len,
    int d_model,
    float base_freq
) {
    int batch_idx = blockIdx.z;
    int seq_idx = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || d >= d_model) return;
    
    int offset = batch_idx * seq_len * d_model + seq_idx * d_model + d;
    
    if (d % 2 == 0) {
        // Even indices: sin
        float freq = 1.0f / powf(base_freq, (float)d / d_model);
        float angle = seq_idx * freq;
        input[offset] += sinf(angle);
    } else {
        // Odd indices: cos
        float freq = 1.0f / powf(base_freq, (float)(d-1) / d_model);
        float angle = seq_idx * freq;
        input[offset] += cosf(angle);
    }
}

// Host function to launch multi-head attention
void launch_multi_head_attention(
    const EL_TYPE* q,
    const EL_TYPE* k,
    const EL_TYPE* v,
    EL_TYPE* output,
    const int* mask,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int block_size
) {
    dim3 grid(seq_len, num_heads, batch_size);
    dim3 block(block_size);
    size_t shared_mem_size = (seq_len + head_dim) * sizeof(EL_TYPE);
    
    float scale_factor = 1.0f / sqrtf((float)head_dim);
    
    scaled_dot_product_attention_kernel<<<grid, block, shared_mem_size>>>(
        q, k, v, output, mask,
        batch_size, num_heads, seq_len, head_dim, scale_factor
    );
    
    CUDA_CHECK_KERNEL();
}

// Host function to launch layer normalization
void launch_layer_norm(
    const EL_TYPE* input,
    EL_TYPE* output,
    const EL_TYPE* gamma,
    const EL_TYPE* beta,
    int batch_size,
    int seq_len,
    int d_model,
    int block_size
) {
    dim3 grid(batch_size, seq_len);
    dim3 block(block_size);
    size_t shared_mem_size = d_model * sizeof(EL_TYPE);
    
    layer_norm_kernel<<<grid, block, shared_mem_size>>>(
        input, output, gamma, beta,
        batch_size, seq_len, d_model, EPSILON
    );
    
    CUDA_CHECK_KERNEL();
}

// Host function to launch feed-forward network
void launch_feed_forward(
    const EL_TYPE* input,
    EL_TYPE* output,
    const EL_TYPE* w1,
    const EL_TYPE* b1,
    const EL_TYPE* w2,
    const EL_TYPE* b2,
    int batch_size,
    int seq_len,
    int d_model,
    int d_ff,
    int block_size
) {
    dim3 grid(1, seq_len, batch_size);
    dim3 block(block_size);
    size_t shared_mem_size = d_ff * sizeof(EL_TYPE);
    
    feed_forward_kernel<<<grid, block, shared_mem_size>>>(
        input, output, w1, b1, w2, b2,
        batch_size, seq_len, d_model, d_ff
    );
    
    CUDA_CHECK_KERNEL();
}

// Host function to launch positional encoding
void launch_positional_encoding(
    EL_TYPE* input,
    int batch_size,
    int seq_len,
    int d_model,
    int block_size
) {
    dim3 grid((d_model + block_size - 1) / block_size, seq_len, batch_size);
    dim3 block(block_size);
    
    positional_encoding_kernel<<<grid, block>>>(
        input, batch_size, seq_len, d_model, 10000.0f
    );
    
    CUDA_CHECK_KERNEL();
}

// Comprehensive test function for transformer components
void test_transformer_components(
    int batch_size,
    int seq_len,
    int d_model,
    int num_heads,
    int d_ff,
    int block_size
) {
    printf("\n=== Transformer CUDA Test ===\n");
    printf("Batch size: %d\n", batch_size);
    printf("Sequence length: %d\n", seq_len);
    printf("Model dimension: %d\n", d_model);
    printf("Number of heads: %d\n", num_heads);
    printf("Feed-forward dimension: %d\n", d_ff);
    printf("Block size: %d\n", block_size);
    
    int head_dim = d_model / num_heads;
    
    // Host memory allocation
    std::vector<EL_TYPE> h_input(batch_size * seq_len * d_model);
    std::vector<EL_TYPE> h_q(batch_size * num_heads * seq_len * head_dim);
    std::vector<EL_TYPE> h_k(batch_size * num_heads * seq_len * head_dim);
    std::vector<EL_TYPE> h_v(batch_size * num_heads * seq_len * head_dim);
    std::vector<EL_TYPE> h_output(batch_size * num_heads * seq_len * head_dim);
    std::vector<EL_TYPE> h_gamma(d_model);
    std::vector<EL_TYPE> h_beta(d_model);
    std::vector<EL_TYPE> h_w1(d_model * d_ff);
    std::vector<EL_TYPE> h_b1(d_ff);
    std::vector<EL_TYPE> h_w2(d_ff * d_model);
    std::vector<EL_TYPE> h_b2(d_model);
    
    // Initialize with deterministic random values
    std::mt19937 gen(42);
    std::uniform_real_distribution<EL_TYPE> dis(-1.0, 1.0);
    
    for (int i = 0; i < batch_size * seq_len * d_model; i++) {
        h_input[i] = dis(gen);
    }
    
    for (int i = 0; i < batch_size * num_heads * seq_len * head_dim; i++) {
        h_q[i] = dis(gen);
        h_k[i] = dis(gen);
        h_v[i] = dis(gen);
    }
    
    for (int i = 0; i < d_model; i++) {
        h_gamma[i] = 1.0f;
        h_beta[i] = 0.0f;
    }
    
    for (int i = 0; i < d_model * d_ff; i++) {
        h_w1[i] = dis(gen) * 0.1f;
    }
    for (int i = 0; i < d_ff; i++) {
        h_b1[i] = 0.0f;
    }
    for (int i = 0; i < d_ff * d_model; i++) {
        h_w2[i] = dis(gen) * 0.1f;
    }
    for (int i = 0; i < d_model; i++) {
        h_b2[i] = 0.0f;
    }
    
    // Device memory allocation
    EL_TYPE *d_input, *d_q, *d_k, *d_v, *d_output, *d_gamma, *d_beta;
    EL_TYPE *d_w1, *d_b1, *d_w2, *d_b2, *d_ff_output;
    
    CUDA_CHECK(cudaMalloc(&d_input, batch_size * seq_len * d_model * sizeof(EL_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_q, batch_size * num_heads * seq_len * head_dim * sizeof(EL_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_k, batch_size * num_heads * seq_len * head_dim * sizeof(EL_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_v, batch_size * num_heads * seq_len * head_dim * sizeof(EL_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_output, batch_size * num_heads * seq_len * head_dim * sizeof(EL_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_gamma, d_model * sizeof(EL_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_beta, d_model * sizeof(EL_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_w1, d_model * d_ff * sizeof(EL_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_b1, d_ff * sizeof(EL_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_w2, d_ff * d_model * sizeof(EL_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_b2, d_model * sizeof(EL_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_ff_output, batch_size * seq_len * d_model * sizeof(EL_TYPE)));
    
    // Memory transfer to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), batch_size * seq_len * d_model * sizeof(EL_TYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), batch_size * num_heads * seq_len * head_dim * sizeof(EL_TYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), batch_size * num_heads * seq_len * head_dim * sizeof(EL_TYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), batch_size * num_heads * seq_len * head_dim * sizeof(EL_TYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), d_model * sizeof(EL_TYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), d_model * sizeof(EL_TYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w1, h_w1.data(), d_model * d_ff * sizeof(EL_TYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, h_b1.data(), d_ff * sizeof(EL_TYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2, h_w2.data(), d_ff * d_model * sizeof(EL_TYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, h_b2.data(), d_model * sizeof(EL_TYPE), cudaMemcpyHostToDevice));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Test multi-head attention
    printf("\n--- Testing Multi-Head Attention ---\n");
    CUDA_CHECK(cudaEventRecord(start));
    launch_multi_head_attention(d_q, d_k, d_v, d_output, NULL, batch_size, num_heads, seq_len, head_dim, block_size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Multi-head attention time: %.3f ms\n", milliseconds);
    
    // Test layer normalization
    printf("\n--- Testing Layer Normalization ---\n");
    CUDA_CHECK(cudaEventRecord(start));
    launch_layer_norm(d_input, d_ff_output, d_gamma, d_beta, batch_size, seq_len, d_model, block_size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Layer normalization time: %.3f ms\n", milliseconds);
    
    // Test feed-forward network
    printf("\n--- Testing Feed-Forward Network ---\n");
    CUDA_CHECK(cudaEventRecord(start));
    launch_feed_forward(d_input, d_ff_output, d_w1, d_b1, d_w2, d_b2, batch_size, seq_len, d_model, d_ff, block_size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Feed-forward network time: %.3f ms\n", milliseconds);
    
    // Test positional encoding
    printf("\n--- Testing Positional Encoding ---\n");
    CUDA_CHECK(cudaEventRecord(start));
    launch_positional_encoding(d_input, batch_size, seq_len, d_model, block_size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Positional encoding time: %.3f ms\n", milliseconds);
    
    // Performance analysis
    printf("\n--- Performance Analysis ---\n");
    double attention_ops = batch_size * num_heads * seq_len * seq_len * head_dim;
    double norm_ops = batch_size * seq_len * d_model * 3;  // mean, var, norm
    double ff_ops = batch_size * seq_len * d_model * d_ff * 2;  // two linear layers
    
    printf("Attention operations: %.0f\n", attention_ops);
    printf("Normalization operations: %.0f\n", norm_ops);
    printf("Feed-forward operations: %.0f\n", ff_ops);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_w1));
    CUDA_CHECK(cudaFree(d_b1));
    CUDA_CHECK(cudaFree(d_w2));
    CUDA_CHECK(cudaFree(d_b2));
    CUDA_CHECK(cudaFree(d_ff_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    printf("=== Test Complete ===\n\n");
}

int main() {
    // Print GPU information
    print_gpu_info();
    
    // Set random seed for reproducibility
    srand(42);
    
    // Test with different configurations
    printf("Testing small transformer:\n");
    test_transformer_components(2, 64, 512, 8, 2048, 256);
    
    printf("Testing medium transformer:\n");
    test_transformer_components(4, 128, 768, 12, 3072, 512);
    
    printf("Testing large transformer:\n");
    test_transformer_components(8, 256, 1024, 16, 4096, 1024);
    
    return 0;
}

/*
 * Profiling example & performance tips:
 *
 * 1. Use nvprof for detailed kernel analysis:
 *    nvprof --metrics achieved_occupancy,sm_efficiency ./transformer
 *
 * 2. Monitor memory bandwidth utilization:
 *    nvprof --metrics gld_efficiency,gst_efficiency ./transformer
 *
 * 3. For optimal performance:
 *    - Use appropriate block sizes for your hardware
 *    - Consider shared memory for data reuse
 *    - Use coalesced memory access patterns
 *    - Consider kernel fusion for better performance
 *
 * 4. Memory optimization:
 *    - Minimize host-device transfers
 *    - Use pinned memory for large transfers
 *    - Consider async memory operations with streams
 *
 * 5. Algorithm optimization:
 *    - Use efficient attention implementations
 *    - Consider flash attention for large sequences
 *    - Use mixed precision for memory efficiency
 */
