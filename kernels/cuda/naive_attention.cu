/**
 * Naive attention baseline in CUDA.
 *
 * Educational reference implementation: each thread computes one row of the
 * output matrix O = softmax(scale * Q @ K^T) @ V. All loads come from global
 * memory with no shared-memory blocking or tiling.
 *
 * This is intentionally unoptimized to serve as a correctness baseline and a
 * clear starting point for understanding the memory-access patterns that later
 * kernels (shared-memory tiled, FlashAttention, Hopper TMA) aim to improve.
 *
 * Inputs:  Q, K, V of shape (B, H, S, D) in row-major, fp16 or fp32.
 * Output:  O of shape (B, H, S, D).
 * Launch:  grid(S, B*H), block(1) -- one thread per query row per head.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct NaiveAttentionConfig {
    int batch_size;
    int num_heads;
    int seq_len;
    int head_dim;
    float scale;      // 1/sqrt(head_dim) by default
    bool causal;
};

/// Helper: initialise config with sensible defaults.
inline NaiveAttentionConfig make_default_config(int B, int H, int S, int D) {
    NaiveAttentionConfig cfg;
    cfg.batch_size = B;
    cfg.num_heads  = H;
    cfg.seq_len    = S;
    cfg.head_dim   = D;
    cfg.scale      = 1.0f / sqrtf(static_cast<float>(D));
    cfg.causal     = false;
    return cfg;
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

/**
 * naive_attention_kernel: one thread computes one output row O[row, :].
 *
 * Grid:  (S, B*H)   -- one thread per (query-row, batch-head pair)
 * Block: (1,)
 *
 * Two-pass numerically-stable softmax:
 *   Pass 1: row_max = max_j(scale * Q[row] . K[j])
 *   Pass 2: for each j, compute exp(score - row_max), accumulate sum and
 *           weighted V contribution into O.
 *   Final:  O[row] /= exp_sum
 *
 * All data lives in global memory; no shared memory is used.
 */
__global__ void naive_attention_kernel(
    const float* __restrict__ Q,   // (B*H, S, D)
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ O,
    int seq_len,
    int head_dim,
    float scale,
    bool causal
) {
    // Which query row and which (batch, head) pair
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int bh  = blockIdx.y;

    if (row >= seq_len) return;

    // Pointers into this (batch, head) slice
    int slice_offset = bh * seq_len * head_dim;
    const float* Q_row = Q + slice_offset + row * head_dim;
    const float* K_base = K + slice_offset;
    const float* V_base = V + slice_offset;
    float*       O_row = O + slice_offset + row * head_dim;

    int kv_len = causal ? (row + 1) : seq_len;

    // --- Pass 1: find row max for numerical stability ---
    float row_max = -FLT_MAX;
    for (int j = 0; j < kv_len; j++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += Q_row[d] * K_base[j * head_dim + d];
        }
        dot *= scale;
        if (dot > row_max) row_max = dot;
    }

    // --- Pass 2: accumulate exp(score - max) and weighted V ---
    float exp_sum = 0.0f;

    // Zero the output row
    for (int d = 0; d < head_dim; d++) {
        O_row[d] = 0.0f;
    }

    for (int j = 0; j < kv_len; j++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += Q_row[d] * K_base[j * head_dim + d];
        }
        dot *= scale;
        float exp_val = expf(dot - row_max);
        exp_sum += exp_val;

        for (int d = 0; d < head_dim; d++) {
            O_row[d] += exp_val * V_base[j * head_dim + d];
        }
    }

    // --- Normalize ---
    float inv_sum = 1.0f / exp_sum;
    for (int d = 0; d < head_dim; d++) {
        O_row[d] *= inv_sum;
    }
}

// ---------------------------------------------------------------------------
// Host launch wrapper
// ---------------------------------------------------------------------------

/**
 * launch_naive_attention: allocates output and launches the kernel.
 *
 * Q, K, V, O are device pointers of shape (B*H, S, D) in row-major fp32.
 * Caller is responsible for memory allocation of O (same size as Q).
 */
void launch_naive_attention(
    const float* Q,
    const float* K,
    const float* V,
    float*       O,
    const NaiveAttentionConfig& cfg
) {
    int S = cfg.seq_len;
    int BH = cfg.batch_size * cfg.num_heads;

    // One thread per query row, one block-row per (batch, head).
    dim3 block(1);
    dim3 grid(S, BH);

    naive_attention_kernel<<<grid, block>>>(
        Q, K, V, O,
        S, cfg.head_dim, cfg.scale, cfg.causal
    );

    // Check for launch errors (educational builds only; remove in production)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "naive_attention_kernel launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

// ---------------------------------------------------------------------------
// Minimal test / example (compile with: nvcc -o naive_attn naive_attention.cu)
// ---------------------------------------------------------------------------

#ifdef NAIVE_ATTN_MAIN
int main() {
    int B = 1, H = 4, S = 128, D = 64;
    NaiveAttentionConfig cfg = make_default_config(B, H, S, D);

    size_t size = (size_t)B * H * S * D * sizeof(float);

    float *h_Q, *h_K, *h_V, *h_O;
    h_Q = (float*)malloc(size);
    h_K = (float*)malloc(size);
    h_V = (float*)malloc(size);
    h_O = (float*)malloc(size);

    // Fill with small random values
    for (size_t i = 0; i < (size_t)B * H * S * D; i++) {
        h_Q[i] = 0.01f * (float)(i % 37 - 18);
        h_K[i] = 0.01f * (float)(i % 41 - 20);
        h_V[i] = 0.01f * (float)(i % 31 - 15);
    }

    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, size);
    cudaMalloc(&d_K, size);
    cudaMalloc(&d_V, size);
    cudaMalloc(&d_O, size);

    cudaMemcpy(d_Q, h_Q, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, size, cudaMemcpyHostToDevice);

    launch_naive_attention(d_Q, d_K, d_V, d_O, cfg);
    cudaDeviceSynchronize();

    cudaMemcpy(h_O, d_O, size, cudaMemcpyDeviceToHost);

    printf("O[0][0][0][0..3] = %.6f %.6f %.6f %.6f\n",
           h_O[0], h_O[1], h_O[2], h_O[3]);

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    free(h_Q); free(h_K); free(h_V); free(h_O);

    return 0;
}
#endif
