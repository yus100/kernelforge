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
