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
