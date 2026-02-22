"""
Naive fused attention kernel using CuTe DSL.

Implements a basic fused Q·K^T·V attention pass in a single kernel launch
with no tiling or shared-memory optimization. Each thread-block handles one
(batch, head) pair and walks the full sequence length, so this is memory-bound
and intended only as a correctness reference.

Will be superseded by:
  - tiled attention (shared-memory blocking over seq_len)
  - Hopper-optimized attention (TMA + wgmma)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

# CuTe DSL is provided by the CUTLASS Python bindings
import cutlass
from cutlass.cute import (
    Layout,
    Tensor as CuteTensor,
    make_layout,
    make_tensor,
    coalesce,
)
from cutlass.cute.dsl import kernel as cute_kernel


@dataclass
class NaiveAttentionConfig:
    """Configuration for the naive fused attention kernel."""

    batch_size: int = 1
    num_heads: int = 8
    seq_len: int = 128
    head_dim: int = 64
    scale: Optional[float] = None  # defaults to 1/sqrt(head_dim)
    causal: bool = False
    dtype: torch.dtype = torch.float16

    def __post_init__(self):
        if self.scale is None:
            self.scale = self.head_dim ** -0.5


@cute_kernel
def naive_attention_kernel(
    Q: CuteTensor,       # (seq_len, head_dim)
    K: CuteTensor,       # (seq_len, head_dim)
    V: CuteTensor,       # (seq_len, head_dim)
    O: CuteTensor,       # (seq_len, head_dim)
    scale: float,
    seq_len: int,
    head_dim: int,
    causal: bool,
):
    """Naive fused attention: O = softmax(scale * Q @ K^T) @ V.

    One thread-block per (batch, head). Each thread owns one query row and
    iterates over the full key sequence to accumulate the softmax numerator
    and denominator, then writes the weighted sum to O.

    No shared memory, no tiling: purely register-level with global loads.
    """
    # Row index this thread is responsible for
    row = cutlass.cute.thread_idx_x()
    if row >= seq_len:
        return

    # --- Pass 1: compute row-max for numerical stability ---
    row_max = float("-inf")
    for j in range(seq_len):
        if causal and j > row:
            break
        dot = 0.0
        for d in range(head_dim):
            dot += float(Q[row, d]) * float(K[j, d])
        dot *= scale
        if dot > row_max:
            row_max = dot

    # --- Pass 2: compute exp(score - row_max) and accumulate sum ---
    exp_sum = 0.0
    for j in range(seq_len):
        if causal and j > row:
            break
        dot = 0.0
        for d in range(head_dim):
            dot += float(Q[row, d]) * float(K[j, d])
        dot *= scale
        exp_val = cutlass.cute.exp(dot - row_max)
        exp_sum += exp_val

        # Accumulate weighted V into O (init to zero on first iter)
        for d in range(head_dim):
            if j == 0:
                O[row, d] = 0.0
            O[row, d] = float(O[row, d]) + exp_val * float(V[j, d])

    # --- Normalize by softmax denominator ---
    inv_sum = 1.0 / exp_sum
    for d in range(head_dim):
        O[row, d] = float(O[row, d]) * inv_sum


def naive_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    config: Optional[NaiveAttentionConfig] = None,
) -> torch.Tensor:
    """Launch the naive fused attention kernel.

    Args:
        Q: Query tensor of shape (batch, heads, seq_len, head_dim).
        K: Key tensor, same shape as Q.
        V: Value tensor, same shape as Q.
        config: Optional config overriding defaults inferred from Q.

    Returns:
        O: Output tensor of shape (batch, heads, seq_len, head_dim).
    """
    B, H, S, D = Q.shape

    if config is None:
        config = NaiveAttentionConfig(
            batch_size=B,
            num_heads=H,
            seq_len=S,
            head_dim=D,
            dtype=Q.dtype,
        )

    O = torch.empty_like(Q)

    # One thread-block per (batch, head), one thread per query row.
    # Grid: (B * H,)   Block: (S,)
    grid = (B * H,)
    block = (S,)

    # Reshape to merge batch and head dims for per-block indexing
    Q_flat = Q.reshape(B * H, S, D)
    K_flat = K.reshape(B * H, S, D)
    V_flat = V.reshape(B * H, S, D)
    O_flat = O.reshape(B * H, S, D)

    # Build CuTe layouts: each block picks its (b*H + h) slice
    row_layout = make_layout((S, D))

    for bh in range(B * H):
        q_cute = make_tensor(Q_flat[bh], row_layout)
        k_cute = make_tensor(K_flat[bh], row_layout)
        v_cute = make_tensor(V_flat[bh], row_layout)
        o_cute = make_tensor(O_flat[bh], row_layout)

        naive_attention_kernel[grid, block](
            q_cute, k_cute, v_cute, o_cute,
            config.scale,
            S,
            D,
            config.causal,
        )

    return O
