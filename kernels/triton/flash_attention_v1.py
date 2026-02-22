"""
FlashAttention v1 in Triton.

Implements the original FlashAttention algorithm (Dao et al., 2022) with tiled
computation and online softmax. The key idea: iterate over key/value blocks in
the inner loop, maintaining running softmax statistics (max and denominator) so
the full attention matrix never materializes in HBM.

Algorithm outline (forward pass, one query-block):
  1. Load a block of Q rows into SRAM.
  2. For each K/V block:
     a. Compute S_block = Q_block @ K_block^T  (in SRAM)
     b. Apply causal mask if needed.
     c. Update running max and softmax denominator (online softmax).
     d. Rescale previous accumulator, add new softmax(S_block) @ V_block.
  3. Write final O_block = accumulator / denominator to HBM.

This is a single fused kernel with O(N) HBM reads (vs O(N^2) for naive).
"""

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl


@dataclass
class FlashAttentionConfig:
    """Configuration for FlashAttention v1 kernel."""

    batch_size: int = 1
    num_heads: int = 8
    seq_len: int = 1024
    head_dim: int = 64
    scale: Optional[float] = None  # defaults to 1/sqrt(head_dim)
    causal: bool = False
    BLOCK_M: int = 64   # query block size
    BLOCK_N: int = 64   # key/value block size
    dtype: torch.dtype = torch.float16

    def __post_init__(self):
        if self.scale is None:
            self.scale = self.head_dim ** -0.5


@triton.jit
def _flash_attn_v1_fwd(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    seq_len,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """Forward kernel: one program instance handles one (batch, head, q-block)."""
    # -- program IDs --
    pid_m = tl.program_id(0)  # which query block
    pid_bh = tl.program_id(1)  # which (batch, head) pair
    pid_b = pid_bh // tl.num_programs(1)  # not used directly; strides handle it
    # We pass (B*H) as grid dim 1, strides already embed batch and head offsets.

    # -- base pointers for this (batch, head) --
    qkv_offset = pid_bh.to(tl.int64) * stride_qh  # stride_qh == stride between heads
    # Actually we need to decompose: batch offset + head offset.
    # With contiguous (B, H, S, D) layout, stride_qb = H*S*D, stride_qh = S*D.
    # pid_bh = b * H + h, so offset = b * stride_qb + h * stride_qh.
    # But we can also treat the flat (B*H) view: offset = pid_bh * S * D = pid_bh * stride_qh.
    # This works because stride_qh == S * D for contiguous layout.
    Q_block_ptr = Q_ptr + pid_bh.to(tl.int64) * stride_qh
    K_block_ptr = K_ptr + pid_bh.to(tl.int64) * stride_kh
    V_block_ptr = V_ptr + pid_bh.to(tl.int64) * stride_vh
    O_block_ptr = O_ptr + pid_bh.to(tl.int64) * stride_oh

    # -- offsets within this block --
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # query row indices
    offs_d = tl.arange(0, HEAD_DIM)                     # head_dim indices

    # Load Q block: (BLOCK_M, HEAD_DIM)
    q_ptrs = Q_block_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len, other=0.0)
    q = (q * scale).to(tl.float16)

    # -- online softmax accumulators --
    # m_i: running row-wise max, shape (BLOCK_M,)
    m_i = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
    # l_i: running row-wise sum of exp, shape (BLOCK_M,)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    # acc: running weighted sum, shape (BLOCK_M, HEAD_DIM)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # -- iterate over K/V blocks --
    # For causal: only iterate up to the block that contains the last valid key
    if CAUSAL:
        kv_len = tl.minimum((pid_m + 1) * BLOCK_M, seq_len)
    else:
        kv_len = seq_len
    num_kv_blocks = tl.cdiv(kv_len, BLOCK_N)

    for j in range(0, num_kv_blocks):
        offs_n = j * BLOCK_N + tl.arange(0, BLOCK_N)  # key row indices

        # Load K block: (BLOCK_N, HEAD_DIM) -> we need K^T so load as (HEAD_DIM, BLOCK_N)?
        # Actually, Triton dot expects (M, K) @ (K, N). We want S = Q @ K^T.
        # Q is (BLOCK_M, HEAD_DIM), K^T is (HEAD_DIM, BLOCK_N).
        # Load K as (BLOCK_N, HEAD_DIM) then transpose via tl.trans or use dot with trans_b.
        k_ptrs = K_block_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=offs_n[:, None] < seq_len, other=0.0)

        # S = Q @ K^T, shape (BLOCK_M, BLOCK_N)
        s = tl.dot(q, tl.trans(k))

        # Apply causal mask: zero out positions where key index > query index
        if CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))

        # Mask out-of-bounds keys
        s = tl.where(offs_n[None, :] < seq_len, s, float("-inf"))

        # -- online softmax update --
        # New block max
        m_ij = tl.max(s, axis=1)  # (BLOCK_M,)
        m_new = tl.maximum(m_i, m_ij)

        # Correction factor for previous accumulator
        alpha = tl.exp(m_i - m_new)

        # Softmax numerator for this block
        p = tl.exp(s - m_new[:, None])  # (BLOCK_M, BLOCK_N)

        # Update running sum: rescale old sum and add new
        l_i = l_i * alpha + tl.sum(p, axis=1)

        # Rescale previous accumulator and add new contribution
        acc = acc * alpha[:, None]

        # Load V block: (BLOCK_N, HEAD_DIM)
        v_ptrs = V_block_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=offs_n[:, None] < seq_len, other=0.0)

        # acc += P @ V, shape (BLOCK_M, HEAD_DIM)
        acc += tl.dot(p.to(tl.float16), v)

        # Update running max
        m_i = m_new

    # -- finalize: normalize by softmax denominator --
    acc = acc / l_i[:, None]

    # -- store output --
    o_ptrs = O_block_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < seq_len)


def flash_attention_v1(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    config: Optional[FlashAttentionConfig] = None,
) -> torch.Tensor:
    """Launch FlashAttention v1 forward pass.

    Args:
        Q: Query tensor, shape (B, H, S, D), fp16.
        K: Key tensor, same shape as Q.
        V: Value tensor, same shape as Q.
        config: Optional config; inferred from Q if not provided.

    Returns:
        O: Output tensor, shape (B, H, S, D), fp16.
    """
    B, H, S, D = Q.shape
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "Inputs must be on CUDA"
    assert Q.dtype == torch.float16, "FlashAttention v1 expects fp16 inputs"

    if config is None:
        config = FlashAttentionConfig(
            batch_size=B,
            num_heads=H,
            seq_len=S,
            head_dim=D,
            dtype=Q.dtype,
        )

    O = torch.empty_like(Q)

    # Grid: (num_query_blocks, B * H)
    num_m_blocks = triton.cdiv(S, config.BLOCK_M)
    grid = (num_m_blocks, B * H)

    _flash_attn_v1_fwd[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        S,
        config.scale,
        BLOCK_M=config.BLOCK_M,
        BLOCK_N=config.BLOCK_N,
        HEAD_DIM=D,
        CAUSAL=config.causal,
    )

    return O
