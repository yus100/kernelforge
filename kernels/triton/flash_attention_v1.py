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
