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
