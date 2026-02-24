"""
Unified benchmark harness for KernelForge.

Loads a kernel by name, runs it with configurable problem sizes, and reports
wall-clock time plus effective throughput (GFLOP/s and GB/s). Supports all
backend types (Triton, CuTe DSL, CUDA) through a simple registry.

Usage:
    python -m benchmarks.benchmark_suite --kernel flash_attention_v1 --seq-len 1024
    python -m benchmarks.benchmark_suite --kernel naive_attention_cute --causal
    python -m benchmarks.benchmark_suite --list
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import time

import torch


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """Problem size and run parameters for a single benchmark."""

    batch_size: int = 2
    num_heads: int = 8
    seq_len: int = 1024
    head_dim: int = 64
    causal: bool = False
    dtype: torch.dtype = torch.float16
    warmup_iters: int = 10
    bench_iters: int = 100
    device: str = "cuda"


@dataclass
class BenchmarkResult:
    """Timing and throughput numbers from a single benchmark run."""

    kernel_name: str
    config: BenchmarkConfig
    avg_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    gflops: float = 0.0    # effective GFLOP/s
    gbps: float = 0.0      # effective memory bandwidth GB/s
    status: str = "ok"     # "ok", "error", "skipped"
    error_msg: str = ""
