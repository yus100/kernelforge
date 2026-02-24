"""
Roofline analysis for attention kernels.

Plots a roofline model given arithmetic intensity and achieved FLOP/s for one
or more kernels. Ships with H100 SXM specs by default but accepts any GPU
config via the GPUSpec dataclass.

Usage:
    python -m benchmarks.roofline_analysis --kernel flash_attention_v1 --seq-len 1024
    python -m benchmarks.roofline_analysis --gpu b200 --from-benchmark results.json
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# GPU specifications
# ---------------------------------------------------------------------------

@dataclass
class GPUSpec:
    """Peak capabilities of a GPU, used to draw roofline ceilings."""

    name: str = "H100 SXM"
    peak_fp16_tflops: float = 989.4   # peak FP16 tensor TFLOP/s
    peak_fp32_tflops: float = 66.9    # peak FP32 TFLOP/s (non-tensor)
    peak_bf16_tflops: float = 989.4   # peak BF16 tensor TFLOP/s
    mem_bandwidth_tb: float = 3.35    # HBM bandwidth in TB/s

    @property
    def peak_fp16_gflops(self) -> float:
        return self.peak_fp16_tflops * 1e3

    @property
    def peak_fp32_gflops(self) -> float:
        return self.peak_fp32_tflops * 1e3

    @property
    def mem_bandwidth_gbs(self) -> float:
        """Memory bandwidth in GB/s."""
        return self.mem_bandwidth_tb * 1e3

    @property
    def ridge_point_fp16(self) -> float:
        """Arithmetic intensity (FLOP/byte) where memory and compute ceilings meet."""
        return self.peak_fp16_gflops / self.mem_bandwidth_gbs


# Pre-configured GPU specs
GPU_PRESETS: Dict[str, GPUSpec] = {
    "h100": GPUSpec(
        name="H100 SXM",
        peak_fp16_tflops=989.4,
        peak_fp32_tflops=66.9,
        peak_bf16_tflops=989.4,
        mem_bandwidth_tb=3.35,
    ),
    "b200": GPUSpec(
        name="B200",
        peak_fp16_tflops=2250.0,
        peak_fp32_tflops=180.0,
        peak_bf16_tflops=2250.0,
        mem_bandwidth_tb=8.0,
    ),
    "a100": GPUSpec(
        name="A100 SXM",
        peak_fp16_tflops=312.0,
        peak_fp32_tflops=19.5,
        peak_bf16_tflops=312.0,
        mem_bandwidth_tb=2.039,
    ),
}


@dataclass
class KernelPoint:
    """A single kernel's measured position on the roofline."""

    name: str
    arithmetic_intensity: float  # FLOP/byte
    achieved_gflops: float
