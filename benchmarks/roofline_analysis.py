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


# ---------------------------------------------------------------------------
# Roofline plotting
# ---------------------------------------------------------------------------

def plot_roofline(
    kernels: List[KernelPoint],
    gpu: Optional[GPUSpec] = None,
    save_path: str = "roofline.png",
    title: Optional[str] = None,
    show: bool = False,
):
    """Plot a roofline model with kernel data points.

    Args:
        kernels: List of measured kernel points to overlay.
        gpu: GPU spec for drawing ceilings. Defaults to H100 SXM.
        save_path: Where to save the PNG. Pass None to skip saving.
        title: Plot title. Auto-generated if None.
        show: If True, call plt.show() interactively.
    """
    if gpu is None:
        gpu = GPU_PRESETS["h100"]

    fig, ax = plt.subplots(figsize=(11, 7))

    # Arithmetic intensity range for ceiling lines
    ai = np.logspace(-2, 4, 500)

    # Memory ceiling: perf = bandwidth * AI
    mem_ceil = gpu.mem_bandwidth_gbs * ai

    # Compute ceilings (horizontal lines)
    fp16_ceil = np.full_like(ai, gpu.peak_fp16_gflops)
    fp32_ceil = np.full_like(ai, gpu.peak_fp32_gflops)

    # Roofline = min(memory ceiling, compute ceiling)
    roof_fp16 = np.minimum(mem_ceil, fp16_ceil)
    roof_fp32 = np.minimum(mem_ceil, fp32_ceil)

    # Draw ceilings
    ax.loglog(ai, roof_fp16, "b-", linewidth=2.0, label=f"FP16 tensor ({gpu.peak_fp16_tflops:.0f} TFLOP/s)")
    ax.loglog(ai, roof_fp32, "r-", linewidth=1.5, label=f"FP32 ({gpu.peak_fp32_tflops:.0f} TFLOP/s)")
    ax.loglog(ai, mem_ceil, "k--", linewidth=1.0, alpha=0.4, label=f"HBM BW ({gpu.mem_bandwidth_tb:.2f} TB/s)")

    # Ridge point annotation
    ridge = gpu.ridge_point_fp16
    ax.axvline(x=ridge, color="blue", linestyle=":", alpha=0.3)
    ax.annotate(
        f"ridge = {ridge:.1f} FLOP/B",
        xy=(ridge, gpu.peak_fp16_gflops * 0.6),
        fontsize=8, color="blue", alpha=0.6,
    )

    # Plot kernel data points
    markers = ["o", "s", "^", "D", "v", "P", "*", "X"]
    colors = plt.cm.tab10.colors
    for i, kp in enumerate(kernels):
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        ax.loglog(
            kp.arithmetic_intensity, kp.achieved_gflops,
            marker=marker, color=color, markersize=10, markeredgecolor="black",
            markeredgewidth=0.5, linestyle="none", label=kp.name, zorder=5,
        )
        # Efficiency annotation
        peak_at_ai = min(gpu.mem_bandwidth_gbs * kp.arithmetic_intensity, gpu.peak_fp16_gflops)
        efficiency = kp.achieved_gflops / peak_at_ai * 100 if peak_at_ai > 0 else 0
        ax.annotate(
            f"  {efficiency:.0f}%",
            xy=(kp.arithmetic_intensity, kp.achieved_gflops),
            fontsize=7, color=color,
        )

    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=12)
    ax.set_ylabel("Performance (GFLOP/s)", fontsize=12)
    ax.set_title(title or f"Roofline Model: {gpu.name}", fontsize=14)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, which="both", alpha=0.2)
    ax.set_xlim(ai[0], ai[-1])
    ax.set_ylim(1, gpu.peak_fp16_gflops * 2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def point_from_benchmark(
    kernel_name: str,
    flops: int,
    bytes_accessed: int,
    avg_ms: float,
) -> KernelPoint:
    """Create a KernelPoint from raw benchmark numbers."""
    ai = flops / bytes_accessed if bytes_accessed > 0 else 0.0
    gflops = (flops / avg_ms) * 1e-6 if avg_ms > 0 else 0.0
    return KernelPoint(name=kernel_name, arithmetic_intensity=ai, achieved_gflops=gflops)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="KernelForge roofline analysis")
    parser.add_argument("--gpu", type=str, default="h100",
                        choices=list(GPU_PRESETS.keys()),
                        help="GPU preset for peak ceilings")
    parser.add_argument("--output", type=str, default="roofline.png",
                        help="Output PNG path")
    parser.add_argument("--show", action="store_true",
                        help="Show interactive plot window")
    parser.add_argument("--demo", action="store_true",
                        help="Plot with example kernel points for demonstration")
    args = parser.parse_args()

    gpu = GPU_PRESETS[args.gpu]

    if args.demo:
        # Example points for illustration
        kernels = [
            KernelPoint("naive_attention_cuda", arithmetic_intensity=5.0, achieved_gflops=120.0),
            KernelPoint("naive_attention_cute", arithmetic_intensity=8.0, achieved_gflops=350.0),
            KernelPoint("flash_attention_v1", arithmetic_intensity=45.0, achieved_gflops=4500.0),
        ]
    else:
        # No real data yet; show empty roofline
        kernels = []
        print("No kernel data provided. Use --demo for example points,")
        print("or integrate with benchmark_suite for real measurements.")

    plot_roofline(kernels, gpu=gpu, save_path=args.output, show=args.show)
    print(f"Roofline saved to {args.output}")


if __name__ == "__main__":
    main()
