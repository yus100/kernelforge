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


# ---------------------------------------------------------------------------
# Kernel registry
# ---------------------------------------------------------------------------

# Maps kernel name -> callable(Q, K, V, **kwargs) -> O
_REGISTRY: Dict[str, Callable] = {}


def register_kernel(name: str, fn: Callable):
    """Register a kernel function under the given name."""
    _REGISTRY[name] = fn


def list_kernels() -> List[str]:
    """Return sorted list of registered kernel names."""
    return sorted(_REGISTRY.keys())


def _populate_registry():
    """Lazily import and register all available kernels."""
    # Triton FlashAttention v1
    try:
        from kernels.triton.flash_attention_v1 import (
            flash_attention_v1,
            FlashAttentionConfig,
        )

        def _triton_flash_v1(Q, K, V, causal=False, **kw):
            cfg = FlashAttentionConfig(
                batch_size=Q.shape[0],
                num_heads=Q.shape[1],
                seq_len=Q.shape[2],
                head_dim=Q.shape[3],
                causal=causal,
            )
            return flash_attention_v1(Q, K, V, cfg)

        register_kernel("flash_attention_v1", _triton_flash_v1)
    except ImportError:
        pass

    # CuTe DSL naive attention
    try:
        from kernels.cutedsl.naive_attention import (
            naive_attention,
            NaiveAttentionConfig,
        )

        def _cute_naive(Q, K, V, causal=False, **kw):
            cfg = NaiveAttentionConfig(
                batch_size=Q.shape[0],
                num_heads=Q.shape[1],
                seq_len=Q.shape[2],
                head_dim=Q.shape[3],
                causal=causal,
            )
            return naive_attention(Q, K, V, cfg)

        register_kernel("naive_attention_cute", _cute_naive)
    except ImportError:
        pass

    # CUDA naive attention (requires pybind/ctypes wrapper, placeholder)
    # Registered as a stub that raises NotImplementedError.
    def _cuda_naive_stub(Q, K, V, **kw):
        raise NotImplementedError(
            "CUDA naive_attention requires compilation. "
            "See kernels/cuda/naive_attention.cu for the source."
        )

    register_kernel("naive_attention_cuda", _cuda_naive_stub)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _attention_flops(B: int, H: int, S: int, D: int, causal: bool) -> int:
    """Approximate FLOPs for attention: Q@K^T (2*S*S*D) + softmax(S*S) + P@V (2*S*S*D)."""
    qk_flops = 2 * S * S * D
    pv_flops = 2 * S * S * D
    softmax_flops = 5 * S * S  # exp, sub, div, sum, max (rough)
    total = B * H * (qk_flops + pv_flops + softmax_flops)
    if causal:
        total //= 2  # roughly half the work
    return total


def _attention_bytes(B: int, H: int, S: int, D: int, dtype: torch.dtype) -> int:
    """Bytes read + written: Q, K, V read + O written, each (B, H, S, D)."""
    elem = torch.tensor([], dtype=dtype).element_size()
    return 4 * B * H * S * D * elem  # 3 reads + 1 write


class BenchmarkRunner:
    """Runs a named kernel with CUDA event timing and reports results."""

    def __init__(self):
        if not _REGISTRY:
            _populate_registry()

    def run(
        self,
        kernel_name: str,
        config: Optional[BenchmarkConfig] = None,
    ) -> BenchmarkResult:
        """Benchmark a single kernel by name."""
        if config is None:
            config = BenchmarkConfig()

        result = BenchmarkResult(kernel_name=kernel_name, config=config)

        if kernel_name not in _REGISTRY:
            result.status = "error"
            result.error_msg = f"Unknown kernel: {kernel_name}. Use list_kernels()."
            return result

        fn = _REGISTRY[kernel_name]
        B, H, S, D = config.batch_size, config.num_heads, config.seq_len, config.head_dim

        # Allocate inputs
        Q = torch.randn(B, H, S, D, dtype=config.dtype, device=config.device)
        K = torch.randn(B, H, S, D, dtype=config.dtype, device=config.device)
        V = torch.randn(B, H, S, D, dtype=config.dtype, device=config.device)

        # Warmup
        try:
            for _ in range(config.warmup_iters):
                fn(Q, K, V, causal=config.causal)
            torch.cuda.synchronize()
        except NotImplementedError as e:
            result.status = "skipped"
            result.error_msg = str(e)
            return result
        except Exception as e:
            result.status = "error"
            result.error_msg = str(e)
            return result

        # Timed iterations using CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        times_ms = []

        for _ in range(config.bench_iters):
            start_event.record()
            fn(Q, K, V, causal=config.causal)
            end_event.record()
            torch.cuda.synchronize()
            times_ms.append(start_event.elapsed_time(end_event))

        result.avg_ms = sum(times_ms) / len(times_ms)
        result.min_ms = min(times_ms)
        result.max_ms = max(times_ms)

        # Throughput
        flops = _attention_flops(B, H, S, D, config.causal)
        nbytes = _attention_bytes(B, H, S, D, config.dtype)
        result.gflops = (flops / result.avg_ms) * 1e-6   # GFLOP/s
        result.gbps = (nbytes / result.avg_ms) * 1e-6     # GB/s

        return result

    def run_all(self, config: Optional[BenchmarkConfig] = None) -> List[BenchmarkResult]:
        """Benchmark every registered kernel."""
        return [self.run(name, config) for name in list_kernels()]


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_result(r: BenchmarkResult):
    """Print a single benchmark result as a formatted line."""
    if r.status != "ok":
        print(f"  {r.kernel_name:30s}  [{r.status}] {r.error_msg}")
        return
    print(
        f"  {r.kernel_name:30s}  "
        f"avg={r.avg_ms:8.3f} ms  "
        f"min={r.min_ms:8.3f} ms  "
        f"max={r.max_ms:8.3f} ms  "
        f"{r.gflops:8.1f} GFLOP/s  "
        f"{r.gbps:8.1f} GB/s"
    )


def print_results(results: List[BenchmarkResult]):
    """Print a table of benchmark results."""
    print("=" * 100)
    print(f"  {'Kernel':30s}  {'avg':>11s}  {'min':>11s}  {'max':>11s}  {'GFLOP/s':>10s}  {'GB/s':>9s}")
    print("-" * 100)
    for r in results:
        print_result(r)
    print("=" * 100)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="KernelForge benchmark suite")
    parser.add_argument("--kernel", type=str, default=None,
                        help="Kernel name to benchmark (omit to run all)")
    parser.add_argument("--list", action="store_true",
                        help="List available kernels and exit")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    runner = BenchmarkRunner()

    if args.list:
        print("Available kernels:")
        for name in list_kernels():
            print(f"  - {name}")
        return

    cfg = BenchmarkConfig(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        causal=args.causal,
        warmup_iters=args.warmup,
        bench_iters=args.iters,
    )

    if args.kernel:
        result = runner.run(args.kernel, cfg)
        print_result(result)
    else:
        results = runner.run_all(cfg)
        print_results(results)


if __name__ == "__main__":
    main()
