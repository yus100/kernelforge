# KernelForge

High-performance GPU kernel implementations across CUDA, Triton, CuTe DSL, and Pallas.

A toolkit for implementing, analyzing, and benchmarking GPU kernels with rigorous performance analysis. Documents the complete optimization journey from naive implementations to production-grade kernels.

## Overview

- **Cross-framework kernels** -- FlashAttention and transformer primitives in CUDA, Triton, CuTe DSL, and Pallas
- **Performance analysis** -- Roofline modeling, PTX inspection, memory profiling, automated reporting
- **Megakernel fusion** -- Fused transformer blocks, custom gradients, beyond isolated ops
- **Production targets** -- PyTorch extensions, llama.cpp plugins, Rust/Candle bindings
- **Hardware evolution** -- Hopper/Blackwell features (TMA, wgmma, warp specialization)

## Repository Structure

```
kernelforge/
├── kernels/
│   ├── cuda/              # CUDA C++ kernel implementations
│   ├── triton/            # Triton (Python DSL) kernels
│   ├── cutedsl/           # CuTe DSL kernels
│   └── pallas/            # JAX Pallas kernels
├── benchmarks/            # Performance measurement scripts
├── analysis/              # Profiling, roofline modeling, reporting
├── educational/
│   ├── notebooks/         # Jupyter notebooks for learning
│   └── visualizations/    # Visual explanations of kernel behavior
├── integrations/          # PyTorch, Rust/Candle, llama.cpp bindings
├── experiments/           # Research explorations and prototypes
├── tools/                 # Development utilities (autotuning, etc.)
└── docs/                  # Guides and references
```

## Status

Work in progress. Currently implemented:

- Analysis infrastructure (roofline modeling, metrics, profiling, report generation)
- Autotuning tools for attention kernels

Coming next: kernel implementations across all four frameworks.

## Setup

```bash
pip install -r requirements.txt
```

Optional framework dependencies (install as needed):
- `torch` for PyTorch/Triton kernels
- `jax[cuda]` for Pallas kernels
- CUDA toolkit for raw CUDA kernels

## License

MIT
