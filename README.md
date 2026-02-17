# KernelForge 🔥

> **High-performance GPU kernel implementations across CUDA, Triton, and CuTe DSL**

A comprehensive toolkit for implementing, analyzing, and benchmarking GPU kernels with rigorous performance analysis. Documents the complete optimization journey from naive implementations to production-grade kernels.

## What This Is

A research-grade GPU kernel laboratory featuring:

- **Cross-framework implementations** - FlashAttention and transformer primitives in CUDA, Triton, and CuTe DSL
- **Performance archaeology** - Roofline modeling, PTX inspection, memory profiling, automated analysis
- **Megakernel fusion** - Beyond isolated ops: fused transformer blocks, custom gradients
- **Production targets** - PyTorch extensions, llama.cpp plugins, Rust/Candle bindings
- **Hardware evolution** - Hopper/Blackwell features (TMA, wgmma, warp specialization)

## 🚧 Work In Progress

Active development. Core kernels exist but many features are incomplete or planned.

Currently implemented:
- Basic CUDA and Triton attention kernels
- Preliminary benchmark infrastructure
- Early profiling tools

In development:
- Complete kernel suite across all frameworks
- Comprehensive analysis and visualization tools
- Framework integrations

Planned:
- Advanced optimizations (quantization, sparsity, multi-GPU)
- Production deployment guides

## Repository Structure
```
kernelforge/
├── kernels/           # CUDA, Triton, CuTe implementations
├── benchmarks/        # Performance measurement
├── analysis/          # Profiling and visualization
├── integrations/      # PyTorch, Rust, llama.cpp
├── experiments/       # Research explorations
└── docs/              # Guides and references
```

## License

MIT
