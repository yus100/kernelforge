# GPU Architecture Guide

A practical introduction to modern NVIDIA GPU architecture for kernel developers. Covers the H100 (Hopper) and B200 (Blackwell) generations, focusing on the concepts that matter when writing and optimizing kernels.

## Execution Model

A GPU is a massively parallel processor organized around a simple hierarchy:

1. **GPC (Graphics Processing Cluster)** -- a cluster of SMs sharing an interconnect
2. **SM (Streaming Multiprocessor)** -- the fundamental compute unit
3. **Warps** -- groups of 32 threads that execute in lockstep (SIMT)

The H100 has 132 SMs (144 on the full die, some disabled for yield). The B200 scales to 192 SMs. Each SM can schedule multiple warps concurrently, hiding memory latency through occupancy rather than caching.

When you launch a CUDA kernel, the runtime distributes thread blocks across available SMs. Each SM can host multiple thread blocks simultaneously, limited by register usage, shared memory allocation, and thread count. This is "occupancy," and it directly affects how well the GPU can hide latency.

### Warp Execution

Every instruction is issued per-warp (32 threads). When a warp stalls on a memory access, the SM switches to another ready warp at zero cost. This is the GPU's primary latency-hiding mechanism. High occupancy means more warps available to fill stall cycles.

Hopper introduced **warp specialization** as a first-class pattern: within a thread block, different warp groups can take on different roles (producer/consumer). One group loads data via TMA while another computes on previously loaded data, overlapping memory and compute at the warp-group level.

## Memory Hierarchy

This is where most kernel optimization effort goes. The memory system has four levels, each with dramatically different bandwidth and capacity.

### HBM (High Bandwidth Memory)

The main device memory. Large but relatively slow compared to on-chip resources.

| GPU  | Capacity | Bandwidth     |
|------|----------|---------------|
| H100 SXM | 80 GB   | 3.35 TB/s     |
| B200     | 192 GB  | 8.0 TB/s      |

HBM is where your tensors live. Every global memory access goes here (unless cached). The key optimization: minimize trips to HBM by maximizing data reuse in faster memories. This is the core idea behind kernel fusion and tiling.

### L2 Cache

A device-wide cache sitting between the SMs and HBM.

| GPU  | Size   |
|------|--------|
| H100 | 50 MB  |
| B200 | 128 MB |

L2 is hardware-managed and transparent. You don't explicitly load into it, but you can influence its effectiveness through access patterns. Sequential, coalesced accesses get good L2 hit rates. Random access patterns thrash it.

On Hopper, you can use `SetMaxNSmPolicy` to partition the L2 among different data streams, preventing one kernel's working set from evicting another's.

### Shared Memory (SMEM)

Per-SM on-chip memory, explicitly managed by the programmer. This is the workhorse of kernel optimization.

| GPU  | Per SM  |
|------|---------|
| H100 | 228 KB  |
| B200 | 228 KB  |

Shared memory is banked (32 banks, 4 bytes each). Threads in a warp accessing different banks get served simultaneously. Threads hitting the same bank serialize (bank conflict), unless they all read the same address (broadcast).

Typical usage pattern:
1. Cooperatively load a tile from global memory into shared memory
2. Synchronize the thread block (`__syncthreads()`)
3. Compute on the tile with fast shared memory reads
4. Repeat for the next tile

This "tiling" pattern is fundamental to nearly every high-performance kernel. It converts expensive global memory traffic into cheap shared memory accesses by exploiting data reuse within a tile.

Shared memory and L1 cache share the same physical SRAM on each SM. On Hopper, the split is configurable: you can allocate more to shared memory (up to 228 KB) at the cost of less L1 cache, or vice versa. Compute-heavy kernels with explicit tiling typically want maximum shared memory.

### Registers

The fastest storage, private to each thread.

| GPU  | Per SM     | Per Thread (max) |
|------|------------|------------------|
| H100 | 256 KB     | 255 registers    |
| B200 | 256 KB     | 255 registers    |

Register pressure is a key tuning knob. Using too many registers per thread reduces occupancy (fewer warps can fit on the SM). Using too few means spilling to local memory (which is actually slow global memory). The sweet spot depends on the kernel.

You can control this with `__launch_bounds__` or compiler flags like `maxrregcount`. Profiling tools (Nsight Compute) report register usage and its effect on occupancy.

### Summary Table

| Level     | Scope       | Capacity (H100) | Bandwidth (approx) | Managed By    |
|-----------|-------------|------------------|---------------------|---------------|
| Registers | Per thread  | 255 regs         | ~20 TB/s            | Compiler      |
| SMEM      | Per SM      | 228 KB           | ~10-15 TB/s         | Programmer    |
| L2        | Device-wide | 50 MB            | ~5-6 TB/s           | Hardware      |
| HBM       | Device-wide | 80 GB            | 3.35 TB/s           | Programmer    |

The bandwidth gap between registers and HBM is roughly 6x. Every level of the hierarchy you can serve data from saves significant time. This is why tiling and fusion matter so much.

## Tensor Cores

Tensor cores are specialized matrix multiply-accumulate (MMA) units embedded in each SM. They operate on small matrix tiles and deliver throughput far beyond what regular CUDA cores can achieve.

### How They Work

A tensor core computes `D = A * B + C` on small matrix fragments. The supported tile shapes and data types vary by generation:

| GPU  | Key Precisions              | Peak (dense)   |
|------|-----------------------------|----------------|
| H100 | FP16, BF16, FP8, INT8, TF32 | ~990 TFLOPS (FP16) |
| B200 | FP16, BF16, FP8, FP4, INT8  | ~2.25 PFLOPS (FP4) |

To use tensor cores, you don't write scalar multiply-adds. Instead, you use warp-level matrix operations that feed data to the tensor core units. There are three main interfaces, from low to high level:

1. **WMMA (Warp Matrix Multiply Accumulate)** -- the original PTX/CUDA API. Portable but limited tile sizes.
2. **MMA PTX instructions** -- lower-level, more tile size options, more control.
3. **wgmma (Warp Group MMA)** -- Hopper-specific. Operates at warp-group granularity (128 threads) and can source operands directly from shared memory, reducing register pressure.

For new Hopper/Blackwell kernels, wgmma is the preferred path. It's what cuBLAS and FlashAttention use internally.

### Why This Matters for Kernels

The arithmetic intensity required to keep tensor cores busy is high. If your kernel spends most of its time loading data from HBM, tensor cores will sit idle. This is why memory optimization (tiling, fusion, async loads) is a prerequisite to getting good tensor core utilization. The kernel must be compute-bound, not memory-bound, to benefit from tensor cores.

## TMA (Tensor Memory Accelerator)

Introduced with Hopper, TMA is a hardware unit that handles multi-dimensional memory copies between global memory and shared memory. Before TMA, loading a 2D tile from a larger tensor required address calculations and boundary checks in every thread. TMA offloads all of this to dedicated hardware.

### What TMA Does

- Copies N-dimensional tiles (up to 5D) between global and shared memory
- Handles out-of-bounds clamping or zero-fill automatically
- Performs address calculation in hardware (no thread-side pointer arithmetic)
- Supports async execution, overlapping loads with compute

### Why It Matters

A single thread can issue a TMA descriptor and the hardware loads the entire tile asynchronously. The remaining threads in the warp group are free to compute on previously loaded data. This is the foundation of the producer-consumer warp specialization pattern on Hopper.

Without TMA, loading a 2D tile requires: computing row pointers, handling boundary conditions, issuing per-thread loads, and synchronizing. TMA replaces all of this with a single instruction plus a descriptor setup.

In practice, TMA is used through:
- CuTe's `copy` abstractions (recommended for new code)
- Raw PTX (`cp.async.bulk.tensor`)
- CUTLASS's TMA-based copy operations

## Putting It Together

A modern high-performance kernel on Hopper typically combines all of these elements:

1. **TMA** loads tiles from HBM into shared memory asynchronously
2. **Warp specialization** overlaps loading (producer warps) with compute (consumer warps)
3. **wgmma** performs matrix operations on shared memory tiles using tensor cores
4. **Software pipelining** keeps multiple tiles in flight, hiding latency across loop iterations
5. **Register tiling** accumulates partial results in registers before writing back

This is the pattern behind FlashAttention-3, CUTLASS 3.x GEMMs, and other state-of-the-art kernels. Each component addresses a different bottleneck, and they compose to approach peak hardware utilization.

## Blackwell (B200) Additions

Blackwell builds on Hopper's architecture with several enhancements:

- **2x SM count** (192 vs 132), increasing raw parallelism
- **Larger L2 cache** (128 MB vs 50 MB), improving hit rates for larger working sets
- **FP4 tensor cores**, enabling 4-bit inference at extreme throughput
- **5th-gen NVLink** with higher inter-GPU bandwidth for multi-GPU workloads
- **Enhanced TMA** with additional addressing modes

The programming model is largely the same. Kernels written for Hopper (using TMA, wgmma, warp specialization) carry forward to Blackwell with minimal changes, benefiting from the increased resources automatically.

## Further Reading

- NVIDIA H100 Whitepaper (architecture details, SM diagrams)
- NVIDIA Blackwell Architecture Whitepaper
- CUTLASS documentation (CuTe layout algebra, TMA usage patterns)
- Nsight Compute documentation (profiling methodology, roofline analysis)
- "FlashAttention-2" and "FlashAttention-3" papers (practical examples of these techniques)
