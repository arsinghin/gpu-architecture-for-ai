<div align="center">

# The Anatomy of Silicon

### 01 — How GPUs Actually Execute AI Workloads

**From Python code to threads, warps, grids, SMs, memory, and the silicon doing the actual work.**

![Article](https://img.shields.io/badge/Article-Published-brightgreen)
![Lab](https://img.shields.io/badge/Lab-Available-brightgreen)
![Diagrams](https://img.shields.io/badge/Diagrams-In_Progress-yellow)

Part of [GPU Architecture for AI](../README.md) · [Roadmap](../docs/roadmap.md)

</div>

---

## Overview

This article explains what happens inside a GPU when an AI workload executes.

> **What actually happens inside a GPU when you run a neural network?**

Most AI frameworks hide this process. That abstraction is useful for building models. It becomes less useful when you need to understand:

- GPU utilization
- memory behavior
- kernel performance
- latency and throughput
- divergence
- occupancy
- synchronization
- scaling

This article builds the execution model from first principles. The goal is to move from:

```python
output = model(input)
```

to a mental model of:

```text
Python → PyTorch → GPU operation → Kernel → Grid → Thread Blocks → Warps → Threads → SM → Execution + Memory
```

## Table of Contents

- [What You Will Learn](#what-you-will-learn)
- [From PyTorch to GPU Execution](#from-pytorch-to-gpu-execution)
- [Lab 01 — GPU Execution](#lab-01--gpu-execution)
  - [Python Experiments](#python-experiments)
  - [CUDA Experiments](#cuda-experiments)
- [Learning Outcomes](#learning-outcomes)
- [What This Article Does Not Cover](#what-this-article-does-not-cover)
- [Why the Next Article Starts With Memory](#why-the-next-article-starts-with-memory)
- [Diagrams](#diagrams)
- [References](#references)
- [Related Material](#related-material)
- [Next: Article 02](#next-article-02--gpu-memory)

## What You Will Learn

| Concept | Key Idea |
|---------|----------|
| **CPU vs. GPU** | CPUs optimize for low latency and strong single-thread performance; GPUs optimize for throughput through massive parallelism. |
| **SIMD vs. SIMT** | How GPUs execute many threads together — SIMD (Single Instruction, Multiple Data) vs. SIMT (Single Instruction, Multiple Threads). |
| **Threads** | A CUDA thread is a logical unit of execution — not one physical GPU core. Thousands or millions can be launched for a workload. |
| **Thread Blocks** | Threads are grouped into blocks, which can cooperate through shared memory and synchronization. |
| **Grids** | A kernel launch creates a grid of blocks. A grid can contain far more blocks than the GPU can run simultaneously; blocks are scheduled onto available SMs. |
| **Warps** | Threads within a block are grouped into 32-thread warps — the basic unit of the CUDA SIMT execution model. |
| **SMs** | A Streaming Multiprocessor is a major execution structure containing registers, shared memory, caches, and scheduling and execution resources. Organization varies by architecture. |
| **Warp Schedulers** | The SM schedules eligible warp instructions. When one warp waits, another eligible warp executes — keeping execution resources busy. |
| **Registers** | Very fast, limited per-thread storage. High register usage limits how many threads and blocks can stay resident on an SM. |
| **Shared Memory** | On-chip memory available to threads within a block — used for data reuse, cooperation, tiling, and reducing repeated access to slower memory. |
| **L1 and L2 Caches** | Keep frequently accessed data close to execution: registers → shared memory / L1 → L2 → device memory (simplified view; implementation is architecture-specific). |
| **Kernel Execution** | A kernel is a function executed on the GPU. The host launches it with a grid/block configuration; the GPU schedules the resulting blocks. |
| **Latency Hiding** | Multiple warps stay available for execution. When one waits for data, another runs. The GPU does not remove memory latency — it keeps useful work running while some work waits. |
| **Divergence** | When threads in the same warp follow different control-flow paths, the warp becomes less efficient. |
| **Occupancy** | Active warps on an SM relative to the maximum resident warps. Higher occupancy helps hide latency but does not guarantee performance. |

Every concept is paired with a runnable experiment in [Lab 01](#lab-01--gpu-execution).

## From PyTorch to GPU Execution

The article uses PyTorch as the bridge between AI code and GPU execution:

```text
Python
   ↓
PyTorch
   ↓
Tensor Operation
   ↓
CUDA / Optimized GPU Implementation
   ↓
Kernel
   ↓
Grid
   ↓
Thread Blocks
   ↓
Warps
   ↓
SMs
   ↓
Execution + Memory
```

> **Note:** This is a mental model — not a claim that every PyTorch operation maps to exactly one simple CUDA kernel. Frameworks and libraries use optimized implementations, fused operations, generated kernels, and specialized libraries.

## Lab 01 — GPU Execution

Every concept in this article is paired with a runnable experiment in **Lab 01**, located at [`01-gpu-execution/labs/`](labs/). The experiments are numbered and designed to be run in order.

From the repository root:

```bash
python 01-gpu-execution/labs/python/01_device_check.py
```

For environment setup, see [`docs/setup.md`](../docs/setup.md). For full lab details, see the [lab README](labs/README.md).

### Python Experiments

#### 01 · Device Check

**File:** [`python/01_device_check.py`](labs/python/01_device_check.py)

Demonstrates:

- PyTorch CUDA availability
- GPU count
- current CUDA device
- GPU name
- compute capability
- device memory

#### 02 · CPU vs. GPU

**File:** [`python/02_cpu_vs_gpu.py`](labs/python/02_cpu_vs_gpu.py)

Demonstrates:

- CPU tensors and CUDA tensors
- GPU computation
- basic GPU timing
- synchronization

> **Note:** This experiment is educational. It should not be treated as a universal CPU-vs-GPU benchmark.

#### 03 · Asynchronous Timing

**File:** [`python/03_async_timing.py`](labs/python/03_async_timing.py)

Demonstrates why GPU timing requires care, by comparing:

```text
Naive CPU timing
   ↓
Synchronized timing
   ↓
CUDA event timing
```

The main lesson:

> Launching GPU work does not necessarily mean the GPU has finished that work.

### CUDA Experiments

#### 01 · Hello Threads

**File:** [`cuda/01_hello_threads.cu`](labs/cuda/01_hello_threads.cu)

Introduces CUDA kernels, threads, thread blocks, and kernel launches.

#### 02 · Thread Indexing

**File:** [`cuda/02_thread_indexing.cu`](labs/cuda/02_thread_indexing.cu)

Introduces `threadIdx`, `blockIdx`, `blockDim`, and `gridDim`, and the common one-dimensional global index:

```cpp
int global_id = blockIdx.x * blockDim.x + threadIdx.x;
```

#### 03 · Warp Mapping

**File:** [`cuda/03_warp_mapping.cu`](labs/cuda/03_warp_mapping.cu)

Demonstrates how threads map to warps and lanes:

```text
Thread → Warp → Lane
```

On NVIDIA CUDA GPUs: **32 threads = 1 warp.**

#### 04 · Divergence

**File:** [`cuda/04_divergence.cu`](labs/cuda/04_divergence.cu)

Compares a uniform control-flow pattern with a divergent pattern:

```cpp
if (condition)
{
    // path A
}
else
{
    // path B
}
```

When threads in the same warp take different paths, the warp must account for both.

> **Note:** This experiment is intentionally **not** presented as "divergence always causes exactly X% slowdown." Performance depends on the workload and hardware. The experiment exists to make the behavior measurable.

#### 05 · Device Properties

**File:** [`cuda/05_device_properties.cu`](labs/cuda/05_device_properties.cu)

Reports the properties of the GPU running the program — including its execution and memory resources — connecting the abstract programming model to actual hardware.

## Learning Outcomes

After completing this article and its lab, you should be able to mentally trace:

```text
Python
   ↓
PyTorch
   ↓
Kernel
   ↓
Grid
   ↓
Thread Block
   ↓
Warp
   ↓
Thread
   ↓
SM
   ↓
Execution Resources
   ↓
Memory
```

You should also understand three distinctions that correct common intuition:

| Intuition | Reality |
|-----------|---------|
| A CUDA thread is a GPU core | A thread is a logical unit of execution — thousands can be in flight |
| High GPU utilization means good performance | Utilization measures activity, not efficiency |
| High occupancy means high performance | Occupancy helps hide latency; other bottlenecks can still dominate |

## What This Article Does Not Cover

This article deliberately does not attempt to fully explain topics that require their own treatment:

| Area | Topics | Covered In |
|------|--------|------------|
| Memory systems | HBM, GDDR, memory bandwidth, coalescing, tiling, arithmetic intensity, Roofline analysis | Article 02 |
| AI compute | Tensor Cores, FP8, FP4 | Article 03 |
| Multi-GPU and production | multi-GPU communication, NVLink, distributed inference, production serving | Articles 08–10 |

## Why the Next Article Starts With Memory

Once you understand the execution hierarchy:

```text
Thread → Warp → Block → SM
```

the next question becomes:

> **Where does the data come from?**

That leads through the memory hierarchy:

```text
Registers
   ↓
Shared Memory
   ↓
L1
   ↓
L2
   ↓
GPU Device Memory
   ↓
Host Memory
```

and eventually to:

**Latency · Bandwidth · Capacity · Data Reuse · Arithmetic Intensity · Roofline Model**

That is the focus of Article 02.

## Diagrams

Diagrams for this article live in [`diagrams/01-gpu-execution/`](diagrams/) — currently in progress.

| Diagram | Concept |
|---------|---------|
| `01_cpu_vs_gpu` | CPU vs. GPU design tradeoffs |
| `02_gpu_execution_hierarchy` | The full execution hierarchy |
| `03_thread_block_grid` | Threads, blocks, and grids |
| `04_warp_mapping` | Thread-to-warp and lane mapping |
| `05_sm_execution` | SM internals and warp scheduling |
| `06_memory_hierarchy` | Registers through device memory |
| `07_kernel_execution` | Kernel launch and block scheduling |
| `08_pytorch_to_gpu` | PyTorch code to GPU execution |
| `09_latency_hiding` | Warp switching under stalls |
| `10_warp_divergence` | Divergent control flow within a warp |

Diagrams reinforce concepts already explained in the article. They do not introduce unexplained terminology.

## References

The execution model in this article is based primarily on the NVIDIA CUDA Programming Guide, which describes GPUs as collections of SMs, organizes launched threads into thread blocks and grids, and groups block threads into 32-thread warps for SIMT execution.

- NVIDIA Corporation — *CUDA C++ Programming Guide*: <https://docs.nvidia.com/cuda/cuda-programming-guide/>

> For implementation details, always consult the documentation for the specific GPU architecture and CUDA version being used.

## Related Material

| Resource | Location |
|----------|----------|
| Main project | [`README.md`](../README.md) |
| Lab 01 — GPU Execution | [`labs/01-gpu-execution/`](labs/) |
| Setup guide | [`docs/setup.md`](../docs/setup.md) |
| Project roadmap | [`docs/roadmap.md`](../docs/roadmap.md) |
| Glossary | [`docs/glossary.md`](../docs/glossary.md) |

## Next: Article 02 — GPU Memory

**GPU Memory Explained: HBM, SRAM, Cache and the Memory Wall**

Core question:

> **Why can a GPU with enormous compute power still be slow?**

---

<p align="center">
<sub>
<a href="../README.md">Main README</a> ·
<a href="labs/README.md">Lab 01</a> ·
<a href="../docs/roadmap.md">Roadmap</a> ·
<a href="../docs/glossary.md">Glossary</a>
</sub>
</p>
