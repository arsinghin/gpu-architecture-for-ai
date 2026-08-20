<div align="center">

# Lab 01 — GPU Execution

**Small, runnable programs that make the GPU execution model directly observable.**

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?logo=nvidia&logoColor=white)

The article explains the concepts. This lab lets you **run them**.

Accompanies [01 — How GPUs Actually Execute AI Workloads](../README.md)

Part of [GPU Architecture for AI](../../README.md) · [Roadmap](../../docs/roadmap.md)

</div>

---

## Overview

Instead of only reading about threads, blocks, warps, SMs, kernels, divergence, and asynchronous execution, this lab writes small programs that expose each idea directly.

Each experiment is deliberately minimal — just enough to make one concept observable.

## Table of Contents

- [What You Will Learn](#what-you-will-learn)
- [Requirements](#requirements)
- [Repository Structure](#repository-structure)
- [Run Order](#run-order)
- [Experiments](#experiments)
  - [Python Experiments](#python-experiments)
    - [Experiment 1 — Find the GPU](#experiment-1--find-the-gpu)
    - [Experiment 2 — CPU vs. GPU](#experiment-2--cpu-vs-gpu)
    - [Experiment 3 — Why GPU Timing Is Different](#experiment-3--why-gpu-timing-is-different)
  - [CUDA Experiments](#cuda-experiments)
    - [Experiment 4 — Hello, Threads](#experiment-4--hello-threads)
    - [Experiment 5 — Thread Indexing](#experiment-5--thread-indexing)
    - [Experiment 6 — Threads Become Warps](#experiment-6--threads-become-warps)
    - [Experiment 7 — Warp Divergence](#experiment-7--warp-divergence)
    - [Experiment 8 — Inspect the GPU](#experiment-8--inspect-the-gpu)
- [The Mental Model](#the-mental-model)
- [Important Limitations](#important-limitations)
- [What Comes Next](#what-comes-next)
- [References](#references)
- [License](#license)

## What You Will Learn

By the end of this lab, you should be able to explain the execution path:

```text
Python
   ↓
PyTorch / CUDA
   ↓
Kernel
   ↓
Grid
   ↓
Thread Blocks
   ↓
Warps
   ↓
Threads
   ↓
SM
   ↓
Execution + Memory
```

You will also see why:

- a GPU thread is not a physical GPU core
- threads are organized into blocks, and blocks form a grid
- NVIDIA GPUs execute threads in groups called warps
- block and thread indices identify the work assigned to a thread
- different threads in a warp can follow different control-flow paths
- GPU work is often asynchronous from the CPU's point of view
- GPU timing must be done carefully
- the same mathematical operation can behave differently depending on how it is executed

## Requirements

- **Python:** Python 3.9+ and PyTorch — install via the [official selector](https://pytorch.org/get-started/locally/)
- **CUDA:** an NVIDIA GPU, a working NVIDIA driver, and the CUDA Toolkit (`nvcc`)
- The CUDA Toolkit version and the CUDA runtime used by PyTorch do not have to be identical — what matters is that the environment is compatible and the programs compile and run successfully

Full verification commands and troubleshooting: [`docs/setup.md`](../../docs/setup.md).

## Repository Structure

```text
01-gpu-execution/labs/
│
├── README.md
│
├── python/
│   ├── 01_device_check.py
│   ├── 02_cpu_vs_gpu.py
│   └── 03_async_timing.py
│
└── cuda/
    ├── 01_hello_threads.cu
    ├── 02_thread_indexing.cu
    ├── 03_warp_mapping.cu
    ├── 04_divergence.cu
    └── 05_device_properties.cu
```

The experiments intentionally start very small. **Do not skip ahead.** The point is to build the execution model one layer at a time.

## Run Order

All commands in this README assume your working directory is `01-gpu-execution/labs/`:

```bash
cd 01-gpu-execution/labs
```

Run the experiments in order. The last column shows the concept(s) each one makes observable — after finishing, you should be able to connect every concept to something you actually ran:

| # | Experiment | File | Concepts Made Observable |
|:--:|------------|------|--------------------------|
| 1 | Find the GPU | [`python/01_device_check.py`](python/01_device_check.py) | CUDA device |
| 2 | CPU vs. GPU | [`python/02_cpu_vs_gpu.py`](python/02_cpu_vs_gpu.py) | CPU tensor, CUDA tensor, GPU computation |
| 3 | Asynchronous Timing | [`python/03_async_timing.py`](python/03_async_timing.py) | Synchronization, asynchronous execution |
| 4 | Hello, Threads | [`cuda/01_hello_threads.cu`](cuda/01_hello_threads.cu) | Kernel, thread |
| 5 | Thread Indexing | [`cuda/02_thread_indexing.cu`](cuda/02_thread_indexing.cu) | Block, grid, thread indexing |
| 6 | Warp Mapping | [`cuda/03_warp_mapping.cu`](cuda/03_warp_mapping.cu) | Warp, lane |
| 7 | Warp Divergence | [`cuda/04_divergence.cu`](cuda/04_divergence.cu) | Divergence |
| 8 | Device Properties | [`cuda/05_device_properties.cu`](cuda/05_device_properties.cu) | GPU hardware properties |

Do not worry about memorizing everything. The goal is to make the hierarchy feel obvious.

## Experiments

### Python Experiments

#### Experiment 1 — Find the GPU

**File:** [`python/01_device_check.py`](python/01_device_check.py)

```bash
python python/01_device_check.py
```

This experiment answers:

> **Does PyTorch see a CUDA-capable GPU — and which GPU is it?**

Expected output will look similar to:

```text
PyTorch version: 2.x.x
CUDA available: True
CUDA device count: 1
Current device: 0
GPU name: NVIDIA ...
```

The exact GPU name depends on your hardware.

#### Experiment 2 — CPU vs. GPU

**File:** [`python/02_cpu_vs_gpu.py`](python/02_cpu_vs_gpu.py)

```bash
python python/02_cpu_vs_gpu.py
```

This experiment performs the same element-wise operation on CPU and GPU tensors. It demonstrates:

- CPU tensors and CUDA tensors
- moving data to the GPU
- GPU computation
- synchronization
- basic timing

> **Note:** Do not treat the result as a universal CPU-vs-GPU benchmark. The goal is to understand **where the operation executes**, not to declare one processor universally faster. Small workloads can be dominated by launch and transfer overhead.

#### Experiment 3 — Why GPU Timing Is Different

**File:** [`python/03_async_timing.py`](python/03_async_timing.py)

```bash
python python/03_async_timing.py
```

This experiment compares:

- ordinary CPU timing
- synchronized GPU timing
- CUDA event timing

The important lesson:

> **Launching GPU work is not the same thing as waiting for GPU work to finish.**

CUDA operations can execute asynchronously from the CPU's point of view, which means a naive timer can measure the wrong thing. For serious GPU benchmarking, use CUDA-aware timing and understand synchronization.

### CUDA Experiments

#### Experiment 4 — Hello, Threads

**File:** [`cuda/01_hello_threads.cu`](cuda/01_hello_threads.cu)

```bash
nvcc -O2 cuda/01_hello_threads.cu -o hello_threads
./hello_threads
```

This is the smallest CUDA experiment in the lab. It launches multiple GPU threads and lets each thread identify itself:

```text
Hello from block 0, thread 0
Hello from block 0, thread 1
Hello from block 0, thread 2
...
```

The exact output order may vary. That is intentional.

> Do not assume that GPU threads print in numerical order. The GPU is executing parallel work, not politely reading your list from top to bottom.

#### Experiment 5 — Thread Indexing

**File:** [`cuda/02_thread_indexing.cu`](cuda/02_thread_indexing.cu)

```bash
nvcc -O2 cuda/02_thread_indexing.cu -o thread_indexing
./thread_indexing
```

This experiment demonstrates `threadIdx`, `blockIdx`, `blockDim`, and `gridDim`, and the common global-thread-index calculation:

```cpp
int global_id = blockIdx.x * blockDim.x + threadIdx.x;
```

For example, with **4 blocks × 8 threads per block**:

```text
Block 0: threads 0–7  → global IDs 0–7
Block 1: threads 0–7  → global IDs 8–15
Block 2: threads 0–7  → global IDs 16–23
Block 3: threads 0–7  → global IDs 24–31
```

This is the basic mapping used to divide data and work among GPU threads.

#### Experiment 6 — Threads Become Warps

**File:** [`cuda/03_warp_mapping.cu`](cuda/03_warp_mapping.cu)

```bash
nvcc -O2 cuda/03_warp_mapping.cu -o warp_mapping
./warp_mapping
```

This experiment shows the mapping:

```text
Global Thread
   ↓
Warp ID
   ↓
Lane ID
```

For NVIDIA CUDA GPUs, the warp size is 32:

```text
Thread 0  → Warp 0, Lane 0
Thread 1  → Warp 0, Lane 1
...
Thread 31 → Warp 0, Lane 31
Thread 32 → Warp 1, Lane 0
Thread 33 → Warp 1, Lane 1
```

This is why block sizes that are multiples of 32 are commonly useful. The CUDA programming model groups threads into 32-thread warps for SIMT execution.

#### Experiment 7 — Warp Divergence

**File:** [`cuda/04_divergence.cu`](cuda/04_divergence.cu)

```bash
nvcc -O2 cuda/04_divergence.cu -o divergence
./divergence
```

This experiment compares two kernels:

| Kernel | Control Flow |
|--------|--------------|
| **A** | All threads follow the same path |
| **B** | Threads follow different branches |

The goal is **not** to produce a universal percentage such as:

> "Divergence makes your GPU exactly 2× slower."

That would be nonsense. The cost depends on the workload, architecture, compiler, memory behavior, and branch structure.

Instead, the experiment lets you observe how different control-flow patterns can affect execution. NVIDIA's CUDA programming model explains that when threads in a warp take different control-flow paths, the paths can be executed with inactive threads masked for the path they are not taking.

#### Experiment 8 — Inspect the GPU

**File:** [`cuda/05_device_properties.cu`](cuda/05_device_properties.cu)

```bash
nvcc -O2 cuda/05_device_properties.cu -o device_properties
./device_properties
```

The program reports useful hardware information, such as:

- GPU name
- compute capability
- SM count
- warp size
- maximum threads per block
- shared memory per block
- registers per block
- global memory

The exact values depend on your GPU. This experiment connects the abstract execution model to the actual hardware sitting inside your machine.

## The Mental Model

After finishing the lab, you should be able to look at a CUDA kernel and mentally trace where it executes — the path from [What You Will Learn](#what-you-will-learn), extended one level deeper with the memory resources each thread consumes:

```text
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
Registers / Shared Memory / Cache
   ↓
Execution Resources
```

When looking at a PyTorch operation, trace the same path from the top — starting at Python, passing through the framework's optimized implementations, and arriving at the same kernel.

## Important Limitations

These experiments are deliberately simplified. They are designed to build a mental model, not to reproduce every detail of a modern NVIDIA GPU:

- GPU architecture differs between generations.
- Not every PyTorch operation maps directly to one CUDA kernel. PyTorch may use optimized libraries, generated kernels, fused operations, or other implementations.
- Warp behavior is described using the CUDA programming model; hardware implementation details can differ.
- Timing results depend on the GPU, driver, CUDA version, workload size, clocks, and system state.
- CPU-vs-GPU results from these tiny experiments should not be treated as production benchmarks.
- Occupancy is not the same thing as performance.
- Divergence does not have one fixed performance penalty.
- CUDA-specific experiments require an NVIDIA GPU and CUDA environment.

The purpose of this lab is **understanding**, followed by measurement. The serious performance experiments come later.

## What Comes Next

This lab answers:

> **How does the GPU execute work?**

The next problem is:

> **Where does the data come from — and why can a powerful GPU still spend its time waiting?**

That is the subject of [Article 02 — GPU Memory](../../docs/roadmap.md#part-02--gpu-memory) (planned) and its future lab, where we will measure:

- memory bandwidth
- coalesced access
- strided access
- cache behavior
- shared-memory reuse
- tiling
- arithmetic intensity
- memory-bound vs. compute-bound behavior

The next lab focuses much more heavily on measurement.

## References

- NVIDIA — *CUDA C++ Programming Guide*: <https://docs.nvidia.com/cuda/cuda-programming-guide/>
- NVIDIA — *CUDA Samples*: <https://github.com/NVIDIA/cuda-samples>
- PyTorch — *CUDA Documentation*: <https://docs.pytorch.org/docs/stable/cuda.html>
- PyTorch — *CUDA Semantics*: <https://docs.pytorch.org/docs/stable/notes/cuda.html>

## License

This lab is part of the [GPU Architecture for AI](../../README.md) project and is released under the [MIT License](../../LICENSE).

---

<p align="center">
<sub>
<a href="../../README.md">Main README</a> ·
<a href="../README.md">Article 01</a> ·
<a href="../../docs/setup.md">Setup</a> ·
<a href="../../docs/roadmap.md">Roadmap</a> ·
<a href="../../docs/glossary.md">Glossary</a>
</sub>
</p>
