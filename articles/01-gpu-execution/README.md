# Article 01 — How GPUs Actually Execute AI Workloads

## The Anatomy of Silicon

**From Python code to threads, warps, grids, SMs, memory, and the silicon doing the actual work.**

---

# Article

**The Anatomy of Silicon: How GPUs Actually Execute AI Workloads**

The article explains what happens inside a GPU when an AI workload executes.

The goal is to move from:

```python
output = model(input)
```

to a mental model of:

```text
Python
  ↓
PyTorch
  ↓
GPU operation
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

---

# Core Question

> **What actually happens inside a GPU when you run a neural network?**

Most AI frameworks hide this process.

That abstraction is useful for building models.

It becomes less useful when you need to understand:

- GPU utilization
- memory behavior
- kernel performance
- latency
- throughput
- divergence
- occupancy
- synchronization
- scaling

This article builds the execution model from first principles.

---

# What You Will Learn

The article covers:

## CPU vs GPU

Why CPUs and GPUs optimize for different goals.

```text
CPU
→ low latency
→ complex control flow
→ strong single-thread performance

GPU
→ high throughput
→ massive parallelism
→ many concurrent threads
```

---

## SIMD vs SIMT

How GPUs execute many threads together.

The important distinction is:

```text
SIMD
Single Instruction
Multiple Data
```

versus:

```text
SIMT
Single Instruction
Multiple Threads
```

---

## Threads

A CUDA thread is a logical unit of execution.

It is not equivalent to:

```text
one physical GPU core
```

Thousands or millions of logical threads can be launched for a workload.

---

## Thread Blocks

Threads are grouped into blocks.

A block provides a unit in which threads can cooperate through mechanisms such as shared memory and synchronization.

---

## Grids

A kernel launch creates a grid containing thread blocks.

A large grid can contain far more blocks than the GPU can execute simultaneously.

The GPU schedules blocks onto available SM resources.

---

## Warps

On NVIDIA CUDA GPUs, threads within a block are grouped into 32-thread warps.

The warp is the basic grouping used by the CUDA SIMT execution model.

---

## SMs

An NVIDIA Streaming Multiprocessor is a major execution structure on the GPU.

It contains resources such as:

- registers
- shared memory
- cache resources
- scheduling hardware
- execution resources

The exact hardware organization changes between GPU architectures.

---

## Warp Schedulers

The SM schedules eligible warp instructions for execution.

When one warp is waiting, another eligible warp can be selected.

This helps the GPU keep execution resources busy.

---

## Registers

Registers provide very fast storage for values used by individual threads.

Registers are limited.

High register usage can affect how many threads or blocks can remain resident on an SM.

---

## Shared Memory

Shared memory is an on-chip memory space available to threads within a thread block.

It is commonly used for:

- data reuse
- cooperation between threads
- tiling
- reducing repeated accesses to slower memory

---

## L1 and L2

Caches help keep frequently accessed data closer to execution resources.

A simplified view is:

```text
Registers
   ↓
Shared Memory / L1
   ↓
L2
   ↓
GPU Device Memory
```

The exact implementation depends on the GPU architecture.

---

## Kernel Execution

A kernel is a function executed on the GPU.

The host launches the kernel with an execution configuration defining the grid and thread-block dimensions.

The GPU then schedules the resulting blocks onto available execution resources.

---

## Latency Hiding

GPUs can keep multiple warps available for execution.

If one warp is waiting for data, another warp may be able to execute.

The GPU does not magically remove memory latency.

It attempts to keep useful work available while some work waits.

---

## Divergence

If threads in the same warp follow different control-flow paths, the warp can become less efficient.

For example:

```cpp
if (condition)
{
    ...
}
else
{
    ...
}
```

If some threads take one path and others take the other path, the execution model must account for both paths.

---

## Occupancy

Occupancy describes how many warps are active on an SM relative to the maximum supported resident warps.

Higher occupancy can help hide latency.

It does not automatically mean better performance.

A kernel can have high occupancy and still be slow because of:

- memory bandwidth
- instruction throughput
- synchronization
- memory access patterns
- other bottlenecks

---

# PyTorch → GPU Execution

The article uses PyTorch as the bridge between AI code and GPU execution.

A simplified view is:

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

This is a mental model, not a claim that every PyTorch operation maps to exactly one simple CUDA kernel.

Frameworks and libraries can use optimized implementations, fused operations, generated kernels, and specialized libraries.

---

# The Article 01 Lab

The corresponding lab is:

```text
labs/01-gpu-execution/
```

It contains Python and CUDA experiments.

---

# Python Experiments

## 01 — Device Check

```text
python/01_device_check.py
```

Demonstrates:

- PyTorch CUDA availability
- GPU count
- current CUDA device
- GPU name
- compute capability
- device memory

---

## 02 — CPU vs GPU

```text
python/02_cpu_vs_gpu.py
```

Demonstrates:

- CPU tensors
- CUDA tensors
- GPU computation
- basic GPU timing
- synchronization

The experiment is educational.

It should not be treated as a universal CPU-vs-GPU benchmark.

---

## 03 — Asynchronous Timing

```text
python/03_async_timing.py
```

Demonstrates why GPU timing requires care.

The experiment compares:

```text
Naive CPU timing
        ↓
Synchronized timing
        ↓
CUDA event timing
```

The main lesson:

> Launching GPU work does not necessarily mean the GPU has already finished that work.

---

# CUDA Experiments

## 01 — Hello Threads

```text
cuda/01_hello_threads.cu
```

Introduces:

- CUDA kernels
- threads
- thread blocks
- kernel launches

---

## 02 — Thread Indexing

```text
cuda/02_thread_indexing.cu
```

Introduces:

```text
threadIdx
blockIdx
blockDim
gridDim
```

and the common one-dimensional global index:

```cpp
int global_id =
    blockIdx.x * blockDim.x + threadIdx.x;
```

---

## 03 — Warp Mapping

```text
cuda/03_warp_mapping.cu
```

Demonstrates:

```text
Thread
   ↓
Warp
   ↓
Lane
```

For NVIDIA CUDA:

```text
32 threads
=
1 warp
```

---

## 04 — Divergence

```text
cuda/04_divergence.cu
```

Compares a uniform control-flow pattern with a divergent pattern.

The experiment is intentionally not presented as:

```text
"Divergence always causes exactly X% slowdown."
```

Performance depends on the workload and hardware.

The experiment exists to make the behavior measurable.

---

## 05 — Device Properties

```text
cuda/05_device_properties.cu
```

Reports properties of the GPU running the program, including relevant execution and memory resources.

This connects the abstract programming model to actual hardware.

---

# Learning Outcome

After completing Article 01 and Lab 01, you should be able to mentally trace:

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

You should also understand that:

```text
Thread ≠ GPU Core
```

and:

```text
GPU Utilization ≠ Automatically Good Performance
```

and:

```text
High Occupancy ≠ Automatically High Performance
```

---

# What This Article Does Not Cover Deeply

Article 01 deliberately does not attempt to fully explain:

- HBM
- GDDR
- memory bandwidth
- memory coalescing
- tiling
- arithmetic intensity
- Roofline analysis
- Tensor Cores
- FP8
- FP4
- multi-GPU communication
- NVLink
- distributed inference
- production serving

Those topics require their own explanations.

They become increasingly important in later articles.

---

# Why the Next Article Starts With Memory

Once you understand:

```text
Thread
  ↓
Warp
  ↓
Block
  ↓
SM
```

the next question becomes:

> Where does the data come from?

That leads to:

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

and eventually:

```text
Latency
Bandwidth
Capacity
Data Reuse
Arithmetic Intensity
Roofline Model
```

That is the focus of Article 02.

---

# Diagrams

Article 01 diagrams are stored in:

```text
diagrams/01-gpu-execution/
```

Planned visual concepts include:

```text
01_cpu_vs_gpu
02_gpu_execution_hierarchy
03_thread_block_grid
04_warp_mapping
05_sm_execution
06_memory_hierarchy
07_kernel_execution
08_pytorch_to_gpu
09_latency_hiding
10_warp_divergence
```

The diagrams should reinforce concepts already explained in the article.

They should not introduce unexplained terminology.

---

# References

The execution model in this article is based primarily on the NVIDIA CUDA Programming Guide.

Official documentation:

https://docs.nvidia.com/cuda/cuda-programming-guide/

The CUDA programming model describes GPUs as collections of SMs, organizes launched threads into thread blocks and grids, and groups block threads into 32-thread warps for SIMT execution.

For implementation details, always consult the documentation for the specific GPU architecture and CUDA version being used.

---

# Related Repository Material

Main project:

```text
../../README.md
```

Lab:

```text
../../labs/01-gpu-execution/README.md
```

Setup:

```text
../../docs/setup.md
```

Roadmap:

```text
../../docs/roadmap.md
```

Glossary:

```text
../../docs/glossary.md
```

---

# Status

```text
Article: Published
Lab: Available
Diagrams: In progress
```

---

# Next

**Article 02 — GPU Memory Explained: HBM, SRAM, Cache and the Memory Wall**

Core question:

> **Why can a GPU with enormous compute power still be slow?**
