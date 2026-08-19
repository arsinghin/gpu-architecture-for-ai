# GPU Architecture for AI — Glossary

> Short explanations of the GPU, CUDA, AI, memory, and performance terms used throughout this project.

This glossary is a quick reference.

It is not intended to replace the articles or official documentation.

For deeper explanations, follow the relevant article and lab.

---

# A

## AI Accelerator

Hardware designed to perform workloads commonly used in artificial intelligence efficiently.

Modern GPUs are widely used as AI accelerators because they provide large amounts of parallel compute and specialized hardware for operations such as matrix multiplication.

## Arithmetic Intensity

The amount of computation performed relative to the amount of data moved from memory.

A common simplified definition is:

```text
Arithmetic Intensity =
Operations performed
--------------------
Bytes moved
````

Higher arithmetic intensity means more computation is performed for each byte moved.

It is an important concept in the Roofline Model.

## Asynchronous Execution

Execution where the CPU or a GPU thread can initiate an operation without necessarily waiting for that operation to finish immediately.

CUDA uses asynchronous execution extensively.

This is one reason naive CPU timing can produce incorrect GPU performance measurements.

---

# B

## Bandwidth

The amount of data that can be transferred per unit of time.

For GPU memory, bandwidth is commonly expressed in:

```text
GB/s
TB/s
```

High bandwidth allows the GPU to move large amounts of data quickly.

## Block

Short name for a thread block.

A block is a group of CUDA threads that can cooperate through mechanisms such as shared memory and synchronization.

All threads in a CUDA thread block execute on a single SM.

## Block Dimension

The number of threads in each dimension of a thread block.

CUDA exposes this through:

```text
blockDim.x
blockDim.y
blockDim.z
```

---

# C

## Cache

Small, fast memory used to keep recently or frequently accessed data closer to computation.

Caches are generally managed by hardware.

This differs from shared memory, which is explicitly managed by the programmer.

## CUDA

NVIDIA's parallel computing platform and programming model.

CUDA provides APIs, libraries, tools, and programming interfaces for using NVIDIA GPUs for general-purpose computation.

## CUDA Core

A commonly used NVIDIA marketing term for a general arithmetic execution unit.

The term should not be treated as a direct equivalent of a CPU core.

The exact organization and capabilities of execution resources differ across GPU architectures.

## CUDA Event

A CUDA timing and synchronization mechanism.

CUDA events are commonly used to measure elapsed time on the GPU timeline.

## CUDA Kernel

A function launched for execution on a CUDA-capable NVIDIA GPU.

A kernel is executed by many GPU threads according to the launch configuration.

## CUDA Toolkit

The software development toolkit used to build CUDA applications.

It includes tools such as:

```text
nvcc
```

as well as libraries, headers, development tools, and other components.

## Compute Capability

A CUDA architecture identifier used to describe the capabilities supported by a GPU.

It is commonly represented as:

```text
major.minor
```

For example:

```text
8.0
9.0
```

The exact capabilities depend on the GPU architecture.

## Compute-Bound

A workload is compute-bound when computation throughput is the main limiting factor.

Adding more memory bandwidth may not significantly improve performance if the workload is already limited by arithmetic execution.

## Compute Unit

A major AMD GPU execution structure.

It is conceptually related to NVIDIA's SM but should not be treated as an identical structure.

---

# D

## Data Parallelism

A programming approach where the same operation is applied independently to many pieces of data.

Example:

```text
output[i] = input[i] × 2
```

can be applied independently to many elements.

GPU programming makes extensive use of data parallelism.

## Device

In CUDA terminology, the GPU is generally referred to as the device.

The CPU is generally referred to as the host.

## Device Memory

Memory associated with the GPU.

For discrete GPUs, this commonly refers to memory such as GDDR or HBM.

## Divergence

A situation where threads within a warp follow different control-flow paths.

For example:

```cpp
if (x > 0)
    ...
else
    ...
```

If different threads take different branches, the warp may need to execute multiple paths with different lanes active.

This can reduce execution efficiency.

---

# E

## Execution Unit

A hardware resource capable of performing instructions or specialized operations.

The exact types and organization of execution units vary between GPU architectures.

## Effective Bandwidth

The amount of useful data transferred by a workload divided by its execution time.

It is commonly used when evaluating memory-bound kernels.

It is different from the GPU's theoretical peak memory bandwidth.

---

# F

## FP32

32-bit floating-point representation.

FP32 is widely used in numerical computing and has historically been a common precision for GPU workloads.

## FP16

16-bit floating-point representation.

FP16 uses less storage and can enable higher throughput on hardware that supports efficient FP16 computation.

## FP8

An 8-bit floating-point format.

Modern AI accelerators can support FP8 formats for selected workloads.

Using lower precision can increase throughput and reduce memory requirements, but numerical behavior must be considered.

## FP4

A very low-precision numerical format used by some modern AI hardware for selected workloads.

Its usefulness depends on hardware support, software support, and the numerical requirements of the model.

---

# G

## GDDR

Graphics Double Data Rate memory.

GDDR is a type of high-speed DRAM commonly used as GPU device memory in many graphics and compute GPUs.

## Global Memory

CUDA's general-purpose device memory space accessible by GPU threads.

It is much larger than registers or shared memory but typically has higher access latency.

## Global Thread ID

A calculated identifier that uniquely maps a CUDA thread to a piece of work.

A common one-dimensional form is:

```cpp
int i =
    blockIdx.x * blockDim.x
    + threadIdx.x;
```

## Grid

A collection of CUDA thread blocks launched for a kernel.

A grid can contain many more blocks than can execute simultaneously on the GPU.

## Grid Dimension

The dimensions of a CUDA grid.

CUDA exposes:

```text
gridDim.x
gridDim.y
gridDim.z
```

---

# H

## HBM

High Bandwidth Memory.

HBM is a high-bandwidth memory technology used by many data-center and AI accelerators.

HBM provides high memory bandwidth and is commonly used where large amounts of data must be supplied to GPU compute resources.

## Host

In CUDA terminology, the CPU and the system environment running the CPU-side program are generally referred to as the host.

## Host Memory

Memory associated with the CPU/system.

For a typical discrete GPU system, moving data between host memory and GPU device memory involves a communication path such as PCIe.

---

# I

## Inference

Using a trained AI model to produce predictions or outputs.

For an LLM, inference includes generating tokens in response to an input.

## Instruction

An operation executed by a processor or execution unit.

GPU performance depends on factors including instruction throughput, dependencies, memory access, and available execution resources.

## Instruction-Level Parallelism

The ability to execute multiple independent instructions from a program without waiting for each instruction to finish sequentially.

GPUs exploit multiple forms of parallelism, including instruction-level and thread-level parallelism.

---

# K

## Kernel

A function designed to execute on the GPU.

A kernel launch creates a grid of threads organized into thread blocks.

Example:

```cpp
__global__ void add(float* a, float* b, float* c)
{
    ...
}
```

## Kernel Launch

The operation that starts execution of a CUDA kernel on the GPU.

A launch specifies the execution configuration, including the grid and block dimensions.

---

# L

## L1 Cache

A small cache located close to an SM.

L1 caches are designed to reduce the need to access slower levels of the memory hierarchy.

The exact organization and behavior depend on the GPU architecture.

## L2 Cache

A larger cache shared across multiple SMs on NVIDIA GPUs.

It sits farther from individual execution resources than L1 but closer than device memory.

## Lane

The position of a thread within a warp.

For a 32-thread NVIDIA warp:

```text
Lane 0
Lane 1
...
Lane 31
```

## Latency

The time required for an operation to produce a result or for data to become available.

Latency is different from bandwidth.

A memory system can have:

```text
high bandwidth
```

while individual accesses still experience significant latency.

## Latency Hiding

A GPU performance technique in which the processor switches among available work while some work is waiting.

For example:

```text
Warp 0 → waiting for memory
Warp 1 → execute
Warp 2 → execute
Warp 3 → execute
```

The GPU does not make the memory access instant.

It keeps working on other available work.

## Local Memory

A CUDA memory space associated with individual threads.

Despite its name, local memory is not necessarily physically located in a small on-chip memory structure like registers.

Register spilling can cause values to be placed in local memory.

---

# M

## Memory-Bound

A workload is memory-bound when memory movement or memory throughput is the main performance limitation.

Adding more arithmetic execution capacity may not help much if the GPU cannot supply data quickly enough.

## Memory Coalescing

An access pattern where threads in a warp access memory in a way that allows the hardware to combine their requests efficiently.

Coalesced access is especially important for global-memory performance.

## Memory Hierarchy

The collection of memory and storage levels available to a GPU.

A simplified model is:

```text
Registers
    ↓
Shared Memory / L1
    ↓
L2
    ↓
GPU Device Memory
```

Different levels trade capacity, latency, bandwidth, and accessibility.

## Memory Wall

The performance problem created when computation becomes much faster than the system can supply the required data.

A GPU may have enormous arithmetic throughput but still spend time waiting for data.

## Memory Bandwidth

The rate at which memory can transfer data.

Usually expressed in:

```text
GB/s
TB/s
```

## Memory Capacity

The amount of data a memory system can hold.

For GPU device memory, capacity is commonly expressed in:

```text
GB
GiB
```

## Memory Latency

The time between requesting data and receiving usable data.

High bandwidth does not automatically mean low latency.

---

# N

## NVLink

NVIDIA's high-speed interconnect technology for communication between GPUs and, depending on the system and generation, other components.

It is designed to provide higher-bandwidth GPU communication than relying only on conventional PCIe paths.

## NVSwitch

NVIDIA switching technology designed to connect multiple GPUs through high-bandwidth GPU interconnects.

It is used in some large multi-GPU systems.

---

# O

## Occupancy

The ratio of active warps on an SM to the maximum number of active warps that the SM can support.

Higher occupancy can provide more available warps for latency hiding.

However:

> High occupancy does not automatically mean high performance.

Occupancy is a useful diagnostic, not a universal performance score.

---

# P

## PCIe

Peripheral Component Interconnect Express.

A common high-speed interconnect used to connect CPUs, GPUs, storage devices, and other components.

In GPU systems, PCIe can provide the communication path between a CPU/system and a discrete GPU.

## Prefill

The phase of LLM inference in which the model processes the input prompt before generating new tokens.

Prefill often involves substantial parallel computation.

## PyTorch

An open-source machine-learning framework commonly used for training and deploying AI models.

PyTorch can execute tensor operations on CPUs and GPUs.

---

# Q

## Quantization

Representing model values using lower numerical precision than the original representation.

Examples include:

```text
FP16
FP8
INT8
INT4
```

Quantization can reduce memory usage and improve performance, but it can also affect numerical accuracy.

---

# R

## Register

A very small, fast storage location associated with GPU thread execution.

Registers are private to individual threads.

Register availability is limited.

High register usage can reduce the number of simultaneously resident threads or blocks.

## Register Pressure

The amount of demand a kernel places on the available register resources.

High register pressure can reduce occupancy and may lead to register spilling.

## Register Spilling

A situation where values that cannot remain in registers are placed in another memory space.

Spilling can reduce performance because the alternative storage is generally slower than registers.

## Roofline Model

A performance model used to reason about whether a workload is limited by computation or memory bandwidth.

A simplified view is:

```text
Performance
    ↑
    │          Compute Roof
    │        ───────────────
    │       /
    │      /
    │     /
    │    /
    └────────────────────────→
          Arithmetic Intensity
```

The model connects:

* Arithmetic Intensity
* Compute Throughput
* Memory Bandwidth

It provides a useful upper-bound mental model, not a guarantee of actual performance.

---

# S

## SM

Short for Streaming Multiprocessor.

An NVIDIA SM is a major execution structure containing resources used to execute many GPU threads.

These resources include scheduling hardware, registers, shared memory, and execution resources.

The exact organization changes across GPU generations.

## SIMD

Single Instruction, Multiple Data.

A parallel execution model where one instruction operates across multiple data elements.

SIMT is related to SIMD but provides a thread-oriented programming model.

## SIMT

Single Instruction, Multiple Threads.

NVIDIA's CUDA programming model organizes threads into warps for execution.

Threads in a warp execute according to the SIMT model while retaining individual thread state and control flow.

## Shared Memory

Fast, programmer-managed memory shared by threads within a CUDA thread block.

Shared memory is commonly used for:

* Data reuse
* Tiling
* Cooperation between threads
* Reducing repeated accesses to larger memory
* Synchronization

## Synchronization

A mechanism that coordinates execution between threads or between the CPU and GPU.

Examples include:

```text
cudaDeviceSynchronize()
```

and thread-block synchronization primitives.

Synchronization can be necessary for correctness but can also reduce performance if overused.

## Strided Access

A memory-access pattern where consecutive threads access memory locations separated by a fixed distance.

For example:

```text
Thread 0 → A[0]
Thread 1 → A[32]
Thread 2 → A[64]
```

Large strides can reduce memory efficiency.

---

# T

## Tensor

A multi-dimensional array used extensively in machine learning.

Examples:

```text
Scalar → 0D
Vector → 1D
Matrix → 2D
Image batch → 4D
```

PyTorch represents model inputs, weights, activations, and outputs using tensors.

## Tensor Core

Specialized NVIDIA GPU hardware designed to accelerate matrix operations commonly used in AI and other numerical workloads.

Tensor Cores support specific data types and matrix operations depending on the GPU architecture.

## Thread

The smallest unit of computation exposed by CUDA's programming model.

A CUDA kernel is executed by many threads.

Each thread has its own thread identity and execution state.

## Thread Block

A group of CUDA threads.

Threads within a block:

* Execute on one SM
* Can cooperate
* Can use shared memory
* Can synchronize with each other

## Thread Block Cluster

An additional grouping available on supported NVIDIA architectures.

A cluster groups thread blocks together and provides additional opportunities for communication and synchronization.

It is an advanced CUDA feature and is not required for understanding the introductory labs.

## Thread Dimension

The number of threads in each dimension of a CUDA thread block.

CUDA exposes:

```text
blockDim.x
blockDim.y
blockDim.z
```

## Throughput

The amount of useful work completed per unit of time.

Examples:

```text
TFLOPS
tokens/sec
images/sec
requests/sec
GB/s
```

Throughput is different from latency.

---

# U

## Utilization

A general measure of how much of a resource is being used.

GPU utilization can refer to different measurements depending on the tool.

Do not automatically interpret:

```text
GPU utilization = 100%
```

as:

```text
GPU performance = optimal
```

A GPU can be fully occupied while executing an inefficient workload.

## Unified Memory

CUDA memory functionality that provides a managed memory model accessible by CPUs and GPUs.

It can simplify memory management, but it does not make data movement free.

---

# W

## Warp

A group of 32 CUDA threads on NVIDIA GPUs.

Warps are the basic grouping used by the CUDA SIMT execution model.

A block containing:

```text
128 threads
```

contains:

```text
128 / 32 = 4 warps
```

## Warp Divergence

See [Divergence](#divergence).

It occurs when threads within a warp follow different control-flow paths.

## Warp Lane

See [Lane](#lane).

A lane identifies a thread's position within its warp.

## Warp Scheduler

Hardware responsible for selecting eligible warp instructions for execution on an SM.

Schedulers help keep execution resources busy while other warps wait.

---

# X

## XMX

Intel's Xe Matrix Extensions.

Hardware acceleration features used by Intel GPUs for matrix operations.

The exact capabilities depend on the GPU architecture.

---

# AI Workload Terms

## Activation

An intermediate tensor produced while a neural network processes input.

Activations can consume substantial GPU memory, especially during training.

## Batch Size

The number of examples processed together in one operation.

For example:

```text
batch size = 32
```

means 32 examples are processed together.

Increasing batch size can improve GPU utilization in some workloads but also increases memory usage.

## Decode

The token-generation phase of autoregressive LLM inference.

During decode, the model typically generates new tokens one at a time while reusing previously computed information such as the KV cache.

## GEMM

General Matrix-Matrix Multiplication.

GEMM operations are fundamental to many neural-network workloads.

A simplified form is:

```text
C = A × B
```

possibly with additional scaling or accumulation.

## KV Cache

Key-Value cache used by transformer-based autoregressive models.

It stores previously computed attention key and value representations so they do not need to be recomputed for every generated token.

The KV cache can become a major memory consumer during LLM inference.

## LLM

Large Language Model.

A neural network trained to model and generate language.

Modern LLMs are commonly transformer-based and rely heavily on matrix operations.

## Matrix Multiplication

An operation multiplying matrices.

Example:

```text
C = A × B
```

Matrix multiplication is one of the most important computational patterns in modern AI workloads.

## Model Parallelism

A strategy where a model's computation or parameters are distributed across multiple devices.

Model parallelism includes approaches such as:

* Tensor parallelism
* Pipeline parallelism

## Tensor Parallelism

A form of model parallelism where individual tensor operations or model layers are partitioned across multiple GPUs.

---

# Performance Terms

## Bottleneck

The component or resource currently limiting system performance.

Possible bottlenecks include:

* Compute
* Memory bandwidth
* Memory latency
* Synchronization
* CPU
* PCIe
* GPU interconnect
* Kernel launch overhead

Finding the bottleneck should come before optimizing.

## Kernel Launch Overhead

The time and system work required to launch a GPU kernel.

For very small workloads, launch overhead can become a significant fraction of total runtime.

## Scaling Efficiency

A measure of how effectively additional GPUs increase useful performance.

A simplified definition is:

```text
Scaling Efficiency =
Actual Speedup
--------------
Ideal Speedup
```

For example, if four GPUs provide only 3× speedup:

```text
3 / 4 = 75%
```

scaling efficiency is approximately 75%.

---

# Memory Terms

## Data Reuse

Using data multiple times after loading it into a faster or closer memory level.

Data reuse can reduce expensive memory traffic.

## Tiling

Dividing a larger computation into smaller pieces that fit into faster memory such as shared memory or cache.

Tiling is widely used in optimized matrix operations.

## Memory Transaction

A unit of memory traffic performed by the hardware memory system.

The exact transaction behavior depends on the architecture and memory space.

Efficient access patterns allow threads' requests to be served efficiently.

---

# Interconnect Terms

## GPU Topology

The physical and logical arrangement of CPUs, GPUs, PCIe connections, NVLink connections, switches, and other system components.

Topology can strongly affect multi-GPU communication performance.

## Collective Communication

Communication operation involving multiple GPUs or processes.

Common collectives include:

* All-Reduce
* All-Gather
* Broadcast
* Reduce
* Scatter

Collectives are fundamental to distributed AI workloads.

## All-Reduce

A collective operation that combines values from multiple devices and distributes the combined result back to all participants.

It is widely used in distributed training.

---

# Production AI Terms

## Batching

Processing multiple requests or examples together.

Batching can improve hardware utilization and throughput.

However, larger batches can increase memory usage and latency.

## Dynamic Batching

Automatically grouping requests that arrive at similar times into a batch.

Dynamic batching can improve GPU utilization in inference systems while attempting to control latency.

## Latency

See the main definition under [L](#l).

In production AI systems, latency may refer to:

* Request latency
* Time to first token
* Time per output token
* End-to-end latency

## Tokens per Second

A throughput measurement commonly used for LLM inference.

It describes how many tokens a model generates or processes per second.

The exact meaning should always be specified because:

```text
input tokens/sec
```

and:

```text
output tokens/sec
```

are different measurements.

## Cost per Token

The estimated infrastructure cost associated with processing or generating a token.

It depends on factors such as:

* GPU cost
* Utilization
* Model size
* Batch size
* Throughput
* Sequence length
* Serving architecture

---

# Architecture Terms

## Chiplet

A smaller silicon die used as part of a larger packaged system.

Chiplet-based designs can improve manufacturing flexibility and yield, but introduce packaging and interconnect considerations.

## Monolithic GPU

A GPU design where major compute resources are implemented on a single large die.

## Packaging

The physical technology used to connect and integrate semiconductor dies, memory, and other components into a usable device.

Modern AI accelerators increasingly depend on advanced packaging.

---

# Quick Reference

## Execution hierarchy

```text
Grid
  ↓
Thread Blocks
  ↓
Warps
  ↓
Threads
```

## Simplified hardware view

```text
GPU
 │
 ├── SM
 │    ├── Warp Schedulers
 │    ├── Registers
 │    ├── Shared Memory / L1
 │    └── Execution Resources
 │
 └── Other SMs
```

## Simplified memory hierarchy

```text
Registers
    ↓
Shared Memory / L1
    ↓
L2
    ↓
GPU Device Memory
```

## Simplified AI execution path

```text
Python
  ↓
PyTorch
  ↓
Tensor Operations
  ↓
Optimized GPU Implementation
  ↓
Kernel
  ↓
Grid
  ↓
Blocks
  ↓
Warps
  ↓
SMs
  ↓
Execution + Memory
```

## Performance questions

When a GPU workload is slow, ask:

1. Is there enough parallel work?
2. Is the GPU waiting for data?
3. Is memory bandwidth the bottleneck?
4. Is computation the bottleneck?
5. Are warps diverging?
6. Is register or shared-memory usage limiting residency?
7. Is synchronization expensive?
8. Is CPU ↔ GPU communication expensive?
9. Is GPU ↔ GPU communication expensive?
10. Is the entire system actually keeping the GPU busy?

These questions are more useful than simply asking:

> "How many cores does this GPU have?"
