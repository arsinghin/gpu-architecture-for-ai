<div align="center">

# Roadmap

**GPU Architecture for AI**

A hands-on journey from Python code to GPU silicon — from basic execution to production AI systems.

![Progress](https://img.shields.io/badge/Articles-1_of_10_Published-brightgreen)

Part of [GPU Architecture for AI](../README.md) · [Setup](setup.md) · [Glossary](glossary.md)

</div>

---

## Overview

Modern AI systems depend heavily on GPUs, but many AI engineers learn to use GPUs without ever building a mental model of what happens underneath a PyTorch operation.

You write:

```python
output = model(input)
```

and the framework takes care of the rest. That is convenient. It is also where many performance problems become mysterious.

This project works backward from that line. The articles are published externally as **The Anatomy of Silicon** series; this repository holds the labs, benchmarks, and diagrams that make the concepts runnable and measurable.

> The goal is not to memorize GPU terminology. The goal is to understand **why GPUs behave the way they do**, measure that behavior with real experiments, and eventually use that knowledge to reason about production AI workloads.

The complete learning path:

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
SMs
  ↓
Execution
  ↓
Memory
  ↓
Performance
  ↓
AI Workloads
  ↓
Multi-GPU Systems
  ↓
Production
```

## Table of Contents

- [Overview](#overview)
- [How to Use This Repository](#how-to-use-this-repository)
- [Learning Philosophy](#learning-philosophy)
- [The Series at a Glance](#the-series-at-a-glance)
- [Part 01 — GPU Execution](#part-01--gpu-execution)
- [Part 02 — GPU Memory](#part-02--gpu-memory)
- [Part 03 — Tensor Cores and AI Compute](#part-03--tensor-cores-and-ai-compute)
- [Part 04 — GPU Architecture Evolution](#part-04--gpu-architecture-evolution)
- [Part 05 — GPU Architecture Beyond NVIDIA](#part-05--gpu-architecture-beyond-nvidia)
- [Part 06 — GPU Performance Engineering](#part-06--gpu-performance-engineering)
- [Part 07 — GPU Execution of LLMs](#part-07--gpu-execution-of-llms)
- [Part 08 — GPU Interconnects](#part-08--gpu-interconnects)
- [Part 09 — Multi-GPU Systems](#part-09--multi-gpu-systems)
- [Part 10 — Production GPU Systems](#part-10--production-gpu-systems)
- [Article → Lab → Experiment](#article--lab--experiment)
- [Target Repository Structure](#target-repository-structure)
- [Benchmarking Philosophy](#benchmarking-philosophy)
- [Hardware Coverage](#hardware-coverage)
- [What This Project Is Not](#what-this-project-is-not)
- [The Progression](#the-progression)
- [The Final Mental Model](#the-final-mental-model)
- [How the Roadmap Will Evolve](#how-the-roadmap-will-evolve)

## How to Use This Repository

The project has three connected layers:

```text
Articles (published externally)
   ↓  explain the concept
Labs (this repository)
   ↓  run the concept
Benchmarks
   ↓  measure the behavior
```

The articles focus on explanation and mental models. The labs contain runnable code. The benchmarks measure what happens when the code actually runs.

You do not need to read every article before touching the code. However, the recommended path is:

```text
Read
  ↓
Run
  ↓
Change the code
  ↓
Measure
  ↓
Explain what happened
```

That last step matters.

> If you can run a benchmark but cannot explain why the result changed, you have learned how to execute a script — not how the GPU works.

## Learning Philosophy

The project follows a simple progression, from mental model to production system:

```text
Mental Model
  ↓
Runnable Example
  ↓
Measurement
  ↓
Optimization
  ↓
Architecture
  ↓
System Design
  ↓
Production
```

Each stage answers a different question:

| Stage | Question |
|-------|----------|
| Beginner | What is happening? |
| Intermediate | Why is it happening? |
| Advanced | How does the hardware make it happen? |
| Performance Engineering | What is limiting it? |
| Production Engineering | How do I design the entire system around those limits? |

The project deliberately avoids jumping directly into advanced GPU optimization.

> You cannot meaningfully optimize something you cannot mentally trace.

## The Series at a Glance

The series is planned as **ten parts**. Each part builds on the previous one, and parts are published in order. Only Part 01 has been released so far.

| Part | Focus | Status | Material |
|:--:|--------|:------:|----------|
| [01](#part-01--gpu-execution) | GPU Execution | ✅ Published | [Article](../articles/01-gpu-execution/README.md) · [Lab](../labs/01-gpu-execution/README.md) |
| [02](#part-02--gpu-memory) | GPU Memory | 🗓️ Planned | — |
| [03](#part-03--tensor-cores-and-ai-compute) | Tensor Cores and AI Compute | 🗓️ Planned | — |
| [04](#part-04--gpu-architecture-evolution) | GPU Architecture Evolution | 🗓️ Planned | — |
| [05](#part-05--gpu-architecture-beyond-nvidia) | GPU Architecture Beyond NVIDIA | 🗓️ Planned | — |
| [06](#part-06--gpu-performance-engineering) | GPU Performance Engineering | 🗓️ Planned | — |
| [07](#part-07--gpu-execution-of-llms) | GPU Execution of LLMs | 🗓️ Planned | — |
| [08](#part-08--gpu-interconnects) | GPU Interconnects | 🗓️ Planned | — |
| [09](#part-09--multi-gpu-systems) | Multi-GPU Systems | 🗓️ Planned | — |
| [10](#part-10--production-gpu-systems) | Production GPU Systems | 🗓️ Planned | — |

```text
01 GPU Execution
   ↓
02 GPU Memory
   ↓
03 Tensor Cores and AI Compute
   ↓
04 GPU Architecture Evolution
   ↓
05 GPU Architecture Beyond NVIDIA
   ↓
06 GPU Performance Engineering
   ↓
07 GPU Execution of LLMs
   ↓
08 GPU Interconnects
   ↓
09 Multi-GPU Systems
   ↓
10 Production GPU Systems
```

---

## Part 01 — GPU Execution

**Article 01 — How GPUs Actually Execute AI Workloads** · ✅ Published

> **Core question:** What happens inside a GPU when you run a neural network?

**Concepts**

- CPU vs. GPU · throughput vs. latency
- SIMT
- Threads, warps, thread blocks, grids
- SMs and warp schedulers
- Registers, shared memory, L1, L2
- Kernels
- Latency hiding, occupancy, divergence
- PyTorch execution and asynchronous execution

**Lab:** [`labs/01-gpu-execution/`](../labs/01-gpu-execution/) — available · **Diagrams:** `diagrams/01-gpu-execution/` — in progress

**Experiments**

- Detect the CUDA device
- CPU vs. GPU execution
- Correct GPU timing
- Launch a CUDA kernel
- Inspect thread indexing
- Inspect warp and lane mapping
- Observe divergent control flow
- Inspect GPU hardware properties

> **Reader takeaway:** You should be able to mentally trace `Python → PyTorch → Kernel → Grid → Thread Blocks → Warps → Threads → SM → Execution`.

## Part 02 — GPU Memory

**Article 02 — GPU Memory Explained: HBM, SRAM, Cache and the Memory Wall** · 🗓️ Planned

> **Core question:** Why can a GPU with enormous compute power still be slow?

**Concepts**

- Registers, SRAM, shared memory
- L1 and L2 cache
- GPU device memory (HBM, GDDR) and host RAM
- Memory capacity, latency, bandwidth, transactions
- Coalescing and strided access
- Tiling and data reuse
- Arithmetic intensity and the Roofline model
- Compute-bound vs. memory-bound workloads

**Lab:** `labs/02-gpu-memory/` — planned

**Experiments**

- Sequential memory access
- Strided memory access
- Coalesced vs. non-coalesced access
- Shared-memory reuse
- Tiled matrix multiplication
- Different tile sizes
- Effective bandwidth measurement
- Arithmetic intensity experiments

**Key demonstration**

Run mathematically similar operations with different memory-access patterns, then measure:

- execution time
- effective bandwidth
- memory traffic
- slowdown

The purpose is to turn:

> "Memory access matters"

into:

> "Here is the measurement showing exactly why."

> **Reader takeaway:** A GPU can have enormous compute capacity and still be limited by moving data.

## Part 03 — Tensor Cores and AI Compute

**Article 03 — Tensor Cores and the Hardware Behind AI Math** · 🗓️ Planned

> **Core question:** Why are modern GPUs so good at matrix-heavy AI workloads?

**Concepts**

- Matrix multiplication and GEMM
- Fused operations
- Numerical formats: FP32, FP16, BF16, TF32, FP8, FP4
- Tensor Cores and matrix multiply-accumulate
- Precision vs. performance · accumulation precision
- Quantization and sparsity

**Lab:** `labs/03-tensor-cores/` — planned

**Experiments**

- Matrix multiplication in FP32
- Matrix multiplication in lower precision
- Tensor Core–enabled workloads
- Different matrix sizes and batch sizes
- Precision comparison
- Throughput comparison

> **Reader takeaway:** Modern AI workloads are not simply "run lots of FP32 instructions." Modern AI hardware uses specialized matrix-processing paths and lower numerical precision to achieve much higher throughput.

## Part 04 — GPU Architecture Evolution

**Article 04 — How NVIDIA GPU Architectures Evolved** · 🗓️ Planned

> **Core question:** What actually changes from one GPU generation to the next?

**Architectures covered:** Volta · Turing · Ampere · Ada Lovelace · Blackwell · future architectures as appropriate

**Concepts**

- SM evolution
- Tensor Core evolution
- Memory subsystem and cache changes
- Precision support
- Scheduling changes
- Specialized accelerators
- Interconnect evolution
- Packaging

**Lab:** `labs/04-nvidia-architectures/` — planned

> **Goal:** Do not memorize specifications. Understand what changed — and why those changes matter for workloads.

## Part 05 — GPU Architecture Beyond NVIDIA

**Article 05 — NVIDIA vs AMD vs Intel vs Apple GPUs** · 🗓️ Planned

> **Core question:** Are all GPUs built the same way?
>
> No.

**Architectures covered:** NVIDIA · AMD · Intel · Apple

**Concepts**

- Different execution models and terminology
- Matrix acceleration
- Memory systems and unified memory
- Packaging
- Software ecosystems
- AI acceleration

**Lab:** `labs/05-alternative-architectures/` — planned

> **Important rule:** Do not force different architectures into NVIDIA terminology. These terms are related conceptually but are **not** interchangeable:

| Vendor | Terminology |
|--------|-------------|
| NVIDIA | SM |
| AMD | Compute Units / Workgroup Processors |
| Intel | Xe architecture |
| Apple | Apple GPU architecture |

## Part 06 — GPU Performance Engineering

**Article 06 — Why Your GPU Is Slow** · 🗓️ Planned

> **Core question:** If the GPU is powerful, what is actually limiting performance?

**Concepts**

- GPU utilization and occupancy
- Memory throughput and compute throughput
- Kernel launch overhead and synchronization
- Register pressure and shared-memory usage
- Cache behavior and warp divergence
- Kernel fusion
- Arithmetic intensity and Roofline analysis
- Profiling

**Tools:** PyTorch Profiler · NVIDIA Nsight Systems · NVIDIA Nsight Compute · CUDA events · CUDA runtime APIs

**Lab:** `labs/06-gpu-performance/` — planned

**Experiments**

Compare a naive kernel against an optimized kernel, and measure:

- latency
- throughput
- achieved bandwidth
- utilization
- occupancy
- instruction behavior

> **Objective:** Answer *"What is the bottleneck?"* — not *"Which optimization trick should I try next?"*

## Part 07 — GPU Execution of LLMs

**Article 07 — What Actually Happens When an LLM Runs on a GPU?** · 🗓️ Planned

> **Core question:** How does transformer inference turn into GPU work?

**Concepts**

- Transformer layers, embeddings, linear layers
- Matrix multiplication and attention
- Softmax
- KV cache
- Prefill and decode
- Batch size and sequence length
- Memory bandwidth and compute throughput
- Quantization

**Lab:** `labs/07-llm-workloads/` — planned

**Experiments**

Measure:

- prefill latency
- decode latency
- tokens/sec
- memory usage
- batch-size scaling
- sequence-length scaling

Compare batch sizes 1, 2, 4, 8, …

> **Goal:** Connect low-level GPU behavior with real LLM serving behavior.

## Part 08 — GPU Interconnects

**Article 08 — How GPUs Talk to Each Other** · 🗓️ Planned

> **Core question:** What happens when one GPU is not enough?

**Concepts**

- PCIe, NVLink, NVSwitch
- Host-to-device and device-to-device communication
- GPU topology
- Communication bandwidth and latency
- Collective communication

**Lab:** `labs/08-gpu-interconnects/` — planned

**Experiments**

Measure:

- CPU → GPU
- GPU → CPU
- GPU → GPU

Then compare the effect of topology and communication path where the hardware supports it.

## Part 09 — Multi-GPU Systems

**Article 09 — Scaling AI Across Multiple GPUs** · 🗓️ Planned

> **Core question:** How do we turn multiple GPUs into one useful compute system?

**Concepts**

- Data parallelism, tensor parallelism, pipeline parallelism, model parallelism
- Distributed training
- All-reduce and all-gather
- Communication overhead
- Scaling efficiency

**Lab:** `labs/09-multi-gpu/` — planned

**Experiments**

Compare 1, 2, and 4 GPUs (where available), and measure:

- throughput
- scaling efficiency
- communication overhead
- synchronization overhead

> **Important lesson:** More GPUs do not automatically mean proportionally more performance.

## Part 10 — Production GPU Systems

**Article 10 — Designing Production GPU Systems** · 🗓️ Planned

> **Core question:** How do we turn GPU hardware into a reliable production AI system?

**Concepts**

- Model serving
- Batching and dynamic batching
- Concurrency, latency, throughput
- GPU utilization
- Memory fragmentation
- CUDA Graphs and kernel fusion
- Quantization and model parallelism
- Monitoring and profiling
- Cost per request, cost per token, capacity planning

**Lab:** `labs/10-production-gpu/` — planned

> **Goal:** Move from *GPU optimization* to *system optimization*. A production system is not successful because one kernel is fast. It is successful when the entire system meets its latency, throughput, reliability, and cost requirements.

---

## Article → Lab → Experiment

Every article should eventually have this relationship:

```text
Article
   │ explains
   ↓
Concept
   │ implemented by
   ↓
Lab
   │ measured by
   ↓
Experiment
   │ produces
   ↓
Result
```

For example:

**Article 01:**

```text
Article 01
   ↓
"Threads are grouped into warps"
   ↓
warp_mapping.cu
   ↓
Run on GPU
   ↓
Observe warp/lane mapping
```

**Article 02:**

```text
Article 02
   ↓
"Memory access patterns matter"
   ↓
coalesced_access.cu
strided_access.cu
   ↓
Run benchmark
   ↓
Measure bandwidth
   ↓
Compare results
```

This keeps the repository connected to the actual learning objectives.

## Target Repository Structure

The repository grows together with the article series. The intended long-term structure is:

```text
gpu-architecture-for-ai/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
│
├── articles/
│   ├── 01-gpu-execution/
│   ├── 02-gpu-memory/
│   ├── 03-tensor-cores/
│   ├── 04-nvidia-architectures/
│   ├── 05-alternative-architectures/
│   ├── 06-gpu-performance/
│   ├── 07-llm-workloads/
│   ├── 08-gpu-interconnects/
│   ├── 09-multi-gpu/
│   └── 10-production-gpu/
│
├── labs/
│   ├── 01-gpu-execution/
│   ├── 02-gpu-memory/
│   ├── 03-tensor-cores/
│   ├── 04-nvidia-architectures/
│   ├── 05-alternative-architectures/
│   ├── 06-gpu-performance/
│   ├── 07-llm-workloads/
│   ├── 08-gpu-interconnects/
│   ├── 09-multi-gpu/
│   └── 10-production-gpu/
│
├── benchmarks/
│   ├── scripts/
│   ├── results/
│   └── schemas/
│
├── notebooks/
│
├── diagrams/
│
├── docs/
│   ├── roadmap.md
│   ├── glossary.md
│   ├── hardware-matrix.md
│   └── troubleshooting.md
│
└── scripts/
```

This is the target structure. It does not mean every directory needs to exist immediately — the current repository is a subset of this layout. Directories are added when their corresponding article or experiment exists.

## Benchmarking Philosophy

The repository will eventually contain many performance measurements. Those measurements should follow a few rules.

### 1. Never Trust a Single Run

GPU execution is affected by:

- warm-up
- clocks
- background activity
- cache state
- system load
- driver behavior
- thermal conditions

Use repeated measurements where appropriate.

### 2. Warm Up GPU Workloads

The first execution may not represent steady-state behavior. Where appropriate:

```text
Warm-up
  ↓
Measurement
  ↓
Repeated runs
  ↓
Summary statistic
```

### 3. Synchronize Correctly

GPU operations can execute asynchronously. Do not assume:

```text
start()
gpu_operation()
end()
```

automatically measures GPU execution time. Use CUDA-aware timing or explicit synchronization where appropriate.

### 4. Record the Environment

A benchmark result should record enough information to understand its context. At minimum:

- GPU and GPU memory
- driver
- CUDA, PyTorch, and Python versions
- operating system
- precision
- input size and batch size
- relevant configuration

### 5. Separate Educational from Production Benchmarks

A tiny benchmark can explain a concept. It does not automatically represent production performance.

For example, ten million element-wise operations does not tell us how an entire LLM serving system behaves.

The repository will clearly label simplified experiments as educational.

## Hardware Coverage

The project focuses heavily on NVIDIA CUDA initially because CUDA provides a mature environment for exposing GPU execution concepts. That does not mean the concepts only apply to NVIDIA GPUs.

The project distinguishes between:

- **General GPU concepts:** parallel execution, memory hierarchy, bandwidth, latency, data reuse, matrix computation, GPU utilization
- **Vendor-specific implementation details:** CUDA, SM, CUDA warps, Tensor Cores, NVLink, Nsight

Later parts of the project introduce AMD, Intel, and Apple architectures separately.

### Why CUDA Is the Initial Learning Environment

CUDA is used extensively in the early labs because it provides direct access to concepts such as:

- `threadIdx`, `blockIdx`, `blockDim`, `gridDim`
- warps
- shared memory
- kernel launches

These make the execution model concrete. The current CUDA programming model organizes work into grids, thread blocks, and 32-thread warps, with thread blocks executing on individual SMs.

Later, where useful, equivalent concepts can be explored using other GPU programming environments.

## What This Project Is Not

This repository is not intended to be:

- a complete CUDA reference
- an NVIDIA architecture specification database
- a collection of random CUDA examples
- a collection of benchmark numbers without explanation
- a replacement for official GPU documentation
- a generic deep-learning tutorial
- a collection of optimization tricks without reasoning

The project is specifically about building a mental model of GPU-based AI systems.

## The Progression

The entire project should eventually answer these questions:

| Level | Question | Answered In |
|:--:|-----------|-------------|
| 1 | What is a GPU? | Part 01 |
| 2 | How does a GPU execute thousands of pieces of work? | Part 01 |
| 3 | How do threads, warps, blocks, and SMs interact? | Part 01 |
| 4 | Where does the data live? | Part 02 |
| 5 | Why does memory access affect performance? | Part 02 |
| 6 | How does a GPU accelerate matrix math? | Part 03 |
| 7 | Why do modern AI models benefit from specialized hardware? | Parts 03–04 |
| 8 | Why does my GPU workload become slow? | Part 06 |
| 9 | How does an LLM use the GPU? | Part 07 |
| 10 | How do multiple GPUs communicate? | Part 08 |
| 11 | How do we scale AI across GPUs? | Part 09 |
| 12 | How do we turn all of this into a production system? | Part 10 |

That is the destination.

## The Final Mental Model

At the beginning:

```text
AI Model
   ↓
GPU
```

At the end:

```text
AI Model
   ↓
Framework
   ↓
GPU Operations
   ↓
Kernels
   ↓
Grids
   ↓
Thread Blocks
   ↓
Warps
   ↓
SMs
   ↓
Execution Resources
   ↕
Memory Hierarchy
   ↓
GPU Interconnect
   ↓
Other GPUs
   ↓
Distributed System
   ↓
Production Service
```

The goal of this repository is to make every layer in that diagram understandable.

## How the Roadmap Will Evolve

The roadmap is intentionally not frozen. GPU architectures, software stacks, numerical formats, interconnects, and AI workloads change quickly.

Future articles may therefore:

- add new GPU architectures
- replace outdated examples
- introduce new numerical formats
- add new profiling techniques
- add new benchmark environments
- add new AI workload experiments
- expand multi-GPU experiments

The core learning path should remain stable:

```text
Execution
  ↓
Memory
  ↓
Computation
  ↓
Performance
  ↓
AI Workloads
  ↓
Systems
  ↓
Production
```

That structure should remain useful even as individual GPU generations change.

> Do not skip the fundamentals. The fastest way to understand advanced GPU systems is not to start with the most advanced GPU. It is to understand the simple execution model well enough that the advanced hardware becomes an extension of an existing mental model.

---

<p align="center">
<sub>
<a href="../README.md">Main README</a> ·
<a href="setup.md">Setup</a> ·
<a href="glossary.md">Glossary</a>
</sub>
</p>
