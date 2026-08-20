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

This project works backward from that line, following the complete path:

```text
Python → PyTorch → GPU operation → Kernel → Grid → Thread Blocks → Warps → Threads → SMs → Execution → Memory → Performance → AI Workloads → Multi-GPU → Production
```

The articles are published externally as **The Anatomy of Silicon** series; this repository holds the labs, benchmarks, and diagrams that make the concepts runnable and measurable.

> The goal is not to memorize GPU terminology. The goal is to understand **why GPUs behave the way they do**, measure that behavior with real experiments, and eventually use that knowledge to reason about production AI workloads.

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
- [Target Repository Structure](#target-repository-structure)
- [Benchmarking Philosophy](#benchmarking-philosophy)
- [Hardware Coverage](#hardware-coverage)
- [What This Project Is Not](#what-this-project-is-not)
- [The Final Mental Model](#the-final-mental-model)
- [How the Roadmap Will Evolve](#how-the-roadmap-will-evolve)

## How to Use This Repository

The project has three connected layers: **articles** (published externally) explain the concepts, **labs** (this repository) run them, and **benchmarks** measure what actually happens.

Every part follows the same chain:

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

```text
Article 01: "Threads are grouped into warps"
   ↓
labs: warp_mapping.cu
   ↓
Run on GPU → observe warp/lane mapping

Article 02: "Memory access patterns matter"
   ↓
labs: coalesced_access.cu · strided_access.cu
   ↓
Benchmark → compare bandwidth
```

The recommended way to work through any part:

```text
Read → Run → Change the code → Measure → Explain what happened
```

That last step matters.

> If you can run a benchmark but cannot explain why the result changed, you have learned how to execute a script — not how the GPU works.

## Learning Philosophy

The project deliberately avoids jumping directly into advanced GPU optimization.

> You cannot meaningfully optimize something you cannot mentally trace.

Each stage of the series answers a different question:

| Stage | Question |
|-------|----------|
| Beginner | What is happening? |
| Intermediate | Why is it happening? |
| Advanced | How does the hardware make it happen? |
| Performance Engineering | What is limiting it? |
| Production Engineering | How do I design the entire system around those limits? |

## The Series at a Glance

The series is planned as **ten parts**. Each part builds on the previous one, and parts are published in order. Only Part 01 has been released so far.

| Part | Focus | Status | Material |
|:--:|--------|:------:|----------|
| [01](#part-01--gpu-execution) | GPU Execution | ✅ Published | [Article](../01-gpu-execution/README.md) · [Lab](../01-gpu-execution/labs/README.md) |
| [02](#part-02--gpu-memory) | GPU Memory | 🗓️ Planned | — |
| [03](#part-03--tensor-cores-and-ai-compute) | Tensor Cores and AI Compute | 🗓️ Planned | — |
| [04](#part-04--gpu-architecture-evolution) | GPU Architecture Evolution | 🗓️ Planned | — |
| [05](#part-05--gpu-architecture-beyond-nvidia) | GPU Architecture Beyond NVIDIA | 🗓️ Planned | — |
| [06](#part-06--gpu-performance-engineering) | GPU Performance Engineering | 🗓️ Planned | — |
| [07](#part-07--gpu-execution-of-llms) | GPU Execution of LLMs | 🗓️ Planned | — |
| [08](#part-08--gpu-interconnects) | GPU Interconnects | 🗓️ Planned | — |
| [09](#part-09--multi-gpu-systems) | Multi-GPU Systems | 🗓️ Planned | — |
| [10](#part-10--production-gpu-systems) | Production GPU Systems | 🗓️ Planned | — |

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

**Lab:** [`01-gpu-execution/labs/`](../01-gpu-execution/labs/) — available · **Diagrams:** `01-gpu-execution/diagrams/` — in progress

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

**Lab:** `02-gpu-memory/labs/` — planned

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

Run mathematically similar operations with different memory-access patterns, then measure execution time, effective bandwidth, memory traffic, and slowdown. The purpose is to turn:

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

**Lab:** `03-tensor-cores/labs/` — planned

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

**Lab:** `04-nvidia-architectures/labs/` — planned

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

**Lab:** `05-alternative-architectures/labs/` — planned

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

**Lab:** `06-gpu-performance/labs/` — planned

**Experiments**

Compare a naive kernel against an optimized kernel, and measure latency, throughput, achieved bandwidth, utilization, occupancy, and instruction behavior.

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

**Lab:** `07-llm-workloads/labs/` — planned

**Experiments**

Measure prefill latency, decode latency, tokens/sec, memory usage, batch-size scaling, and sequence-length scaling — comparing batch sizes 1, 2, 4, 8, …

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

**Lab:** `08-gpu-interconnects/labs/` — planned

**Experiments**

Measure CPU → GPU, GPU → CPU, and GPU → GPU transfers, then compare the effect of topology and communication path where the hardware supports it.

## Part 09 — Multi-GPU Systems

**Article 09 — Scaling AI Across Multiple GPUs** · 🗓️ Planned

> **Core question:** How do we turn multiple GPUs into one useful compute system?

**Concepts**

- Data parallelism, tensor parallelism, pipeline parallelism, model parallelism
- Distributed training
- All-reduce and all-gather
- Communication overhead
- Scaling efficiency

**Lab:** `09-multi-gpu/labs/` — planned

**Experiments**

Compare 1, 2, and 4 GPUs (where available), measuring throughput, scaling efficiency, communication overhead, and synchronization overhead.

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

**Lab:** `10-production-gpu/labs/` — planned

> **Goal:** Move from *GPU optimization* to *system optimization*. A production system is not successful because one kernel is fast. It is successful when the entire system meets its latency, throughput, reliability, and cost requirements.

---

## Target Repository Structure

The repository is organized **by part**: each published article gets one self-contained folder holding its companion material, labs, and diagrams. The current repository is a subset of this layout — a part's folder appears only when the part is published, and its optional subfolders (`benchmarks/`, `notebooks/`) only when the corresponding material exists.

```text
gpu-architecture-for-ai/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
│
├── docs/                     # project-wide documentation
│   ├── roadmap.md
│   ├── setup.md
│   └── glossary.md
│
├── scripts/                  # repository tooling
│
└── 01-gpu-execution/         # one folder per published part
    ├── README.md             # article companion: concepts, lab map, references
    ├── labs/                 # runnable experiments
    │   ├── README.md
    │   ├── python/
    │   └── cuda/
    └── diagrams/             # visuals for this article
```

Future parts follow the same internal template:

```text
02-gpu-memory/ · 03-tensor-cores/ · 04-nvidia-architectures/ ·
05-alternative-architectures/ · 06-gpu-performance/ · 07-llm-workloads/ ·
08-gpu-interconnects/ · 09-multi-gpu/ · 10-production-gpu/
```

## Benchmarking Philosophy

Every performance claim in this repository follows the same rules — measure first, explain second:

| Rule | Why |
|------|-----|
| Never trust a single run | Warm-up, clocks, cache state, background load, and thermals all vary between executions. |
| Warm up before measuring | The first execution rarely represents steady-state behavior. |
| Synchronize correctly | GPU work executes asynchronously; naive host timers can measure the wrong thing entirely. |
| Record the environment | GPU, driver, CUDA, framework versions, OS, precision, shapes, and iteration counts — a result without context cannot be reproduced. |
| Label educational vs. production | Tiny benchmarks demonstrate concepts; they do not predict serving performance. |

Practical commands, the environment template, and troubleshooting live in the [Setup Guide](setup.md#benchmarking-guidelines).

## Hardware Coverage

The project focuses heavily on NVIDIA CUDA initially because CUDA provides a mature environment for exposing GPU execution concepts — direct access to `threadIdx`, `blockIdx`, `blockDim`, `gridDim`, warps, shared memory, and kernel launches makes the execution model concrete. The current CUDA model organizes work into grids, thread blocks, and 32-thread warps executing on individual SMs.

That does not mean the concepts only apply to NVIDIA GPUs. The project distinguishes between:

- **General GPU concepts:** parallel execution, memory hierarchy, bandwidth, latency, data reuse, matrix computation, GPU utilization
- **Vendor-specific implementation details:** CUDA, SM, CUDA warps, Tensor Cores, NVLink, Nsight

Later parts introduce AMD, Intel, and Apple architectures separately, and equivalent concepts are explored in other GPU programming environments where useful.

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

## The Final Mental Model

At the beginning, the journey looks like this:

```text
AI Model
   ↓
GPU
```

At the end, every layer is understandable:

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

That is the destination — and the goal of this repository is to make every layer in that diagram understandable.

## How the Roadmap Will Evolve

The roadmap is intentionally not frozen. GPU architectures, software stacks, numerical formats, interconnects, and AI workloads change quickly.

Future articles may therefore:

- add new GPU architectures
- replace outdated examples
- introduce new numerical formats
- add new profiling techniques and benchmark environments
- add new AI workload experiments
- expand multi-GPU experiments

The core learning path — execution → memory → computation → performance → AI workloads → systems → production — should remain stable even as individual GPU generations change.

> Do not skip the fundamentals. The fastest way to understand advanced GPU systems is not to start with the most advanced GPU. It is to understand the simple execution model well enough that the advanced hardware becomes an extension of an existing mental model.

---

<p align="center">
<sub>
<a href="../README.md">Main README</a> ·
<a href="setup.md">Setup</a> ·
<a href="glossary.md">Glossary</a>
</sub>
</p>
