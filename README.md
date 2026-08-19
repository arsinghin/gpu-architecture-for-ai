# GPU Architecture for AI

### From Python code to GPU execution, memory, compute, and production AI systems.

This repository is the hands-on companion to the **The Anatomy of Silicon** article series.

The goal is simple:

> **Understand what actually happens inside a GPU when AI code runs.**

Most AI engineers use GPUs every day.

They write:

```python
output = model(input)
```

and the framework handles the rest.

That is useful.

But when performance drops, memory fills up, inference becomes slow, or multiple GPUs refuse to scale linearly, the abstraction starts hiding the important part.

This project works underneath that abstraction.

---

## What This Project Covers

The learning path moves from the basics to production systems:

```text
Python
  ↓
PyTorch
  ↓
GPU Operations
  ↓
Kernels
  ↓
Threads
  ↓
Warps
  ↓
Thread Blocks
  ↓
SMs
  ↓
Memory
  ↓
Compute
  ↓
Performance
  ↓
AI / LLM Workloads
  ↓
Multi-GPU Systems
  ↓
Production
```

The goal is not to memorize GPU terminology.

The goal is to build a mental model that lets you answer:

> **Why is this GPU workload behaving this way?**

---

# Why This Repository Exists

GPU programming is often taught in disconnected pieces.

You learn:

- CUDA
- PyTorch
- Tensor Cores
- memory bandwidth
- CUDA kernels
- profiling
- LLM inference

but it is easy to miss how they fit together.

This repository connects those layers.

Each topic follows:

```text
Concept
   ↓
Explanation
   ↓
Runnable experiment
   ↓
Measurement
   ↓
Reasoning
```

The articles explain the concepts.

The labs make them executable.

The benchmarks make the behavior measurable.

---
# Lab 01 — GPU Execution

The practical experiments for Article 01 live here:

```text
labs/01-gpu-execution/
```

The lab contains:

```text
labs/01-gpu-execution/
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

The experiments progressively demonstrate:

```text
GPU device
    ↓
CPU vs GPU
    ↓
Asynchronous execution
    ↓
Threads
    ↓
Thread indexing
    ↓
Warps
    ↓
Divergence
    ↓
GPU hardware properties
```

---

# Start Here

Read Article 01 first:

**The Anatomy of Silicon: How GPUs Actually Execute AI Workloads**

Then run:

```bash
python labs/01-gpu-execution/python/01_device_check.py
```

Continue through the experiments in order.

Full setup instructions:

```text
docs/setup.md
```

Project roadmap:

```text
docs/roadmap.md
```

GPU terminology:

```text
docs/glossary.md
```

---

# Learning Path

The complete project is planned as a progression.

## 01 — GPU Execution

### Question

> What happens inside a GPU when AI code runs?

Topics:

- threads
- warps
- blocks
- grids
- SMs
- scheduling
- registers
- shared memory
- cache
- kernels
- divergence
- occupancy
- latency hiding

Status:

**Published + Lab available**

---

## 02 — GPU Memory

### Question

> Why can a powerful GPU still be slow?

Topics:

- registers
- SRAM
- shared memory
- L1
- L2
- HBM
- GDDR
- host RAM
- bandwidth
- latency
- capacity
- coalescing
- tiling
- data reuse
- arithmetic intensity
- Roofline Model
- compute-bound workloads
- memory-bound workloads

Status:

**Planned**

---

## 03 — Tensor Cores and AI Compute

### Question

> Why are modern GPUs so effective at AI workloads?

Topics:

- matrix multiplication
- GEMM
- Tensor Cores
- FP32
- FP16
- BF16
- TF32
- FP8
- FP4
- mixed precision
- quantization
- accumulation
- sparsity

Status:

**Planned**

---

## 04 — GPU Architecture Evolution

### Question

> What actually changes between GPU generations?

Topics include:

- Volta
- Turing
- Ampere
- Ada Lovelace
- Blackwell
- SM evolution
- Tensor Core evolution
- cache changes
- memory systems
- specialized hardware
- interconnects
- packaging

Status:

**Planned**

---

## 05 — GPU Architecture Beyond NVIDIA

### Question

> Are all GPUs built the same way?

Architectures covered:

- NVIDIA
- AMD
- Intel
- Apple

The goal is to understand the similarities and differences without pretending that vendor-specific terminology is interchangeable.

Status:

**Planned**

---

## 06 — GPU Performance Engineering

### Question

> Why is my GPU workload slow?

Topics:

- profiling
- utilization
- occupancy
- memory throughput
- compute throughput
- register pressure
- shared-memory usage
- cache behavior
- divergence
- kernel launch overhead
- synchronization
- kernel fusion
- Roofline analysis

Tools will include appropriate vendor and framework profiling tools.

Status:

**Planned**

---

## 07 — GPU Execution of LLMs

### Question

> What actually happens when an LLM runs on a GPU?

Topics:

- transformer layers
- matrix multiplication
- attention
- prefill
- decode
- KV cache
- batch size
- sequence length
- memory bandwidth
- compute throughput
- quantization

Status:

**Planned**

---

## 08 — GPU Interconnects

### Question

> What happens when one GPU is not enough?

Topics:

- PCIe
- NVLink
- NVSwitch
- GPU topology
- host-to-device communication
- device-to-device communication
- communication bandwidth
- communication latency

Status:

**Planned**

---

## 09 — Multi-GPU Systems

### Question

> How do multiple GPUs work together?

Topics:

- data parallelism
- tensor parallelism
- pipeline parallelism
- model parallelism
- all-reduce
- all-gather
- distributed training
- communication overhead
- scaling efficiency

Status:

**Planned**

---

## 10 — Production GPU Systems

### Question

> How do we turn GPU hardware into a reliable production AI system?

Topics:

- model serving
- batching
- dynamic batching
- concurrency
- latency
- throughput
- GPU utilization
- memory management
- CUDA Graphs
- kernel fusion
- quantization
- monitoring
- capacity planning
- cost per request
- cost per token

Status:

**Planned**

---

# Repository Structure

```text
gpu-architecture-for-ai/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
│
├── articles/
│   └── 01-gpu-execution/
│       └── README.md
│
├── labs/
│   └── 01-gpu-execution/
│       ├── README.md
│       ├── python/
│       └── cuda/
│
├── diagrams/
│   └── 01-gpu-execution/
│
└── docs/
    ├── roadmap.md
    ├── setup.md
    └── glossary.md
```

The repository will grow as new articles and labs are published.

Directories for unfinished articles are not created merely for decoration.

They will be added when the corresponding material exists.

---

# Articles

The `articles/` directory contains supporting material for the published articles.

Example:

```text
articles/
└── 01-gpu-execution/
    └── README.md
```

The article itself is published externally.

The repository contains:

- article information
- learning objectives
- concepts covered
- lab mapping
- diagrams
- references
- implementation notes where useful

---

# Labs

The `labs/` directory contains runnable experiments.

A lab should answer:

> **Can we demonstrate this concept with actual code?**

Labs are intentionally small at the beginning.

Later labs become more performance-oriented.

---

# Diagrams

The `diagrams/` directory stores diagrams used by the project.

For Article 01:

```text
diagrams/
└── 01-gpu-execution/
```

The diagrams are intended to explain concepts such as:

```text
CPU vs GPU
GPU execution hierarchy
Thread blocks
Warps
SM execution
Memory hierarchy
PyTorch → GPU execution
```

The source diagrams should remain version-controlled where practical.

---

# Documentation

Project documentation lives under:

```text
docs/
```

## Roadmap

```text
docs/roadmap.md
```

The complete learning progression.

## Setup

```text
docs/setup.md
```

Environment setup and troubleshooting.

## Glossary

```text
docs/glossary.md
```

Short explanations of GPU and AI terminology.

---

# NVIDIA CUDA

The early labs use CUDA because it exposes the GPU execution model directly.

The CUDA programming model organizes threads into thread blocks and grids. Threads within a block execute on one SM, while blocks in a grid can be scheduled across available SMs. NVIDIA GPUs organize threads within a block into 32-thread warps for SIMT execution. 

This makes CUDA a useful environment for learning the underlying execution model.

The concepts themselves are broader than CUDA.

Later articles compare other GPU architectures.

---

# Experiments, Not Just Code

A central rule of this repository:

> **Do not optimize by folklore. Measure first.**

For example, instead of saying:

> "This memory-access pattern is faster."

the lab should eventually let us measure:

```text
Pattern A
    ↓
Runtime
Bandwidth
Utilization

Pattern B
    ↓
Runtime
Bandwidth
Utilization
```

Then explain why the difference exists.

---

# Benchmarking

Performance results depend on the environment.

Relevant factors include:

- GPU architecture
- GPU clocks
- memory configuration
- driver
- CUDA version
- framework version
- input size
- data type
- batch size
- system load
- thermal conditions

Benchmark results should therefore include enough information to reproduce the experiment.

A number without context is not a benchmark.

It is trivia.

---

# Educational vs Production Results

Some experiments in this repository are deliberately tiny.

For example:

```text
element-wise operation
```

can be useful for learning GPU execution.

It does not automatically tell you how an LLM serving system will behave.

The repository separates:

```text
Concept demonstration
```

from:

```text
Production performance analysis
```

As the project progresses, experiments become increasingly realistic.

---

# Who This Is For

This project is intended for people who:

- use GPUs for AI
- use PyTorch
- build ML systems
- build LLM systems
- want to understand CUDA
- want to learn GPU performance engineering
- work with inference systems
- work with distributed AI systems
- want to understand GPU architecture from first principles

You do not need to be a hardware engineer to start.

You should be willing to run experiments and inspect what happens.

---

# Prerequisites

For Article 01:

### Required

- basic Python
- basic programming knowledge
- basic understanding of neural networks is helpful

### Helpful but not required

- PyTorch
- Linux
- CUDA
- C/C++

The first article starts from the basics.

The later articles gradually move toward advanced GPU and system concepts.

---

# Current Learning Model

The project follows:

```text
Understand
   ↓
Implement
   ↓
Run
   ↓
Measure
   ↓
Explain
   ↓
Optimize
```

Not:

```text
Copy optimization trick
   ↓
Hope benchmark improves
```

---

# Contributing

Contributions are welcome when they improve the educational value of the project.

Useful contributions include:

- correcting technical errors
- improving explanations
- adding reproducible experiments
- improving benchmark methodology
- adding architecture comparisons
- improving diagrams
- fixing bugs
- improving documentation

When proposing a performance claim, include enough information for another person to reproduce the result.

---

# License

This project is released under the MIT License.

See:

```text
LICENSE
```

---

# Citation

If this repository is useful in your work, research, article, presentation, or educational material, please cite it using:

```text
CITATION.cff
```

GitHub uses `CITATION.cff` to expose citation information for repositories that provide it.

---

# Author

**Alok Ranjan Singh**

AI Engineer

Topics covered across the project:

- AI systems
- GPU architecture
- LLMs
- AI agents
- machine learning systems
- performance engineering

---

# Project Philosophy

The project starts with a very simple question:

> **When I run AI code, what is the hardware actually doing?**

Then it keeps asking:

```text
What?
 ↓
Why?
 ↓
How?
 ↓
How fast?
 ↓
What is limiting it?
 ↓
How do we optimize it?
 ↓
How does it scale?
 ↓
How do we run it in production?
```

That is the entire journey.

---

## Start Here

**Article 01 → GPU Execution → GPU Memory → GPU Compute → Performance → AI Workloads → Multi-GPU → Production**
