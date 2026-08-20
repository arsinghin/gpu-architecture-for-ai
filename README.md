<div align="center">
  
# GPU Architecture for AI

**From Python code to GPU execution, memory, compute, and production AI systems.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?logo=nvidia&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)

The hands-on companion to **The Anatomy of Silicon** article series.

</div>

---

Most AI engineers use GPUs every day. They write:

```python
output = model(input)
```

and the framework handles the rest. That is useful. But when performance drops, memory fills up, inference becomes slow, or multiple GPUs refuse to scale linearly, the abstraction starts hiding the important part.

This repository works underneath that abstraction.

> **The goal is simple: understand what actually happens inside a GPU when AI code runs.**

---

## Table of Contents

- [Why This Repository Exists](#why-this-repository-exists)
- [What This Project Covers](#what-this-project-covers)
- [Learning Path](#learning-path)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Labs](#labs)
- [Why CUDA?](#why-cuda)
- [Experiments, Not Just Code](#experiments-not-just-code)
- [Benchmarking](#benchmarking)
- [Educational vs. Production Results](#educational-vs-production-results)
- [Who This Is For](#who-this-is-for)
- [Prerequisites](#prerequisites)
- [Documentation](#documentation)
- [Project Philosophy](#project-philosophy)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Author](#author)

## Why This Repository Exists

GPU programming is often taught in disconnected pieces: CUDA, PyTorch, Tensor Cores, memory bandwidth, kernels, profiling, LLM inference. It is easy to miss how they fit together.

This repository connects those layers. Every topic follows the same method:

```
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

The articles explain the concepts. The labs make them executable. The benchmarks make the behavior measurable.

The goal is not to memorize GPU terminology. The goal is to build a mental model that lets you answer:

> **Why is this GPU workload behaving this way?**

## What This Project Covers

The learning path moves from the basics to production systems:

```
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

## Learning Path

| # | Part | Guiding Question | Status |
|:--:|------|------------------|:------:|
| 01 | [GPU Execution](01-gpu-execution/) | What happens inside a GPU when AI code runs? | ✅ Published · Lab available |
| 02 | GPU Memory | Why can a powerful GPU still be slow? | 🗓️ Planned |
| 03 | Tensor Cores and AI Compute | Why are modern GPUs so effective at AI workloads? | 🗓️ Planned |
| 04 | GPU Architecture Evolution | What actually changes between GPU generations? | 🗓️ Planned |
| 05 | GPU Architecture Beyond NVIDIA | Are all GPUs built the same way? | 🗓️ Planned |
| 06 | GPU Performance Engineering | Why is my GPU workload slow? | 🗓️ Planned |
| 07 | GPU Execution of LLMs | What actually happens when an LLM runs on a GPU? | 🗓️ Planned |
| 08 | GPU Interconnects | What happens when one GPU is not enough? | 🗓️ Planned |
| 09 | Multi-GPU Systems | How do multiple GPUs work together? | 🗓️ Planned |
| 10 | Production GPU Systems | How do we turn GPU hardware into a reliable production AI system? | 🗓️ Planned |

Parts are published in order, and each published article is accompanied by its own lab. The detailed topic breakdown for every part lives in [`docs/roadmap.md`](docs/roadmap.md).

## Getting Started

1. **Read the published articles** listed in [`docs/roadmap.md`](docs/roadmap.md). The series starts from the basics and builds up.
2. **Set up your environment** by following [`docs/setup.md`](docs/setup.md).
3. **Run the labs** for each published article, in order.

Each lab lives under `labs/` and contains numbered experiments designed to be run sequentially:

```bash
# Python experiments
python <nn-topic>/labs/python/<experiment>.py

# CUDA experiments
nvcc <nn-topic>/labs/cuda/<experiment>.cu -o experiment && ./experiment
```

Every lab has its own `README.md` describing what its experiments demonstrate and how to interpret the results.

## Repository Structure

```
gpu-architecture-for-ai/
├──  <nn-topic>/
│   ├── README.md              # Article companion (concepts, lab map)
│   ├── labs/
│   │   ├── README.md          # Lab guide
│   │   ├── python/
│   │   └── cuda/
│   └── diagrams/              # Visuals for this article
├── docs/                      # Setup guide, roadmap, and glossary
├── README.md
├── LICENSE
└── CITATION.cff
```

For every published article, a matching `<nn-topic>/` directory is added using the same convention. Each article directory contains its companion `README.md`, `labs/`, and `diagrams/` when the corresponding material exists.

| Directory              | Purpose                                                                                                       |
| ---------------------- | ------------------------------------------------------------------------------------------------------------- |
| `<nn-topic>/`          | Article-specific companion material, including concepts, learning objectives, labs, diagrams, and references. |
| `<nn-topic>/labs/`     | Runnable experiments associated with the article.                                                             |
| `<nn-topic>/diagrams/` | Version-controlled diagrams associated with the article.                                                      |
| `docs/`                | Project documentation: roadmap, setup guide, and glossary.                                                    |

## Labs

The `labs/` directory contains runnable experiments. A lab exists to answer one question:

> Can we demonstrate this concept with actual code?

Labs are intentionally small at the beginning, just enough to make a concept observable, and become more performance-oriented as the series progresses. A typical lab contains:

- **Python experiments:** observing GPU behavior from the framework level using PyTorch and standard tooling.
- **CUDA experiments:** exposing the underlying execution model directly, including threads, warps, blocks, scheduling, divergence, and hardware properties.

## Why CUDA?

The early labs use CUDA because it exposes the GPU execution model directly. The CUDA programming model organizes threads into thread blocks and grids: threads within a block execute on one SM, blocks in a grid can be scheduled across available SMs, and threads within a block are organized into 32-thread warps for SIMT execution.

This makes CUDA a useful environment for learning the underlying execution model. However, the concepts themselves are broader than CUDA. Later articles compare other GPU architectures, including NVIDIA, AMD, Intel, and Apple.

## Experiments, Not Just Code

A central rule of this repository:

> **Do not optimize by folklore. Measure first.**

Instead of claiming "this memory-access pattern is faster," a lab lets you measure:

```
Pattern A → runtime · bandwidth · utilization
Pattern B → runtime · bandwidth · utilization
```

and then explains *why* the difference exists.

## Benchmarking

Performance results depend on the environment. Relevant factors include:

- GPU architecture, clocks, and memory configuration
- Driver, CUDA, and framework versions
- Input size, data type, and batch size
- System load and thermal conditions

Benchmark results should therefore include enough information for another person to reproduce the experiment.

> A number without context is not a benchmark. It is trivia.

## Educational vs. Production Results

Some experiments in this repository are deliberately tiny. An element-wise operation is useful for learning GPU execution, but it does not tell you how an LLM serving system will behave.

The repository therefore separates **concept demonstration** from **production performance analysis**. As the project progresses, experiments become increasingly realistic.

## Who This Is For

This project is intended for people who:

- Use GPUs for AI and PyTorch
- Build ML, LLM, or inference systems
- Work with distributed AI systems
- Want to understand CUDA or GPU performance engineering
- Want to understand GPU architecture from first principles

You do not need to be a hardware engineer to start. You should be willing to run experiments and inspect what happens.

## Prerequisites

**Required:**

- Basic Python
- Basic programming knowledge

**Helpful but not required:**

- Basic understanding of neural networks
- PyTorch
- Linux, CUDA, C/C++

The series starts from the basics and gradually moves toward advanced GPU and system concepts.

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/roadmap.md`](docs/roadmap.md) | The complete learning progression, including detailed topics for each part |
| [`docs/setup.md`](docs/setup.md) | Environment setup and troubleshooting |
| [`docs/glossary.md`](docs/glossary.md) | Short explanations of GPU and AI terminology |

## Project Philosophy

The project starts with a very simple question:

> When I run AI code, what is the hardware actually doing?

Then it keeps asking:

```
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

Every topic follows the learning model:

```
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

```
Copy optimization trick
 ↓
Hope benchmark improves
```

That is the entire journey.

## Contributing

Contributions are welcome when they improve the educational value of the project. Useful contributions include:

- Correcting technical errors
- Improving explanations and documentation
- Adding reproducible experiments
- Improving benchmark methodology
- Adding architecture comparisons
- Improving diagrams
- Fixing bugs

When proposing a performance claim, include enough information for another person to reproduce the result.

## License

This project is released under the [MIT License](LICENSE).

## Citation

If this repository is useful in your work, research, article, presentation, or educational material, please cite it using [`CITATION.cff`](CITATION.cff). GitHub uses `CITATION.cff` to expose citation information for repositories that provide it.

## Author

**Alok Ranjan Singh** · AI Engineer

Topics covered across the project: AI systems · GPU architecture · LLMs · AI agents · machine learning systems · performance engineering
