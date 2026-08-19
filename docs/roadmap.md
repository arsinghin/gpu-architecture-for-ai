# GPU Architecture for AI — Roadmap

> A hands-on journey from Python code to GPU silicon, from basic execution to production AI systems.

This repository is the practical companion to the **The Anatomy of Silicon** article series.

The goal is not to memorize GPU terminology.

The goal is to understand **why GPUs behave the way they do**, measure that behavior with real experiments, and eventually use that knowledge to reason about production AI workloads.

---

# Table of Contents

- [Project Goal](#project-goal)
- [How to Use This Repository](#how-to-use-this-repository)
- [Learning Philosophy](#learning-philosophy)
- [Learning Path](#learning-path)
- [Part I — GPU Execution](#part-i--gpu-execution)
- [Part II — GPU Memory](#part-ii--gpu-memory)
- [Part III — GPU Computation](#part-iii--gpu-computation)
- [Part IV — GPU Architectures](#part-iv--gpu-architectures)
- [Part V — Alternative GPU Architectures](#part-v--alternative-gpu-architectures)
- [Part VI — GPU Performance Engineering](#part-vi--gpu-performance-engineering)
- [Part VII — AI and LLM Workloads](#part-vii--ai-and-llm-workloads)
- [Part VIII — GPU Interconnects](#part-viii--gpu-interconnects)
- [Part IX — Multi-GPU Systems](#part-ix--multi-gpu-systems)
- [Part X — Production GPU Systems](#part-x--production-gpu-systems)
- [Repository Structure](#repository-structure)
- [Article → Lab → Experiment](#article--lab--experiment)
- [Benchmarking Philosophy](#benchmarking-philosophy)
- [Hardware Coverage](#hardware-coverage)
- [What This Project Is Not](#what-this-project-is-not)
- [How the Roadmap Will Evolve](#how-the-roadmap-will-evolve)

---

# Project Goal

Modern AI systems depend heavily on GPUs, but many AI engineers learn to use GPUs without ever building a mental model of what happens underneath a PyTorch operation.

You write:

```python
output = model(input)
````

and the framework takes care of the rest.

That is convenient.

It is also where many performance problems become mysterious.

This project works backward from that line.

The learning path is:

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

The goal is to make this entire path understandable.

# How to Use This Repository

The repository has three connected layers:

```text
Medium Articles
      ↓
Explain the concept
      ↓
GitHub Labs
      ↓
Run the concept
      ↓
Benchmarks
      ↓
Measure the behavior
```

The articles focus on explanation and mental models.

The labs contain runnable code.

The benchmarks measure what happens when the code actually runs.

You do not need to read every article before touching the code.

However, the recommended path is:

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

If you can run a benchmark but cannot explain why the result changed, you have learned how to execute a script, not how the GPU works.

# Learning Philosophy

This project follows a simple progression:

```text
Beginner
   ↓
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

Each stage answers a different question.

### Beginner

What is happening?

### Intermediate

Why is it happening?

### Advanced

How does the hardware make it happen?

### Performance Engineering

What is limiting it?

### Production Engineering

How do I design the entire system around those limits?

The project deliberately avoids jumping directly into advanced GPU optimization.

You cannot meaningfully optimize something you cannot mentally trace.

# Learning Path

The complete roadmap is divided into nine major areas.

```text
01. GPU Execution
        ↓
02. GPU Memory
        ↓
03. GPU Computation
        ↓
04. GPU Architectures
        ↓
05. GPU Performance
        ↓
06. AI / LLM Workloads
        ↓
07. GPU Interconnects
        ↓
08. Multi-GPU Systems
        ↓
09. Production GPU Systems
```

Each area builds on the previous one.

# Part I — GPU Execution

## Article 01 — How GPUs Actually Execute AI Workloads

### Core question

What happens inside a GPU when you run a neural network?

### Concepts

* CPU vs GPU
* Throughput vs latency
* SIMT
* Threads
* Warps
* Thread blocks
* Grids
* SMs
* Warp schedulers
* Registers
* Shared memory
* L1
* L2
* Kernels
* Latency hiding
* Occupancy
* Divergence
* PyTorch execution
* Asynchronous execution

### Lab

```text
labs/01-gpu-execution/
```

### Experiments

* Detect the CUDA device
* CPU vs GPU execution
* Correct GPU timing
* Launch a CUDA kernel
* Inspect thread indexing
* Inspect warp and lane mapping
* Observe divergent control flow
* Inspect GPU hardware properties

### Reader takeaway

The reader should be able to mentally trace:

```text
Python
  ↓
PyTorch
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
Execution
```

# Part II — GPU Memory

## Article 02 — GPU Memory Explained: HBM, SRAM, Cache and the Memory Wall

### Core question

Why can a GPU with enormous compute power still be slow?

### Concepts

* Registers
* SRAM
* Shared memory
* L1 cache
* L2 cache
* GPU device memory
* HBM
* GDDR
* Host RAM
* Memory capacity
* Memory latency
* Memory bandwidth
* Memory transactions
* Coalescing
* Strided access
* Tiling
* Data reuse
* Arithmetic intensity
* Roofline model
* Compute-bound workloads
* Memory-bound workloads

### Lab

```text
labs/02-gpu-memory/
```

### Experiments

* Sequential memory access
* Strided memory access
* Coalesced vs non-coalesced access
* Shared-memory reuse
* Tiled matrix multiplication
* Different tile sizes
* Effective bandwidth measurement
* Arithmetic intensity experiments

### Killer demonstration

Run mathematically similar operations with different memory-access patterns.

Measure:

* Execution time
* Effective bandwidth
* Memory traffic
* Slowdown

The purpose is to turn:

> "Memory access matters"

into:

> "Here is the measurement showing exactly why."

### Reader takeaway

The reader should understand:

> A GPU can have enormous compute capacity and still be limited by moving data.

# Part III — GPU Computation

## Article 03 — Tensor Cores and the Hardware Behind AI Math

### Core question

Why are modern GPUs so good at matrix-heavy AI workloads?

### Concepts

* Matrix multiplication
* GEMM
* Fused operations
* FP32
* FP16
* BF16
* TF32
* FP8
* FP4
* Tensor Cores
* Matrix multiply-accumulate
* Precision vs performance
* Accumulation precision
* Quantization
* Sparsity

### Lab

```text
labs/03-tensor-cores/
```

### Experiments

* Matrix multiplication in FP32
* Matrix multiplication in lower precision
* Tensor Core enabled workloads
* Different matrix sizes
* Different batch sizes
* Precision comparison
* Throughput comparison

### Reader takeaway

The reader should understand why AI workloads are not simply:

> "Run lots of FP32 instructions."

Modern AI hardware uses specialized matrix-processing paths and lower numerical precision to achieve much higher throughput.

# Part IV — GPU Architectures

## Article 04 — How NVIDIA GPU Architectures Evolved

### Core question

What actually changes from one GPU generation to the next?

### Architectures

* Volta
* Turing
* Ampere
* Ada Lovelace
* Blackwell
* Future architectures as appropriate

### Concepts

* SM evolution
* Tensor Core evolution
* Memory subsystem changes
* Cache changes
* Precision support
* Scheduling changes
* Specialized accelerators
* Interconnect evolution
* Packaging

### Lab

```text
labs/04-nvidia-architectures/
```

### Goal

Do not memorize specifications.

Understand what changed and why those changes matter for workloads.

# Part V — Alternative GPU Architectures

## Article 05 — NVIDIA vs AMD vs Intel vs Apple GPUs

### Core question

Are all GPUs built the same way?

No.

### Architectures

* NVIDIA
* AMD
* Intel
* Apple

### Concepts

* Different execution models
* Different terminology
* Matrix acceleration
* Memory systems
* Unified memory
* Packaging
* Software ecosystems
* AI acceleration

### Important rule

Do not force different architectures into NVIDIA terminology.

For example:

```text
NVIDIA → SM
AMD → Compute Units / Workgroup Processors
Intel → Xe architecture
Apple → Apple GPU architecture
```

These terms are related conceptually but are not interchangeable.

# Part VI — GPU Performance Engineering

## Article 06 — Why Your GPU Is Slow

### Core question

If the GPU is powerful, what is actually limiting performance?

### Concepts

* GPU utilization
* Occupancy
* Memory throughput
* Compute throughput
* Kernel launch overhead
* Synchronization
* Register pressure
* Shared-memory usage
* Cache behavior
* Warp divergence
* Kernel fusion
* Arithmetic intensity
* Roofline analysis
* Profiling

### Tools

* PyTorch Profiler
* NVIDIA Nsight Systems
* NVIDIA Nsight Compute
* CUDA events
* CUDA runtime APIs

### Lab

```text
labs/06-gpu-performance/
```

### Experiments

Compare:

```text
Naive kernel
      ↓
Optimized kernel
```

Measure:

* Latency
* Throughput
* Achieved bandwidth
* Utilization
* Occupancy
* Instruction behavior

The objective is to answer:

> What is the bottleneck?

rather than:

> Which optimization trick should I try next?

# Part VII — AI and LLM Workloads

## Article 07 — What Actually Happens When an LLM Runs on a GPU?

### Core question

How does transformer inference turn into GPU work?

### Concepts

* Transformer layers
* Embeddings
* Linear layers
* Matrix multiplication
* Attention
* Softmax
* KV cache
* Prefill
* Decode
* Batch size
* Sequence length
* Memory bandwidth
* Compute throughput
* Quantization

### Lab

```text
labs/07-llm-workloads/
```

### Experiments

Measure:

* Prefill latency
* Decode latency
* Tokens/sec
* Memory usage
* Batch-size scaling
* Sequence-length scaling

Compare:

```text
Batch 1
Batch 2
Batch 4
Batch 8
...
```

The goal is to connect low-level GPU behavior with real LLM serving behavior.

# Part VIII — GPU Interconnects

## Article 08 — How GPUs Talk to Each Other

### Core question

What happens when one GPU is not enough?

### Concepts

* PCIe
* NVLink
* NVSwitch
* Host-to-device communication
* Device-to-device communication
* GPU topology
* Communication bandwidth
* Communication latency
* Collective communication

### Lab

```text
labs/08-gpu-interconnects/
```

### Experiments

Measure:

* CPU → GPU
* GPU → CPU
* GPU → GPU

Then compare the effect of topology and communication path where the hardware supports it.

# Part IX — Multi-GPU Systems

## Article 09 — Scaling AI Across Multiple GPUs

### Core question

How do we turn multiple GPUs into one useful compute system?

### Concepts

* Data parallelism
* Tensor parallelism
* Pipeline parallelism
* Model parallelism
* Distributed training
* All-reduce
* All-gather
* Communication overhead
* Scaling efficiency

### Lab

```text
labs/09-multi-gpu/
```

### Experiments

Compare:

```text
1 GPU
2 GPUs
4 GPUs
```

Measure:

* Throughput
* Scaling efficiency
* Communication overhead
* Synchronization overhead

The important lesson is:

> More GPUs do not automatically mean proportionally more performance.

# Part X — Production GPU Systems

## Article 10 — Designing Production GPU Systems

### Core question

How do we turn GPU hardware into a reliable production AI system?

### Concepts

* Model serving
* Batching
* Dynamic batching
* Concurrency
* Latency
* Throughput
* GPU utilization
* Memory fragmentation
* CUDA Graphs
* Kernel fusion
* Quantization
* Model parallelism
* Monitoring
* Profiling
* Cost per request
* Cost per token
* Capacity planning

### Lab

```text
labs/10-production-gpu/
```

### Goal

Move from:

```text
GPU optimization
```

to:

```text
System optimization
```

A production system is not successful because one kernel is fast.

It is successful when the entire system meets its latency, throughput, reliability, and cost requirements.

# Repository Structure

The repository grows together with the article series.

The intended long-term structure is:

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

This is the target structure.

It does not mean every directory needs to exist immediately.

Directories should be added when their corresponding article or experiment exists.

# Article → Lab → Experiment

Every article should eventually have this relationship:

```text
Article
   │
   │ explains
   ↓
Concept
   │
   │ implemented by
   ↓
Lab
   │
   │ measured by
   ↓
Experiment
   │
   │ produces
   ↓
Result
```

For example:

### Article 01

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

### Article 02

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

# Benchmarking Philosophy

The repository will eventually contain many performance measurements.

Those measurements should follow a few rules.

## 1. Never trust a single run

GPU execution is affected by:

* Warm-up
* Clocks
* Background activity
* Cache state
* System load
* Driver behavior
* Thermal conditions

Use repeated measurements where appropriate.

## 2. Warm up GPU workloads

The first execution may not represent steady-state behavior.

Where appropriate:

```text
Warm-up
   ↓
Measurement
   ↓
Repeated runs
   ↓
Summary statistic
```

## 3. Synchronize correctly

GPU operations can execute asynchronously.

Do not assume:

```text
start()
gpu_operation()
end()
```

automatically measures GPU execution time.

Use CUDA-aware timing or explicit synchronization where appropriate.

## 4. Record the environment

A benchmark result should record enough information to understand its context.

At minimum:

* GPU
* GPU memory
* Driver
* CUDA
* PyTorch
* Python
* Operating system
* Precision
* Input size
* Batch size
* Relevant configuration

## 5. Separate educational benchmarks from production benchmarks

A tiny benchmark can explain a concept.

It does not automatically represent production performance.

For example:

```text
10 million element-wise operations
```

does not tell us how an entire LLM serving system behaves.

The repository will clearly label simplified experiments as educational.

# Hardware Coverage

The project focuses heavily on NVIDIA CUDA initially because CUDA provides a mature environment for exposing GPU execution concepts.

That does not mean the concepts only apply to NVIDIA GPUs.

The project distinguishes between:

### General GPU concepts

Examples:

* Parallel execution
* Memory hierarchy
* Bandwidth
* Latency
* Data reuse
* Matrix computation
* GPU utilization

and:

### Vendor-specific implementation details

Examples:

* CUDA
* SM
* CUDA warps
* Tensor Cores
* NVLink
* Nsight

Later parts of the project will introduce AMD, Intel, and Apple architectures separately.

# CUDA Is the Initial Learning Environment

CUDA is used extensively in the early labs because it provides direct access to concepts such as:

* `threadIdx`
* `blockIdx`
* `blockDim`
* `gridDim`
* Warp
* Shared memory
* Kernel launch

These make the execution model concrete.

The current CUDA programming model organizes work into grids, thread blocks, and 32-thread warps, with thread blocks executing on individual SMs.

Later, where useful, equivalent concepts can be explored using other GPU programming environments.

# What This Project Is Not

This repository is not intended to be:

* A complete CUDA reference
* An NVIDIA architecture specification database
* A collection of random CUDA examples
* A collection of benchmark numbers without explanation
* A replacement for official GPU documentation
* A generic deep-learning tutorial
* A collection of optimization tricks without reasoning

The project is specifically about building a mental model of GPU-based AI systems.

# The Progression

The entire project should eventually answer these questions:

### Level 1

What is a GPU?

### Level 2

How does a GPU execute thousands of pieces of work?

### Level 3

How do threads, warps, blocks, and SMs interact?

### Level 4

Where does the data live?

### Level 5

Why does memory access affect performance?

### Level 6

How does a GPU accelerate matrix math?

### Level 7

Why do modern AI models benefit from specialized hardware?

### Level 8

Why does my GPU workload become slow?

### Level 9

How does an LLM use the GPU?

### Level 10

How do multiple GPUs communicate?

### Level 11

How do we scale AI across GPUs?

### Level 12

How do we turn all of this into a production system?

That is the destination.

# The Final Mental Model

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

# How the Roadmap Will Evolve

The roadmap is intentionally not frozen.

GPU architectures, software stacks, numerical formats, interconnects, and AI workloads change quickly.

Future articles may therefore:

* Add new GPU architectures
* Replace outdated examples
* Introduce new numerical formats
* Add new profiling techniques
* Add new benchmark environments
* Add new AI workload experiments
* Expand multi-GPU experiments

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

Do not skip the fundamentals.

The fastest way to understand advanced GPU systems is not to start with the most advanced GPU.

It is to understand the simple execution model well enough that the advanced hardware becomes an extension of an existing mental model.

```
