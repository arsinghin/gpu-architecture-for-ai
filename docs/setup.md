<div align="center">

# Setup Guide

**GPU Architecture for AI**

Set up the environment, verify your GPU, and run the first experiments.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?logo=nvidia&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey)

Part of [GPU Architecture for AI](../README.md) · [Roadmap](roadmap.md) · [Glossary](glossary.md)

</div>

---

## Overview

This repository contains two broad types of experiments:

```text
Python / PyTorch
   ↓
Higher-level GPU programming

CUDA C++
   ↓
Lower-level GPU execution model
```

The Python experiments are easier to start with. The CUDA experiments expose the execution hierarchy more directly.

The repository starts with NVIDIA CUDA because the first labs expose GPU execution concepts directly through CUDA. Later labs may introduce other GPU programming environments.

If you are completely new to GPU programming, start with the first experiment in [Lab 01](../01-gpu-execution/labs/README.md) and move through the experiments in order.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
  - [Hardware](#hardware)
  - [Software](#software)
  - [No NVIDIA GPU?](#no-nvidia-gpu)
- [Installation](#installation)
  - [Step 1 — Clone the Repository](#step-1--clone-the-repository)
  - [Step 2 — Create a Python Environment](#step-2--create-a-python-environment)
  - [Step 3 — Install PyTorch](#step-3--install-pytorch)
- [Verify the Setup](#verify-the-setup)
- [Running the Labs](#running-the-labs)
- [Benchmarking Guidelines](#benchmarking-guidelines)
- [Troubleshooting](#troubleshooting)
- [Debugging and Development Workflow](#debugging-and-development-workflow)
- [Before Moving On](#before-moving-on)
- [Next Step](#next-step)

## Requirements

### Hardware

| Requirement | Notes |
|-------------|-------|
| NVIDIA GPU | CUDA-capable; a consumer GPU is sufficient for the introductory labs |
| Working NVIDIA driver | Required for all GPU experiments |
| Sufficient GPU memory | Only what the specific experiment needs |

The early experiments are intentionally small. You do not need a data-center GPU to understand threads, blocks, warps, or basic kernels.

### Software

| Requirement | Notes |
|-------------|-------|
| Operating system | Linux, Windows, or another supported development environment |
| Python | 3.9 or later (current PyTorch requirement) |
| PyTorch | Install using the [official selector](https://pytorch.org/get-started/locally/) — never copy an old command |
| NVIDIA driver | See [Verify the Setup](#verify-the-setup) |
| CUDA Toolkit (`nvcc`) | Required only for the CUDA C++ labs |
| Git | For cloning the repository |

### No NVIDIA GPU?

You can still read and study the repository without an NVIDIA GPU, and you can run some CPU-side Python experiments. However, the CUDA experiments require an NVIDIA CUDA-capable environment.

Do not confuse *PyTorch installed* with *a CUDA-capable GPU available* — these are separate things.

## Installation

### Step 1 — Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd gpu-architecture-for-ai
```

Check the structure:

```bash
ls
```

You should see something similar to:

```text
README.md
LICENSE
CITATION.cff
docs
01-gpu-execution
```

The repository is organized by article: one numbered folder per published part, each containing its labs and diagrams. More folders appear as articles are published.

### Step 2 — Create a Python Environment

A virtual environment is strongly recommended.

```bash
python3 -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Verify Python:

```bash
python --version
```

You should have Python 3.9 or newer.

### Step 3 — Install PyTorch

Do not blindly copy an old PyTorch installation command from this repository. PyTorch publishes different installation commands depending on:

- operating system
- package manager
- Python environment
- compute platform

Use the official selector: <https://pytorch.org/get-started/locally/>

For a CUDA-enabled NVIDIA system, select the appropriate CUDA platform shown by the current PyTorch installer, then verify:

```bash
python -c "import torch; print(torch.__version__)"
```

## Verify the Setup

After installation, confirm each layer of the stack — driver, toolkit, framework, GPU — before running experiments.

### NVIDIA Driver

```bash
nvidia-smi
```

A working installation should display the driver version, the GPU and its memory, processes using the GPU, and supported CUDA version information.

If `nvidia-smi` cannot communicate with the GPU, fix the driver/environment **before** troubleshooting CUDA code.

> Do not start debugging the kernel when the operating system cannot see the GPU.

### CUDA Toolkit

The CUDA Toolkit provides development tools, including `nvcc`, which compiles CUDA C++ programs.

```bash
nvcc --version
```

The toolkit version and the driver version are related but are not the same thing. Do not assume that:

```text
CUDA Toolkit version = CUDA driver version = PyTorch CUDA build
```

They represent different parts of the software stack.

Locate `nvcc` on Linux/macOS:

```bash
which nvcc
```

On Windows:

```powershell
where nvcc
```

If `nvcc` cannot be found, the compiler is either not installed or not available through your `PATH`.

### PyTorch and CUDA

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

If the last command reports `CUDA unavailable`, see [Troubleshooting](#troubleshooting).

### Smoke Test

Run the repository's device-check experiment:

```bash
python 01-gpu-execution/labs/python/01_device_check.py
```

Expected output will resemble:

```text
PyTorch version: ...
CUDA available: True
CUDA device count: 1
Current device: 0
GPU name: NVIDIA ...
Compute capability: ...
GPU memory: ... GiB
```

Exact values depend on your system. Record these values when publishing benchmark results — see [Benchmarking Guidelines](#benchmarking-guidelines).

## Running the Labs

Full experiment details live in the [Lab 01 README](../01-gpu-execution/labs/README.md). All commands below are run from the repository root. The same pattern applies to every future part: `<nn-topic>/labs/`.

### Python Experiments

```bash
python 01-gpu-execution/labs/python/01_device_check.py
python 01-gpu-execution/labs/python/02_cpu_vs_gpu.py
python 01-gpu-execution/labs/python/03_async_timing.py
```

The experiments should be run in this order. They introduce:

```text
Device
  ↓
CPU vs GPU
  ↓
GPU timing
```

before moving to lower-level CUDA.

### CUDA Experiments

Compile all five CUDA experiments:

```bash
nvcc -O2 01-gpu-execution/labs/cuda/01_hello_threads.cu     -o hello_threads
nvcc -O2 01-gpu-execution/labs/cuda/02_thread_indexing.cu   -o thread_indexing
nvcc -O2 01-gpu-execution/labs/cuda/03_warp_mapping.cu      -o warp_mapping
nvcc -O2 01-gpu-execution/labs/cuda/04_divergence.cu        -o divergence
nvcc -O2 01-gpu-execution/labs/cuda/05_device_properties.cu -o device_properties
```

Then run each binary:

```bash
./hello_threads
./thread_indexing
./warp_mapping
./divergence
./device_properties
```

Run them in this order:

| # | Experiment | Question It Answers |
|:--:|------------|---------------------|
| 1 | `01_hello_threads` | What is a thread? |
| 2 | `02_thread_indexing` | Where does the thread get its identity? |
| 3 | `03_warp_mapping` | How do threads become warps? |
| 4 | `04_divergence` | What happens when execution paths differ? |
| 5 | `05_device_properties` | What hardware is actually available? |

The progression is intentional.

## Benchmarking Guidelines

The reasoning behind these rules lives in the [Benchmarking Philosophy](roadmap.md#benchmarking-philosophy) section of the roadmap. The practical rules:

- **Warm up.** The first execution rarely represents steady-state behavior.
- **Repeat.** Do not report one timing measurement as if it were a scientific constant — report mean, median, percentile, or range over multiple iterations.
- **Synchronize.** GPU work executes asynchronously, so naive host timing can measure the wrong thing entirely:

  ```python
  start = time.perf_counter()
  result = gpu_operation()   # returns before the GPU finishes
  end = time.perf_counter()
  ```

  Use CUDA synchronization or CUDA events.

- **Keep the workload fixed.** When comparing implementations, change one variable at a time — same input size, data type, batch size, GPU, and software environment.
- **Label educational results.** Tiny benchmarks demonstrate concepts; they do not predict production performance.

### Record the Environment

When publishing benchmark results, record:

```text
GPU:
GPU memory:
Driver:
CUDA Toolkit:
PyTorch:
Python:
Operating system:
Precision:
Input shape:
Batch size:
Iterations:
Warm-up iterations:
```

Example:

```text
GPU: NVIDIA ...
GPU memory: ...
Driver: ...
CUDA Toolkit: ...
PyTorch: ...
Python: ...
OS: ...
Precision: FP32
Input shape: ...
Batch size: ...
Warm-up: 20
Iterations: 100
```

> A benchmark without environment information is difficult to reproduce.

## Troubleshooting

### `torch.cuda.is_available()` returns `False`

Run `nvidia-smi` first. If it fails, troubleshoot the driver — not PyTorch.

If the driver works, check your PyTorch build's CUDA version against your environment (see [Verify the Setup](#verify-the-setup)). A CPU-only PyTorch build is the most common cause — reinstall using the [official selector](https://pytorch.org/get-started/locally/).

### `nvcc: command not found`

Run `which nvcc` (Linux/macOS) or `where nvcc` (Windows). If nothing is returned, the CUDA Toolkit is either not installed or not on your `PATH` — configure your environment for your operating system.

### CUDA program compiles but fails at runtime

Start with `nvidia-smi`, then check the program's CUDA error output. A kernel launch can return before the GPU completes the work, so add explicit error checking and synchronization when debugging — see [Debugging and Development Workflow](#debugging-and-development-workflow).

### `no kernel image is available for execution on the device`

The compiled code does not contain device code compatible with your GPU architecture. Check your compute capability:

```bash
python -c "import torch; p=torch.cuda.get_device_properties(0); print(p.major, p.minor)"
```

Then verify that your CUDA Toolkit supports that architecture.

### GPU program appears to hang

Possible causes include invalid memory access, synchronization problems, incorrect indexing, deadlock, or an extremely large workload. Start with a very small input, add CUDA error checks, and synchronize while debugging.

Do not immediately assume the GPU is broken.

### Output order from `printf` looks strange

GPU threads execute concurrently. Do not expect threads 0, 1, 2, 3 to print in exactly that order — output ordering is not a reliable way to infer execution order.

## Debugging and Development Workflow

GPU debugging becomes much easier when the workload is small enough to reason about. Do not start with 100 million threads, complex shared memory, and multiple synchronization points — and then try to determine which line caused the problem.

For every new CUDA experiment:

```text
Write the smallest possible version
   ↓
Compile with nvcc -O2 → run on a tiny input
   ↓
Check the output · add error handling
   ↓
Synchronize · validate results
   ↓
Increase the workload
   ↓
Add timing · repeat measurements
   ↓
Record the environment
   ↓
Only then optimize
```

### Useful Environment Variable

A common debugging aid:

```bash
CUDA_LAUNCH_BLOCKING=1 ./my_program
```

This forces CUDA kernel launches to behave synchronously from the host's perspective, which can make errors easier to locate. It is a debugging aid only — do not use it as a default performance configuration.

## Before Moving On

Before starting the memory labs, you should be comfortable with:

```text
Thread → Warp → Thread Block → Grid → SM
```

You should also understand that a **kernel launch** creates this hierarchy, and that the CPU *launches* GPU work while the GPU *executes* it asynchronously.

If these concepts still feel unclear, repeat [Lab 01](../01-gpu-execution/labs/README.md). The next article assumes this foundation.

## Next Step

After completing Lab 01, the focus changes from:

> **How does the GPU execute work?**

to:

> **How does the GPU get the data needed to execute that work?**

That is the subject of [Article 02 — GPU Memory](roadmap.md#part-02--gpu-memory) (planned), where bandwidth, latency, cache behavior, coalescing, tiling, and the memory wall become important.

---

<p align="center">
<sub>
<a href="../README.md">Main README</a> ·
<a href="roadmap.md">Roadmap</a> ·
<a href="glossary.md">Glossary</a>
</sub>
</p>
