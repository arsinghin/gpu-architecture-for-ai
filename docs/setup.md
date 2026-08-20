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

If you are completely new to GPU programming, start with `python/01_device_check.py` and then move through the experiments in order.

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
  - [Step 4 — Verify the NVIDIA Driver](#step-4--verify-the-nvidia-driver)
  - [Step 5 — Install the CUDA Toolkit](#step-5--install-the-cuda-toolkit)
- [Verify the Setup](#verify-the-setup)
- [Running the Labs](#running-the-labs)
  - [Python Experiments](#python-experiments)
  - [CUDA Experiments](#cuda-experiments)
- [Benchmarking Guidelines](#benchmarking-guidelines)
- [Troubleshooting](#troubleshooting)
- [Debugging CUDA Programs](#debugging-cuda-programs)
- [Recommended Development Workflow](#recommended-development-workflow)
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
| Python | 3.9 or later |
| PyTorch | Install using the [official selector](https://pytorch.org/get-started/locally/) |
| NVIDIA driver | See [Step 4](#step-4--verify-the-nvidia-driver) |
| CUDA Toolkit (`nvcc`) | Required only for the CUDA C++ labs |
| Git | For cloning the repository |

> **Note:** The current PyTorch installation documentation requires Python 3.9 or later and provides environment-specific installation commands. Always use the official PyTorch selector for the current command rather than copying an old command from an article.

### No NVIDIA GPU?

You can still read and study the repository without an NVIDIA GPU, and you can run some CPU-side Python experiments. However, the CUDA experiments require an NVIDIA CUDA-capable environment.

```text
Python explanation → can be studied without a GPU
CUDA kernel        → requires an NVIDIA CUDA environment
```

Do not confuse:

```text
PyTorch installed
```

with:

```text
CUDA-capable GPU available
```

These are separate things.

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
articles
labs
diagrams
```

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

For a CUDA-enabled NVIDIA system, select the appropriate CUDA platform shown by the current PyTorch installer. Then verify:

```bash
python -c "import torch; print(torch.__version__)"
```

The command should print the installed PyTorch version.

### Step 4 — Verify the NVIDIA Driver

For the NVIDIA CUDA experiments, the machine needs a working NVIDIA driver.

```bash
nvidia-smi
```

A working installation should display information about:

- the NVIDIA driver
- the GPU and GPU memory
- processes using the GPU
- supported CUDA version information

If `nvidia-smi` cannot communicate with the GPU, fix the driver/environment **before** troubleshooting CUDA code.

> Do not start debugging the kernel when the operating system cannot see the GPU.

### Step 5 — Install the CUDA Toolkit

The CUDA Toolkit provides development tools, including `nvcc`, which compiles CUDA C++ programs.

Check whether CUDA is installed:

```bash
nvcc --version
```

You should see the CUDA compiler version.

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

If `nvcc` cannot be found, the CUDA compiler is either not installed or is not available through your `PATH`.

## Verify the Setup

Before running performance experiments, collect the environment information:

```bash
nvidia-smi
nvcc --version
```

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

Then run the repository's device-check experiment as a smoke test:

```bash
python labs/01-gpu-execution/python/01_device_check.py
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

Full experiment details live in the [Lab 01 README](../labs/01-gpu-execution/README.md). All commands below are run from the repository root.

### Python Experiments

```bash
python labs/01-gpu-execution/python/01_device_check.py
python labs/01-gpu-execution/python/02_cpu_vs_gpu.py
python labs/01-gpu-execution/python/03_async_timing.py
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
nvcc -O2 labs/01-gpu-execution/cuda/01_hello_threads.cu     -o hello_threads
nvcc -O2 labs/01-gpu-execution/cuda/02_thread_indexing.cu   -o thread_indexing
nvcc -O2 labs/01-gpu-execution/cuda/03_warp_mapping.cu      -o warp_mapping
nvcc -O2 labs/01-gpu-execution/cuda/04_divergence.cu        -o divergence
nvcc -O2 labs/01-gpu-execution/cuda/05_device_properties.cu -o device_properties
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

Performance experiments require more care than ordinary scripts. These are the practical rules; the reasoning behind them is described in the [Benchmarking Philosophy](roadmap.md#benchmarking-philosophy) section of the roadmap.

### Warm Up

The first execution may not represent steady-state behavior. Use warm-up iterations where appropriate.

### Repeat

Do not report one timing measurement as if it were a scientific constant. Run multiple iterations and, depending on the experiment, report:

- mean
- median
- percentile
- range

### Synchronize

GPU work can execute asynchronously. For example, this:

```python
start = time.perf_counter()

result = gpu_operation()

end = time.perf_counter()
```

may not measure the complete GPU operation. Use CUDA synchronization or CUDA events where appropriate.

### Keep the Workload Fixed

When comparing two implementations, keep the important variables the same:

- input size
- data type
- batch size
- number of iterations
- GPU
- software environment

Change one important variable at a time.

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

Check:

```bash
nvidia-smi
```

If `nvidia-smi` fails, troubleshoot the NVIDIA driver first.

If `nvidia-smi` works, check:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
```

Then verify that the installed PyTorch build supports the environment. Use the official PyTorch installation selector for the correct package.

### `nvcc: command not found`

Check:

```bash
which nvcc
```

If nothing is returned, the CUDA Toolkit is not available through your `PATH`. Check whether the toolkit is installed and configure the environment appropriately for your operating system.

### CUDA program compiles but fails at runtime

Start with:

```bash
nvidia-smi
```

Then check the program's CUDA error output. For development builds, add explicit error checking after kernel launches and synchronization.

A kernel launch can return before the GPU has completed the work, so synchronization is often useful when debugging.

### `no kernel image is available for execution on the device`

This generally means the compiled code does not contain compatible device code for the GPU architecture being used.

Check the GPU's compute capability:

```bash
python -c "import torch; p=torch.cuda.get_device_properties(0); print(p.major, p.minor)"
```

Then inspect the CUDA compiler and architecture support for your installed toolkit.

### GPU program appears to hang

Possible causes include:

- invalid memory access
- synchronization problems
- incorrect indexing
- deadlock
- extremely large workloads
- an incorrectly written kernel

Start with a very small input. Add CUDA error checks. Use synchronization while debugging.

Do not immediately assume the GPU is broken.

### Output order from `printf` looks strange

GPU threads execute concurrently. Do not expect:

```text
Thread 0
Thread 1
Thread 2
Thread 3
```

to always print in exactly that order.

The ordering of printed output is not a reliable way to infer execution order.

## Debugging CUDA Programs

A useful development pattern is:

```text
Write small kernel
   ↓
Launch small workload
   ↓
Synchronize
   ↓
Check errors
   ↓
Validate output
   ↓
Increase workload
```

Do not start with:

> 100 million threads, complex shared memory, and multiple synchronization points

and then try to determine which line caused the problem. GPU debugging becomes much easier when the workload is small enough to reason about.

### Useful Environment Variable

CUDA provides environment variables that can change runtime behavior. One useful debugging option is:

```bash
CUDA_LAUNCH_BLOCKING=1 ./my_program
```

This forces CUDA kernel launches and related operations to behave synchronously from the host's perspective, which can make errors easier to locate.

Do not use synchronous execution as the default performance configuration. It is primarily a debugging aid.

## Recommended Development Workflow

For every new CUDA experiment:

1. Write the smallest possible version.
2. Compile with `nvcc -O2`.
3. Run on a tiny input.
4. Check the output.
5. Add error handling.
6. Increase the workload.
7. Add timing.
8. Repeat measurements.
9. Record the environment.
10. Only then optimize.

## Before Moving On

Before starting the memory labs, you should be comfortable with:

```text
Thread
  ↓
Warp
  ↓
Thread Block
  ↓
Grid
  ↓
SM
```

You should also understand:

```text
Kernel
  ↓
GPU execution
```

and:

```text
CPU
  ↓
launch GPU work
  ↓
GPU executes asynchronously
```

If these concepts still feel unclear, repeat [Lab 01](../labs/01-gpu-execution/README.md). The next article assumes this foundation.

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
