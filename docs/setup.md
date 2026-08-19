# GPU Architecture for AI — Setup Guide

> Set up the environment, verify your GPU, and run the first experiments.

This guide prepares your machine for the hands-on labs in this repository.

The repository starts with NVIDIA CUDA because the first labs expose GPU execution concepts directly through CUDA.

Later labs may introduce other GPU programming environments.

---

# Table of Contents

* [Before You Start](#before-you-start)
* [Hardware Requirements](#hardware-requirements)
* [Software Requirements](#software-requirements)
* [1. Clone the Repository](#1-clone-the-repository)
* [2. Python Environment](#2-python-environment)
* [3. Install PyTorch](#3-install-pytorch)
* [4. Verify PyTorch](#4-verify-pytorch)
* [5. NVIDIA Driver](#5-nvidia-driver)
* [6. CUDA Toolkit](#6-cuda-toolkit)
* [7. Verify nvcc](#7-verify-nvcc)
* [8. Run the Python Labs](#8-run-the-python-labs)
* [9. Compile CUDA Labs](#9-compile-cuda-labs)
* [10. Run CUDA Labs](#10-run-cuda-labs)
* [11. Recommended Environment Check](#11-recommended-environment-check)
* [12. CPU-Only Machines](#12-cpu-only-machines)
* [13. Benchmarking Rules](#13-benchmarking-rules)
* [14. Reproducibility](#14-reproducibility)
* [15. Common Problems](#15-common-problems)
* [16. Debugging CUDA Programs](#16-debugging-cuda-programs)
* [17. Environment Variables](#17-environment-variables)
* [18. Recommended Development Workflow](#18-recommended-development-workflow)

---

# Before You Start

This repository contains two broad types of experiments:

```text
Python / PyTorch
       ↓
Higher-level GPU programming

CUDA C++
       ↓
Lower-level GPU execution model
```

The Python experiments are easier to start with.

The CUDA experiments expose the execution hierarchy more directly.

If you are completely new to GPU programming, start with:

```text
python/01_device_check.py
```

and then move through the experiments in order.

# Hardware Requirements

## Recommended

For the CUDA labs:

* NVIDIA GPU
* Working NVIDIA driver
* CUDA-capable GPU
* Sufficient GPU memory for the experiment being run

The early experiments are intentionally small.

You do not need a data-center GPU to understand threads, blocks, warps, or basic kernels.

A consumer NVIDIA GPU is enough for most introductory experiments.

# Software Requirements

Recommended baseline:

* Linux, Windows, or another supported development environment
* Python 3.9+
* PyTorch
* NVIDIA driver
* CUDA Toolkit for CUDA C++ labs
* `nvcc`
* Git

The current PyTorch installation documentation requires Python 3.9 or later and provides environment-specific installation commands. Always use the official PyTorch selector for the current installation command rather than copying an old command from an article.

Official PyTorch installation page:

https://pytorch.org/get-started/locally/

# 1. Clone the Repository

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
```

Enter the repository:

```bash
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

# 2. Python Environment

A virtual environment is strongly recommended.

Create one:

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

# 3. Install PyTorch

Do not blindly copy an old PyTorch installation command from this repository.

PyTorch publishes different installation commands depending on:

* Operating system
* Package manager
* Python environment
* Compute platform

Use the official selector:

https://pytorch.org/get-started/locally/

For a CUDA-enabled NVIDIA system, select the appropriate CUDA platform shown by the current PyTorch installer.

Then verify:

```bash
python -c "import torch; print(torch.__version__)"
```

The command should print the installed PyTorch version.

# 4. Verify PyTorch

Run:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

If CUDA is available, also run:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

You should see your NVIDIA GPU name.

You can also run the repository experiment:

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

Exact values depend on your system.

# 5. NVIDIA Driver

For NVIDIA CUDA experiments, the machine needs a working NVIDIA driver.

Run:

```bash
nvidia-smi
```

A working installation should display information about:

* NVIDIA driver
* GPU
* GPU memory
* Processes using the GPU
* Supported CUDA version information

If `nvidia-smi` cannot communicate with the GPU, fix the driver/environment before troubleshooting CUDA code.

Do not start debugging the kernel when the operating system cannot see the GPU.

# 6. CUDA Toolkit

The CUDA Toolkit provides development tools including:

```text
nvcc
```

which compiles CUDA C++ programs.

Check whether CUDA is installed:

```bash
nvcc --version
```

You should see the CUDA compiler version.

The toolkit version and the driver version are related but are not the same thing.

Do not assume that:

```text
CUDA Toolkit version
=
CUDA driver version
=
PyTorch CUDA build
```

They represent different parts of the software stack.

# 7. Verify nvcc

Run:

```bash
which nvcc
```

on Linux/macOS.

On Windows:

```powershell
where nvcc
```

Then:

```bash
nvcc --version
```

If `nvcc` cannot be found, the CUDA compiler is either not installed or is not available through your `PATH`.

# 8. Run the Python Labs

From the repository root:

```bash
python labs/01-gpu-execution/python/01_device_check.py
```

Then:

```bash
python labs/01-gpu-execution/python/02_cpu_vs_gpu.py
```

Then:

```bash
python labs/01-gpu-execution/python/03_async_timing.py
```

The experiments should be run in this order.

They introduce:

```text
Device
  ↓
CPU vs GPU
  ↓
GPU timing
```

before moving to lower-level CUDA.

# 9. Compile CUDA Labs

Move to the repository root.

Compile the first CUDA experiment:

```bash
nvcc -O2 \
  labs/01-gpu-execution/cuda/01_hello_threads.cu \
  -o hello_threads
```

Run:

```bash
./hello_threads
```

Compile thread indexing:

```bash
nvcc -O2 \
  labs/01-gpu-execution/cuda/02_thread_indexing.cu \
  -o thread_indexing
```

Run:

```bash
./thread_indexing
```

Compile warp mapping:

```bash
nvcc -O2 \
  labs/01-gpu-execution/cuda/03_warp_mapping.cu \
  -o warp_mapping
```

Run:

```bash
./warp_mapping
```

Compile divergence:

```bash
nvcc -O2 \
  labs/01-gpu-execution/cuda/04_divergence.cu \
  -o divergence
```

Run:

```bash
./divergence
```

Compile device properties:

```bash
nvcc -O2 \
  labs/01-gpu-execution/cuda/05_device_properties.cu \
  -o device_properties
```

Run:

```bash
./device_properties
```

# 10. Run CUDA Labs

Recommended order:

```text
01_hello_threads
        ↓
02_thread_indexing
        ↓
03_warp_mapping
        ↓
04_divergence
        ↓
05_device_properties
```

The progression is intentional.

First:

> What is a thread?

Then:

> Where does the thread get its identity?

Then:

> How do threads become warps?

Then:

> What happens when execution paths differ?

Finally:

> What hardware is actually available?

# 11. Recommended Environment Check

Before running performance experiments, collect:

```bash
nvidia-smi
```

and:

```bash
nvcc --version
```

Then:

```bash
python -c "import torch; print(torch.__version__)"
```

and:

```bash
python -c "import torch; print(torch.version.cuda)"
```

Then:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

Record these values when publishing benchmark results.

# 12. CPU-Only Machines

You can still read and study the repository without an NVIDIA GPU.

You can also run some CPU-side Python experiments.

However, the CUDA experiments require an NVIDIA CUDA-capable environment.

For example:

```text
Python explanation
      ↓
Can be studied without GPU

CUDA kernel
      ↓
Requires NVIDIA CUDA environment
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

# 13. Benchmarking Rules

Performance experiments require more care than ordinary scripts.

## Warm up

The first execution may not represent steady-state behavior.

Use warm-up iterations where appropriate.

## Repeat

Do not report one timing measurement as if it were a scientific constant.

Run multiple iterations.

Depending on the experiment, report:

* Mean
* Median
* Percentile
* Range

## Synchronize

GPU work can execute asynchronously.

For example, this:

```python
start = time.perf_counter()

result = gpu_operation()

end = time.perf_counter()
```

may not measure the complete GPU operation.

Use CUDA synchronization or CUDA events where appropriate.

## Keep the workload fixed

When comparing two implementations, keep the important variables the same:

* Input size
* Data type
* Batch size
* Number of iterations
* GPU
* Software environment

Change one important variable at a time.

# 14. Reproducibility

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

A benchmark without environment information is difficult to reproduce.

# 15. Common Problems

## `torch.cuda.is_available()` returns `False`

Check:

```bash
nvidia-smi
```

If `nvidia-smi` fails, troubleshoot the NVIDIA driver first.

If `nvidia-smi` works, check:

```bash
python -c "import torch; print(torch.__version__)"
```

```bash
python -c "import torch; print(torch.version.cuda)"
```

Then verify that the installed PyTorch build supports the environment.

Use the official PyTorch installation selector for the correct package.

## `nvcc: command not found`

Check:

```bash
which nvcc
```

If nothing is returned, CUDA Toolkit is not available through your `PATH`.

Check whether the toolkit is installed and configure the environment appropriately for your operating system.

## CUDA program compiles but fails at runtime

Start with:

```bash
nvidia-smi
```

Then check the program's CUDA error output.

For development builds, add explicit error checking after kernel launches and synchronization.

A kernel launch can return before the GPU has completed the work, so synchronization is often useful when debugging.

## `no kernel image is available for execution on the device`

This generally means the compiled code does not contain compatible device code for the GPU architecture being used.

Check the GPU's compute capability:

```bash
python -c "import torch; p=torch.cuda.get_device_properties(0); print(p.major, p.minor)"
```

Then inspect the CUDA compiler and architecture support for your installed toolkit.

## GPU program appears to hang

Possible causes include:

* Invalid memory access
* Synchronization problems
* Incorrect indexing
* Deadlock
* Extremely large workloads
* An incorrectly written kernel

Start with a very small input.

Add CUDA error checks.

Use synchronization while debugging.

Do not immediately assume the GPU is broken.

## Output order from `printf` looks strange

GPU threads execute concurrently.

Do not expect:

```text
Thread 0
Thread 1
Thread 2
Thread 3
```

to always print in exactly that order.

The ordering of printed output is not a reliable way to infer execution order.

# 16. Debugging CUDA Programs

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

```text
100 million threads
complex shared memory
multiple synchronization points
```

and then try to determine which line caused the problem.

GPU debugging becomes much easier when the workload is small enough to reason about.

# 17. Environment Variables

CUDA provides environment variables that can change runtime behavior.

One useful debugging option is:

```text
CUDA_LAUNCH_BLOCKING=1
```

For example:

```bash
CUDA_LAUNCH_BLOCKING=1 ./my_program
```

This forces CUDA kernel launches and related operations to behave synchronously from the host's perspective, which can make errors easier to locate.

Do not use synchronous execution as the default performance configuration.

It is primarily a debugging aid.

# 18. Recommended Development Workflow

For every new CUDA experiment:

### Step 1

Write the smallest possible version.

### Step 2

Compile with:

```bash
nvcc -O2 ...
```

### Step 3

Run a tiny input.

### Step 4

Check the output.

### Step 5

Add error handling.

### Step 6

Increase the workload.

### Step 7

Add timing.

### Step 8

Repeat measurements.

### Step 9

Record the environment.

### Step 10

Only then optimize.

# What You Should Understand Before Moving On

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

If these concepts still feel unclear, repeat Lab 01.

The next article assumes this foundation.

# Next Step

After completing Lab 01, move to:

```text
Article 02
GPU Memory Explained
```

and then:

```text
labs/02-gpu-memory/
```

The focus changes from:

> How does the GPU execute work?

to:

> How does the GPU get the data needed to execute that work?

That is where bandwidth, latency, cache behavior, coalescing, tiling, and the memory wall become important.
