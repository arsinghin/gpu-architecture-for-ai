# Lab 01 — How GPUs Actually Execute AI Workloads

This lab accompanies: **The Anatomy of Silicon: How GPUs Actually Execute AI Workloads**

The article explains the concepts. This lab lets you **run them**.

Instead of only reading about threads, blocks, warps, SMs, kernels, divergence, and asynchronous execution, we will write small programs that expose each idea directly.

---

## What You Will Learn

By the end of this lab, you should be able to explain the execution path:

```text
Python
↓
PyTorch / CUDA
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

You will also see why:

* a GPU thread is not a physical GPU core
* threads are organized into blocks
* blocks form a grid
* NVIDIA GPUs execute threads in groups called warps
* block and thread indices identify the work assigned to a thread
* different threads in a warp can follow different control-flow paths
* GPU work is often asynchronous from the CPU's point of view
* GPU timing must be done carefully
* the same mathematical operation can behave differently depending on how it is executed

---

# Requirements

## For the Python experiments

You need:

* Python 3.9+
* PyTorch
* An NVIDIA GPU with CUDA support for the GPU experiments

Install PyTorch using the installation command appropriate for your system from the official PyTorch installation page:

[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Then verify:

```bash
python -c "import torch; print(torch.__version__)"
```

---

## For the CUDA experiments

You need:

* an NVIDIA GPU
* a working NVIDIA driver
* CUDA Toolkit
* `nvcc`

Check the driver:

```bash
nvidia-smi
```

Check the CUDA compiler:

```bash
nvcc --version
```

You should see information about your installed CUDA compiler.

> The CUDA Toolkit version and the CUDA runtime used by PyTorch do not have to be identical. For these basic experiments, what matters is that your environment is compatible and the programs compile and run successfully.

---

# Repository Structure

```text
01-gpu-execution/
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

The experiments intentionally start very small. Do not skip ahead.

The point is to build the execution model one layer at a time.

---

# Experiment 1 — Find the GPU

File:

```text
python/01_device_check.py
```

Run:

```bash
python python/01_device_check.py
```

This experiment answers:

> **Does PyTorch see a CUDA-capable GPU, and which GPU is it?**

Expected output will look similar to:

```text
PyTorch version: 2.x.x
CUDA available: True
CUDA device count: 1
Current device: 0
GPU name: NVIDIA ...
```

The exact GPU name depends on your hardware.

---

# Experiment 2 — CPU vs GPU

File:

```text
python/02_cpu_vs_gpu.py
```

Run:

```bash
python python/02_cpu_vs_gpu.py
```

This experiment performs the same element-wise operation on CPU and GPU tensors.

It demonstrates:

* CPU tensors
* CUDA tensors
* moving data to the GPU
* GPU computation
* synchronization
* basic timing

Do not treat the result as a universal CPU-vs-GPU benchmark. The goal is to understand **where the operation executes**, not to declare one processor universally faster.

Small workloads can be dominated by launch and transfer overhead.

---

# Experiment 3 — Why GPU Timing Is Different

File:

```text
python/03_async_timing.py
```

Run:

```bash
python python/03_async_timing.py
```

This experiment compares:

* ordinary CPU timing
* synchronized GPU timing
* CUDA event timing

The important lesson:

> **Launching GPU work is not the same thing as waiting for GPU work to finish.**

CUDA operations can execute asynchronously from the CPU's point of view.

That means a naive timer can measure the wrong thing.

For serious GPU benchmarking, use CUDA-aware timing and understand synchronization.

---

# Experiment 4 — Hello, Threads

File:

```text
cuda/01_hello_threads.cu
```

Compile:

```bash
nvcc -O2 cuda/01_hello_threads.cu -o hello_threads
```

Run:

```bash
./hello_threads
```

This is the smallest CUDA experiment in the lab.

It launches multiple GPU threads and lets each thread identify itself.

You should see output similar to:

```text
Hello from block 0, thread 0
Hello from block 0, thread 1
Hello from block 0, thread 2
...
```

The exact output order may vary.

That is intentional.

Do not assume that GPU threads print in numerical order. The GPU is executing parallel work, not politely reading your list from top to bottom.

---

# Experiment 5 — Thread Indexing

File:

```text
cuda/02_thread_indexing.cu
```

Compile:

```bash
nvcc -O2 cuda/02_thread_indexing.cu -o thread_indexing
```

Run:

```bash
./thread_indexing
```

This experiment demonstrates:

```text
threadIdx
blockIdx
blockDim
gridDim
```

and the common global-thread-index calculation:

```cpp
int global_id = blockIdx.x * blockDim.x + threadIdx.x;
```

For example, with:

```text
4 blocks
8 threads per block
```

the global thread IDs become:

```text
Block 0: Thread 0 → Global 0
Thread 1 → Global 1
...
Thread 7 → Global 7

Block 1: Thread 0 → Global 8
Thread 1 → Global 9
...
```

This is the basic mapping used to divide data and work among GPU threads.

---

# Experiment 6 — Threads Become Warps

File:

```text
cuda/03_warp_mapping.cu
```

Compile:

```bash
nvcc -O2 cuda/03_warp_mapping.cu -o warp_mapping
```

Run:

```bash
./warp_mapping
```

This experiment shows:

```text
Global Thread
↓
Warp ID
↓
Lane ID
```

For NVIDIA CUDA GPUs, the warp size is 32.

So:

```text
Thread 0 → Warp 0, Lane 0
Thread 1 → Warp 0, Lane 1
...
Thread 31 → Warp 0, Lane 31
Thread 32 → Warp 1, Lane 0
Thread 33 → Warp 1, Lane 1
```

This is why block sizes that are multiples of 32 are commonly useful.

The CUDA programming model groups threads into 32-thread warps for SIMT execution.

---

# Experiment 7 — Warp Divergence

File:

```text
cuda/04_divergence.cu
```

Compile:

```bash
nvcc -O2 cuda/04_divergence.cu -o divergence
```

Run:

```bash
./divergence
```

This experiment compares two kernels:

```text
Kernel A
All threads follow the same path
```

against:

```text
Kernel B
Threads follow different branches
```

The goal is not to produce a universal percentage such as:

> "Divergence makes your GPU exactly 2× slower."

That would be nonsense.

The cost depends on the workload, architecture, compiler, memory behavior, and branch structure.

Instead, the experiment lets you observe how different control-flow patterns can affect execution.

NVIDIA's CUDA programming model explains that when threads in a warp take different control-flow paths, the paths can be executed with inactive threads masked for the path they are not taking.

---

# Experiment 8 — Inspect the GPU

File:

```text
cuda/05_device_properties.cu
```

Compile:

```bash
nvcc -O2 cuda/05_device_properties.cu -o device_properties
```

Run:

```bash
./device_properties
```

The program reports useful hardware information such as:

```text
GPU name
Compute capability
SM count
Warp size
Maximum threads per block
Shared memory per block
Registers per block
Global memory
```

The exact values depend on your GPU.

This experiment connects the abstract execution model to the actual hardware sitting inside your machine.

---

# What These Experiments Prove

After running the lab, you should be able to connect each concept to something observable.

| Concept                 | Experiment                |
| ----------------------- | ------------------------- |
| CUDA device             | `01_device_check.py`      |
| CPU tensor              | `02_cpu_vs_gpu.py`        |
| CUDA tensor             | `02_cpu_vs_gpu.py`        |
| GPU computation         | `02_cpu_vs_gpu.py`        |
| Synchronization         | `03_async_timing.py`      |
| Kernel                  | `01_hello_threads.cu`     |
| Thread                  | `01_hello_threads.cu`     |
| Block                   | `02_thread_indexing.cu`   |
| Grid                    | `02_thread_indexing.cu`   |
| Thread indexing         | `02_thread_indexing.cu`   |
| Warp                    | `03_warp_mapping.cu`      |
| Lane                    | `03_warp_mapping.cu`      |
| Divergence              | `04_divergence.cu`        |
| GPU hardware properties | `05_device_properties.cu` |

---

# The Mental Model

After finishing the lab, you should be able to look at a CUDA kernel and mentally trace:

```text
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
Registers / Shared Memory / Cache
↓
Execution Resources
```

And when looking at a PyTorch operation:

```text
Python
↓
PyTorch
↓
CUDA operation
↓
GPU kernel / optimized implementation
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

---

# Important Limitations

These experiments are deliberately simplified.

They are designed to build a mental model, not to reproduce every detail of a modern NVIDIA GPU.

In particular:

* GPU architecture differs between generations.
* Not every PyTorch operation maps directly to one CUDA kernel.
* PyTorch may use optimized libraries, generated kernels, fused operations, or other implementations.
* Warp behavior is described using the CUDA programming model; hardware implementation details can differ.
* Timing results depend on the GPU, driver, CUDA version, workload size, clocks, and system state.
* CPU-vs-GPU results from these tiny experiments should not be treated as production benchmarks.
* Occupancy is not the same thing as performance.
* Divergence does not have one fixed performance penalty.
* CUDA-specific experiments require an NVIDIA GPU and CUDA environment.

The purpose of this lab is **understanding**, followed by measurement.

The serious performance experiments come later.

---

# Suggested Experiment Order

Run the experiments in this order:

```text
1. 01_device_check.py
↓
2. 02_cpu_vs_gpu.py
↓
3. 03_async_timing.py
↓
4. 01_hello_threads.cu
↓
5. 02_thread_indexing.cu
↓
6. 03_warp_mapping.cu
↓
7. 04_divergence.cu
↓
8. 05_device_properties.cu
```

Do not worry about memorizing everything.

The goal is to make the hierarchy feel obvious.

---

# Connection to Article 2

Article 1 answers:

> **How does the GPU execute work?**

The next problem is:

> **Where does the data come from, and why can a powerful GPU still spend its time waiting?**

That is the subject of the next lab:

```text
labs/02-gpu-memory/
```

There we will measure:

* memory bandwidth
* coalesced access
* strided access
* cache behavior
* shared-memory reuse
* tiling
* arithmetic intensity
* memory-bound vs compute-bound behavior

The next lab will focus much more heavily on measurements.

---

# References

* NVIDIA CUDA C++ Programming Guide
  [https://docs.nvidia.com/cuda/cuda-programming-guide/](https://docs.nvidia.com/cuda/cuda-programming-guide/)

* NVIDIA CUDA Samples
  [https://github.com/NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples)

* PyTorch CUDA Documentation
  [https://docs.pytorch.org/docs/stable/cuda.html](https://docs.pytorch.org/docs/stable/cuda.html)

* PyTorch CUDA Semantics
  [https://docs.pytorch.org/docs/stable/notes/cuda.html](https://docs.pytorch.org/docs/stable/notes/cuda.html)

---

## License

See the repository root `LICENSE` file.
