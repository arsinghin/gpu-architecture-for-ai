"""
Lab 01 - Experiment 2

CPU vs GPU execution.

This is an educational experiment, not a universal hardware benchmark.
"""

import time

import torch


SIZE = 10_000_000
REPEATS = 10


def benchmark_cpu(x: torch.Tensor) -> float:
    times = []

    for _ in range(REPEATS):
        start = time.perf_counter()

        _ = x * 2.0

        end = time.perf_counter()
        times.append((end - start) * 1000)

    return sum(times) / len(times)


def benchmark_gpu(x: torch.Tensor) -> float:
    times = []

    # Warm-up
    for _ in range(3):
        _ = x * 2.0

    torch.cuda.synchronize()

    for _ in range(REPEATS):
        torch.cuda.synchronize()

        start = time.perf_counter()

        _ = x * 2.0

        torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)

    return sum(times) / len(times)


def main() -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"Tensor size: {SIZE:,} float32 values")
    print()

    cpu_x = torch.randn(SIZE)

    cpu_time = benchmark_cpu(cpu_x)
    print(f"CPU average: {cpu_time:.3f} ms")

    if not torch.cuda.is_available():
        print("\nCUDA is not available.")
        print("Skipping GPU measurement.")
        return

    gpu_x = cpu_x.to("cuda")

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    gpu_time = benchmark_gpu(gpu_x)
    print(f"GPU average: {gpu_time:.3f} ms")

    print()
    print("Important:")
    print("- This is a simple educational benchmark.")
    print("- It does not represent overall CPU vs GPU performance.")
    print("- Transfer time is not included in the GPU kernel timing.")
    print("- Small workloads can be dominated by launch overhead.")


if __name__ == "__main__":
    main()
