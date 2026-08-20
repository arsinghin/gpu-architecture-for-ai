"""
Lab 01 - Experiment 3

Demonstrate asynchronous GPU execution and correct timing.

The experiment compares:
1. A naive CPU timer without explicit synchronization.
2. A CPU timer with synchronization.
3. CUDA event timing.
"""

import time

import torch


SIZE = 20_000_000
REPEATS = 20


def naive_cpu_timer(x: torch.Tensor) -> float:
    times = []

    for _ in range(REPEATS):
        start = time.perf_counter()

        _ = x * 2.0

        end = time.perf_counter()
        times.append((end - start) * 1000)

    return sum(times) / len(times)


def synchronized_cpu_timer(x: torch.Tensor) -> float:
    times = []

    for _ in range(REPEATS):
        torch.cuda.synchronize()

        start = time.perf_counter()

        _ = x * 2.0

        torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)

    return sum(times) / len(times)


def cuda_event_timer(x: torch.Tensor) -> float:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []

    for _ in range(REPEATS):
        torch.cuda.synchronize()

        start_event.record()

        _ = x * 2.0

        end_event.record()
        end_event.synchronize()

        times.append(start_event.elapsed_time(end_event))

    return sum(times) / len(times)


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Tensor size: {SIZE:,} float32 values")
    print()

    x = torch.randn(SIZE, device="cuda")

    # Warm-up
    for _ in range(5):
        _ = x * 2.0

    torch.cuda.synchronize()

    naive = naive_cpu_timer(x)
    synchronized = synchronized_cpu_timer(x)
    cuda_events = cuda_event_timer(x)

    print(f"Naive CPU timer: {naive:.3f} ms")
    print(f"Synchronized CPU timer: {synchronized:.3f} ms")
    print(f"CUDA event timer: {cuda_events:.3f} ms")

    print()
    print("Interpretation:")
    print()
    print("The naive timer may underestimate GPU execution time")
    print("because launching GPU work does not necessarily mean")
    print("the GPU has already finished that work.")
    print()
    print("Synchronization waits for previously issued GPU work.")
    print("CUDA events measure elapsed time on the GPU timeline.")


if __name__ == "__main__":
    main()
