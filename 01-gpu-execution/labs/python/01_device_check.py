"""
Lab 01 - Experiment 1

Check whether PyTorch can see a CUDA-capable GPU.
"""

import torch


def main() -> None:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\nNo CUDA-capable GPU is available to PyTorch.")
        print("The CPU-only experiments can still be used.")
        return

    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()

    print(f"CUDA device count: {device_count}")
    print(f"Current device: {current_device}")
    print(f"GPU name: {torch.cuda.get_device_name(current_device)}")

    properties = torch.cuda.get_device_properties(current_device)

    print(f"Compute capability: {properties.major}.{properties.minor}")
    print(f"GPU memory: {properties.total_memory / (1024**3):.2f} GiB")


if __name__ == "__main__":
    main()
