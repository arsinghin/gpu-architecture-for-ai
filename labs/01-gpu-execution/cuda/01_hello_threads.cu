/*
 * Lab 01 - Experiment 4
 * Hello, Threads
 *
 * Demonstrates:
 * - CUDA kernel
 * - Thread
 * - Thread block
 * - Basic kernel launch
 */

#include <cstdio>
#include <cuda_runtime.h>


__global__ void hello_kernel()
{
    printf(
        "Hello from block %d, thread %d\n",
        blockIdx.x,
        threadIdx.x
    );
}


int main()
{
    constexpr int blocks = 2;
    constexpr int threads_per_block = 8;

    hello_kernel<<<blocks, threads_per_block>>>();

    cudaError_t error = cudaDeviceSynchronize();

    if (error != cudaSuccess)
    {
        fprintf(
            stderr,
            "CUDA error: %s\n",
            cudaGetErrorString(error)
        );

        return 1;
    }

    return 0;
}
