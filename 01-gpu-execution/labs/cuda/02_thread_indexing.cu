/*
 * Lab 01 - Experiment 5
 * Thread Indexing
 *
 * Demonstrates:
 * - gridDim
 * - blockDim
 * - blockIdx
 * - threadIdx
 * - global thread ID
 */

#include <cstdio>
#include <cuda_runtime.h>


__global__ void indexing_kernel()
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    printf(
        "Block %d | Thread %d | Global ID %d\n",
        blockIdx.x,
        threadIdx.x,
        global_id
    );
}


int main()
{
    constexpr int blocks = 4;
    constexpr int threads_per_block = 8;

    indexing_kernel<<<blocks, threads_per_block>>>();

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
