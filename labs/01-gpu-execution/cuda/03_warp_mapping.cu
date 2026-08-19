/*
 * Lab 01 - Experiment 6
 * Warp Mapping
 *
 * Demonstrates:
 * - Warp ID
 * - Lane ID
 * - Relationship between thread ID and warp
 *
 * NVIDIA CUDA warps contain 32 threads.
 */

#include <cstdio>
#include <cuda_runtime.h>


__global__ void warp_mapping_kernel()
{
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread_id / warpSize;
    int lane_id = threadIdx.x % warpSize;

    printf(
        "Block %d | Thread %d | Global Thread %d | "
        "Warp %d | Lane %d\n",
        blockIdx.x,
        threadIdx.x,
        global_thread_id,
        warp_id,
        lane_id
    );
}


int main()
{
    constexpr int blocks = 2;
    constexpr int threads_per_block = 64;

    warp_mapping_kernel<<<blocks, threads_per_block>>>();

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
