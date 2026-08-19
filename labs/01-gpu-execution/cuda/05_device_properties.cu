/*
 * Lab 01 - Experiment 8
 * GPU Device Properties
 *
 * Demonstrates how to inspect the hardware that is
 * actually running the CUDA program.
 */

#include <cstdio>
#include <cuda_runtime.h>


int main()
{
    int device_count = 0;

    cudaError_t error = cudaGetDeviceCount(&device_count);

    if (error != cudaSuccess)
    {
        fprintf(
            stderr,
            "CUDA error: %s\n",
            cudaGetErrorString(error)
        );

        return 1;
    }

    if (device_count == 0)
    {
        printf("No CUDA-capable GPU found.\n");
        return 0;
    }

    printf("CUDA devices: %d\n\n", device_count);

    for (int device = 0; device < device_count; ++device)
    {
        cudaDeviceProp prop{};

        error = cudaGetDeviceProperties(&prop, device);

        if (error != cudaSuccess)
        {
            fprintf(
                stderr,
                "Could not query device %d: %s\n",
                device,
                cudaGetErrorString(error)
            );

            continue;
        }

        printf("GPU %d\n", device);
        printf("------------------------------\n");
        printf("Name: %s\n", prop.name);
        printf(
            "Compute capability: %d.%d\n",
            prop.major,
            prop.minor
        );
        printf(
            "SM count: %d\n",
            prop.multiProcessorCount
        );
        printf(
            "Warp size: %d\n",
            prop.warpSize
        );
        printf(
            "Max threads per block: %d\n",
            prop.maxThreadsPerBlock
        );
        printf(
            "Max threads per SM: %d\n",
            prop.maxThreadsPerMultiProcessor
        );
        printf(
            "Registers per block: %zu\n",
            prop.regsPerBlock
        );
        printf(
            "Shared memory per block: %zu bytes\n",
            prop.sharedMemPerBlock
        );
        printf(
            "Global memory: %.2f GiB\n",
            static_cast<double>(prop.totalGlobalMem) /
                (1024.0 * 1024.0 * 1024.0)
        );
        printf("\n");
    }

    return 0;
}
