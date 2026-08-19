/*
 * Lab 01 - Experiment 7
 * Warp Divergence
 *
 * Demonstrates:
 * - A uniform control-flow path
 * - A divergent control-flow path
 *
 * This experiment is intentionally simple.
 * Timing is only meaningful as a comparison on the same system.
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>


__global__ void uniform_kernel(
    const float* input,
    float* output,
    int n
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        float x = input[i];

        x = x * 2.0f;
        x = x + 1.0f;
        x = x * 0.5f;

        output[i] = x;
    }
}


__global__ void divergent_kernel(
    const float* input,
    float* output,
    int n
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        float x = input[i];

        if ((i % 2) == 0)
        {
            x = x * 2.0f;
            x = x + 1.0f;
        }
        else
        {
            x = x * 0.5f;
            x = x - 1.0f;
        }

        output[i] = x;
    }
}


float benchmark_kernel(
    void (*launch)(const float*, float*, int),
    const float* input,
    float* output,
    int n
)
{
    constexpr int warmup = 10;
    constexpr int repeats = 100;

    for (int i = 0; i < warmup; ++i)
    {
        launch(input, output, n);
    }

    cudaDeviceSynchronize();

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < repeats; ++i)
    {
        launch(input, output, n);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;

    cudaEventElapsedTime(
        &milliseconds,
        start,
        stop
    );

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds / repeats;
}


// Wrappers make the kernel launches compatible with
// the simple benchmark function above.
void launch_uniform(
    const float* input,
    float* output,
    int n
)
{
    constexpr int threads = 256;
    int blocks = (n + threads - 1) / threads;

    uniform_kernel<<<blocks, threads>>>(
        input,
        output,
        n
    );
}


void launch_divergent(
    const float* input,
    float* output,
    int n
)
{
    constexpr int threads = 256;
    int blocks = (n + threads - 1) / threads;

    divergent_kernel<<<blocks, threads>>>(
        input,
        output,
        n
    );
}


int main()
{
    constexpr int n = 1 << 24;
    constexpr size_t bytes = n * sizeof(float);

    float* host_input = static_cast<float*>(malloc(bytes));

    for (int i = 0; i < n; ++i)
    {
        host_input[i] = static_cast<float>(i % 100);
    }

    float* device_input = nullptr;
    float* device_output = nullptr;

    cudaMalloc(&device_input, bytes);
    cudaMalloc(&device_output, bytes);

    cudaMemcpy(
        device_input,
        host_input,
        bytes,
        cudaMemcpyHostToDevice
    );

    float uniform_ms = benchmark_kernel(
        launch_uniform,
        device_input,
        device_output,
        n
    );

    float divergent_ms = benchmark_kernel(
        launch_divergent,
        device_input,
        device_output,
        n
    );

    printf("Uniform kernel: %.4f ms\n", uniform_ms);
    printf("Divergent kernel: %.4f ms\n", divergent_ms);
    printf(
        "Divergent / Uniform: %.2fx\n",
        divergent_ms / uniform_ms
    );

    cudaFree(device_input);
    cudaFree(device_output);
    free(host_input);

    return 0;
}
