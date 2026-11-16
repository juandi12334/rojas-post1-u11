// infoGPU.cu - Muestra las propiedades de la GPU que importan para las
// decisiones tecnicas de la Parte 1 (copy engines, SMs, ancho de banda).
// Compilar: nvcc -O2 -o infoGPU src/infoGPU.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main(void) {
    cudaDeviceProp prop;
    int dev = 0, memClockKHz = 0, busWidth = 0;
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
        fprintf(stderr, "No se pudo leer la GPU\n");
        return 1;
    }
    cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&busWidth, cudaDevAttrGlobalMemoryBusWidth, dev);
    printf("GPU: %s (compute capability %d.%d)\n", prop.name, prop.major,
           prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Memoria global: %.1f GB\n", prop.totalGlobalMem / 1073741824.0);
    printf("Copy engines (asyncEngineCount): %d\n", prop.asyncEngineCount);
    printf("Kernels concurrentes: %d\n", prop.concurrentKernels);
    printf("BW teorico de memoria: %.0f GB/s\n",
           2.0 * memClockKHz * 1e3 * (busWidth / 8) / 1e9);
    return 0;
}
