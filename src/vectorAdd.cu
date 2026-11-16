// vectorAdd.cu - Suma de vectores en CPU y GPU (CUDA)
// Arquitectura de Computadores - Unidad 11
// Compilar: nvcc -O2 -o vectorAdd src/vectorAdd.cu
// Uso: ./vectorAdd            (N = 16M, como pide la guia)
//      ./vectorAdd -tabla     (tabla para N = 1M, 4M y 16M)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define REPS_KERNEL 20      /* repeticiones del kernel para promediar */

/* Revisa el retorno de cada llamada a la API de CUDA */
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            fprintf(stderr, "Error CUDA en %s:%d: %s\n", __FILE__,         \
                    __LINE__, cudaGetErrorString(err_));                   \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

/* Kernel: cada thread calcula un elemento de C */
__global__ void vectorAdd(const float *d_A, const float *d_B,
                          float *d_C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d_C[idx] = d_A[idx] + d_B[idx];
}

/* Referencia en CPU */
static void vectorAddCPU(const float *h_A, const float *h_B,
                         float *h_C, int n) {
    for (int i = 0; i < n; i++) h_C[i] = h_A[i] + h_B[i];
}

static double ahora_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Ejecuta el benchmark para n elementos; devuelve el numero de errores */
static int correr(int n, int imprimir_ok) {
    size_t bytes = (size_t)n * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C_cpu = (float *)malloc(bytes);
    float *h_C_gpu = (float *)malloc(bytes);
    if (!h_A || !h_B || !h_C_cpu || !h_C_gpu) {
        fprintf(stderr, "No hay memoria en el host\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n; i++) {
        h_A[i] = (float)i * 0.5f;
        h_B[i] = (float)i * 1.5f;
    }

    /* --- Benchmark CPU --- */
    double t0 = ahora_ms();
    vectorAddCPU(h_A, h_B, h_C_cpu, n);
    double cpu_ms = ahora_ms() - t0;

    /* --- Benchmark GPU --- */
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    cudaEvent_t ev_ini, ev_fin, tot_ini, tot_fin;
    CUDA_CHECK(cudaEventCreate(&ev_ini));
    CUDA_CHECK(cudaEventCreate(&ev_fin));
    CUDA_CHECK(cudaEventCreate(&tot_ini));
    CUDA_CHECK(cudaEventCreate(&tot_fin));

    int blockSize = BLOCK_SIZE;
    int gridSize = (n + blockSize - 1) / blockSize;

    /* Tiempo total: H2D + kernel + D2H (una sola ejecucion) */
    CUDA_CHECK(cudaEventRecord(tot_ini));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(tot_fin));
    CUDA_CHECK(cudaEventSynchronize(tot_fin));
    float total_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, tot_ini, tot_fin));

    /* Solo el kernel: una pasada de calentamiento y luego el promedio */
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ev_ini));
    for (int r = 0; r < REPS_KERNEL; r++)
        vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaEventRecord(ev_fin));
    CUDA_CHECK(cudaEventSynchronize(ev_fin));
    CUDA_CHECK(cudaGetLastError());
    float gpu_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, ev_ini, ev_fin));
    gpu_ms /= REPS_KERNEL;

    /* Verificar resultados contra la CPU */
    int errores = 0;
    for (int i = 0; i < n; i++)
        if (fabsf(h_C_gpu[i] - h_C_cpu[i]) > 1e-4f) errores++;

    if (imprimir_ok) {
        printf("N = %d elementos (%.0f MB por vector)\n", n,
               bytes / (1024.0 * 1024.0));
        printf("CPU: %.2f ms\n", cpu_ms);
        printf("GPU kernel: %.2f ms\n", gpu_ms);
        printf("GPU total (H2D + kernel + D2H): %.2f ms\n", total_ms);
        printf("Errores: %d\n", errores);
    } else {
        printf("%-8d  %10.2f  %12.3f  %14.2f  %8d\n", n / (1 << 20),
               cpu_ms, gpu_ms, total_ms, errores);
    }

    CUDA_CHECK(cudaEventDestroy(ev_ini));
    CUDA_CHECK(cudaEventDestroy(ev_fin));
    CUDA_CHECK(cudaEventDestroy(tot_ini));
    CUDA_CHECK(cudaEventDestroy(tot_fin));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    return errores;
}

int main(int argc, char **argv) {
    /* Crea el contexto antes de medir (la primera llamada a CUDA es lenta) */
    CUDA_CHECK(cudaFree(0));

    if (argc > 1 && strcmp(argv[1], "-tabla") == 0) {
        printf("%-8s  %10s  %12s  %14s  %8s\n", "N (M)", "CPU (ms)",
               "kernel (ms)", "GPU total (ms)", "errores");
        int tamanos[] = {1 << 20, 1 << 22, 1 << 24};
        for (int k = 0; k < 3; k++) correr(tamanos[k], 0);
        return 0;
    }
    return correr(1 << 24, 1) == 0 ? 0 : 1;
}
