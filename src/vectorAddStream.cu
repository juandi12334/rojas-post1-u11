// vectorAddStream.cu - Suma de vectores con streams y transferencia
// asincrona (cudaMemcpyAsync) frente a la version sincrona.
// Arquitectura de Computadores - Unidad 11
// Compilar: nvcc -O2 -o vectorAddStream src/vectorAddStream.cu
// Uso: ./vectorAddStream [n_streams]     (por defecto NSTREAMS = 4)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N (1 << 24)         /* 16M elementos, igual que vectorAdd.cu */
#define NSTREAMS 4          /* valor por defecto */
#define MAX_STREAMS 64
#define BLOCK_SIZE 256
#define REPS 10             /* repeticiones para promediar */

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            fprintf(stderr, "Error CUDA en %s:%d: %s\n", __FILE__,         \
                    __LINE__, cudaGetErrorString(err_));                   \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

__global__ void vectorAddChunk(const float *d_A, const float *d_B,
                               float *d_C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d_C[idx] = d_A[idx] + d_B[idx];
}

int main(int argc, char **argv) {
    int nstreams = NSTREAMS;
    if (argc > 1) nstreams = atoi(argv[1]);
    if (nstreams < 1 || nstreams > MAX_STREAMS || N % nstreams != 0) {
        fprintf(stderr, "n_streams debe estar entre 1 y %d y dividir a N\n",
                MAX_STREAMS);
        return 1;
    }
    size_t chunk = N / nstreams;
    size_t bytes = (size_t)N * sizeof(float);
    size_t chunkBytes = chunk * sizeof(float);

    /* Memoria del host "pinned" (page-locked): necesaria para que
     * cudaMemcpyAsync sea realmente asincrona */
    float *h_A, *h_B, *h_C;
    CUDA_CHECK(cudaHostAlloc((void **)&h_A, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void **)&h_B, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void **)&h_C, bytes, cudaHostAllocDefault));

    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i * 0.5f;
        h_B[i] = (float)i * 1.5f;
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    int blockSize = BLOCK_SIZE;
    int chunkGrid = (int)((chunk + blockSize - 1) / blockSize);

    cudaStream_t streams[MAX_STREAMS];
    for (int s = 0; s < nstreams; s++)
        CUDA_CHECK(cudaStreamCreate(&streams[s]));

    float syncMs = 0, asyncMs = 0;
    int errores = 0;

    /* Cada configuracion se ejecuta una vez de calentamiento y luego REPS
     * veces; se reporta el promedio de las REPS ejecuciones. */
    for (int r = -1; r < REPS; r++) {
        /* --- Version sincrona: H2D completo, un kernel, D2H completo --- */
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
        vectorAddChunk<<<(N + blockSize - 1) / blockSize, blockSize>>>(
            d_A, d_B, d_C, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (r >= 0) syncMs += ms;

        if (r == 0) {   /* verificacion de la version sincrona */
            for (int i = 0; i < N; i++)
                if (fabsf(h_C[i] - (h_A[i] + h_B[i])) > 1e-4f) errores++;
        }
        for (int i = 0; i < N; i += 4096) h_C[i] = -1.0f;  /* invalida */

        /* --- Version con streams: solapa transferencia y computo --- */
        CUDA_CHECK(cudaEventRecord(start));
        for (int s = 0; s < nstreams; s++) {
            size_t off = (size_t)s * chunk;
            CUDA_CHECK(cudaMemcpyAsync(d_A + off, h_A + off, chunkBytes,
                                       cudaMemcpyHostToDevice, streams[s]));
            CUDA_CHECK(cudaMemcpyAsync(d_B + off, h_B + off, chunkBytes,
                                       cudaMemcpyHostToDevice, streams[s]));
            vectorAddChunk<<<chunkGrid, blockSize, 0, streams[s]>>>(
                d_A + off, d_B + off, d_C + off, (int)chunk);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(h_C + off, d_C + off, chunkBytes,
                                       cudaMemcpyDeviceToHost, streams[s]));
        }
        for (int s = 0; s < nstreams; s++)
            CUDA_CHECK(cudaStreamSynchronize(streams[s]));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (r >= 0) asyncMs += ms;

        if (r == 0) {   /* verificacion de la version con streams */
            for (int i = 0; i < N; i++)
                if (fabsf(h_C[i] - (h_A[i] + h_B[i])) > 1e-4f) errores++;
        }
    }
    syncMs /= REPS;
    asyncMs /= REPS;

    printf("Sincrono (H2D + kernel + D2H completos): %.2f ms\n", syncMs);
    printf("Con streams (%d, solapado): %.2f ms\n", nstreams, asyncMs);
    printf("Reduccion: %.1f%%\n", 100.0 * (syncMs - asyncMs) / syncMs);
    printf("Errores (ambas versiones): %d\n", errores);

    for (int s = 0; s < nstreams; s++)
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));
    return errores == 0 ? 0 : 1;
}
