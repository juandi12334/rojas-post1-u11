// matMul.cu - Multiplicacion de matrices C = A x B en GPU:
// version ingenua (memoria global) y version con tiling en shared memory.
// Arquitectura de Computadores - Unidad 11
// Compilar: nvcc -O2 -o matMul src/matMul.cu
// Uso: ./matMul        (N = 512 y N = 1024)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define TILE 16
#define REPS 10             /* repeticiones de cada kernel para promediar */

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            fprintf(stderr, "Error CUDA en %s:%d: %s\n", __FILE__,         \
                    __LINE__, cudaGetErrorString(err_));                   \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

/* Version ingenua: cada thread lee una fila de A y una columna de B
 * directamente de la memoria global (2*N lecturas por elemento de C). */
__global__ void matMulNaive(const float *d_A, const float *d_B,
                            float *d_C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++)
            sum += d_A[row * n + k] * d_B[k * n + col];
        d_C[row * n + col] = sum;
    }
}

/* Version con tiling: cada bloque carga tiles de TILE x TILE en shared
 * memory, de modo que cada dato de memoria global se lee una vez por
 * bloque y se reutiliza TILE veces. */
__global__ void matMulTiled(const float *d_A, const float *d_B,
                            float *d_C, int n) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (n + TILE - 1) / TILE; t++) {
        /* Cargar el tile de A */
        if (row < n && t * TILE + threadIdx.x < n)
            sA[threadIdx.y][threadIdx.x] =
                d_A[row * n + t * TILE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        /* Cargar el tile de B */
        if (col < n && t * TILE + threadIdx.y < n)
            sB[threadIdx.y][threadIdx.x] =
                d_B[(t * TILE + threadIdx.y) * n + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();            /* el tile ya esta completo */

        for (int k = 0; k < TILE; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();            /* nadie sobrescribe antes de terminar */
    }

    if (row < n && col < n) d_C[row * n + col] = sum;
}

/* Referencia en CPU */
static void matMulCPU(const float *h_A, const float *h_B, float *h_C, int n) {
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            float a = h_A[i * n + k];
            for (int j = 0; j < n; j++)
                h_C[i * n + j] += a * h_B[k * n + j];
        }
}

static double ahora_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Mide un kernel con cudaEvent (promedio de REPS ejecuciones) */
static float medir(void (*lanzar)(const float *, const float *, float *, int,
                                  dim3, dim3),
                   const float *d_A, const float *d_B, float *d_C, int n,
                   dim3 grid, dim3 block) {
    cudaEvent_t ini, fin;
    CUDA_CHECK(cudaEventCreate(&ini));
    CUDA_CHECK(cudaEventCreate(&fin));
    lanzar(d_A, d_B, d_C, n, grid, block);          /* calentamiento */
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ini));
    for (int r = 0; r < REPS; r++) lanzar(d_A, d_B, d_C, n, grid, block);
    CUDA_CHECK(cudaEventRecord(fin));
    CUDA_CHECK(cudaEventSynchronize(fin));
    CUDA_CHECK(cudaGetLastError());
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ini, fin));
    CUDA_CHECK(cudaEventDestroy(ini));
    CUDA_CHECK(cudaEventDestroy(fin));
    return ms / REPS;
}

static void lanzarNaive(const float *a, const float *b, float *c, int n,
                        dim3 g, dim3 bl) {
    matMulNaive<<<g, bl>>>(a, b, c, n);
}
static void lanzarTiled(const float *a, const float *b, float *c, int n,
                        dim3 g, dim3 bl) {
    matMulTiled<<<g, bl>>>(a, b, c, n);
}

static float maxError(const float *x, const float *y, size_t cnt) {
    float m = 0;
    for (size_t i = 0; i < cnt; i++) {
        float e = fabsf(x[i] - y[i]);
        if (e > m) m = e;
    }
    return m;
}

static void correr(int n) {
    size_t cnt = (size_t)n * n;
    size_t bytes = cnt * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C_cpu = (float *)calloc(cnt, sizeof(float));
    float *h_C_naive = (float *)malloc(bytes);
    float *h_C_tiled = (float *)malloc(bytes);
    if (!h_A || !h_B || !h_C_cpu || !h_C_naive || !h_C_tiled) {
        fprintf(stderr, "No hay memoria en el host\n");
        exit(EXIT_FAILURE);
    }

    /* Multiplos de 1/8: los productos y las sumas son exactos en float,
     * asi la comparacion con la CPU no depende del orden de suma. */
    for (size_t i = 0; i < cnt; i++) {
        h_A[i] = (float)(i % 8) / 8.0f;
        h_B[i] = (float)((i * 3 + 1) % 8) / 8.0f;
    }

    double t0 = ahora_ms();
    matMulCPU(h_A, h_B, h_C_cpu, n);
    double cpu_ms = ahora_ms() - t0;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    float naive_ms = medir(lanzarNaive, d_A, d_B, d_C, n, grid, block);
    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, bytes, cudaMemcpyDeviceToHost));

    float tiled_ms = medir(lanzarTiled, d_A, d_B, d_C, n, grid, block);
    CUDA_CHECK(cudaMemcpy(h_C_tiled, d_C, bytes, cudaMemcpyDeviceToHost));

    float err_naive = maxError(h_C_naive, h_C_cpu, cnt);
    float err_tiled = maxError(h_C_tiled, h_C_cpu, cnt);

    printf("N = %d\n", n);
    printf("  CPU (3 bucles):        %9.2f ms\n", cpu_ms);
    printf("  GPU naive:             %9.3f ms   error max vs CPU: %g\n",
           naive_ms, err_naive);
    printf("  GPU tiling (TILE=%d):  %9.3f ms   error max vs CPU: %g\n",
           TILE, tiled_ms, err_tiled);
    printf("  Speedup tiling/naive:  %9.2fx\n", naive_ms / tiled_ms);
    printf("  Resultado %s (tolerancia 1e-3)\n",
           (err_naive < 1e-3f && err_tiled < 1e-3f) ? "CORRECTO" : "INCORRECTO");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_naive); free(h_C_tiled);
}

int main(void) {
    CUDA_CHECK(cudaFree(0));        /* crea el contexto antes de medir */
    correr(512);
    printf("\n");
    correr(1024);
    return 0;
}
