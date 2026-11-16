# Post-Contenido 1 — CUDA Benchmark CPU vs GPU
Arquitectura de Computadores — Unidad 11

**Estudiante:** Juan Diego Rojas Rey — código 1152512
**Programa:** Ingeniería de Sistemas, UFPS

Este repositorio tiene dos partes. La Parte 1 son tres programas CUDA con sus mediciones (`vectorAdd`, `matMul`, `vectorAddStream`). La Parte 2 es un análisis comparativo de aceleradores de IA (GPU, TPU, NPU y FPGA) en `docs/analisis-comparativo.md`.

## Estructura

```
src/vectorAdd.cu          Suma de vectores, CPU vs GPU
src/matMul.cu             Multiplicación de matrices, ingenua vs tiling
src/vectorAddStream.cu    vectorAdd con streams y cudaMemcpyAsync
src/infoGPU.cu            Consulta de copy engines, SMs y ancho de banda
docs/analisis-comparativo.md   Parte 2
capturas/                 Capturas reales de la ejecución en Colab
```

## Entorno de ejecución

No hay GPU local, así que las pruebas se corrieron en Google Colab. Los datos salen de `nvidia-smi`, `nvcc --version` y `infoGPU`.

| Elemento | Valor |
|---|---|
| GPU | NVIDIA Tesla T4 (compute capability 7.5), 40 SM, 14,6 GB |
| Driver / CUDA | 580.82.07 / CUDA 13.0 (nvcc V13.0.88) |
| Sistema operativo | Ubuntu 24.04.4 LTS |
| Copy engines (`asyncEngineCount`) | 3 |
| Ancho de banda teórico de memoria | 320 GB/s |
| Bus PCIe | Gen3 x16 (máximo); `nvidia-smi` mostró Gen1 en reposo porque la GPU estaba en estado de ahorro P8 |

## Compilación y ejecución

```bash
nvcc -O2 -Xcompiler -Wall,-Wextra -o vectorAdd src/vectorAdd.cu && ./vectorAdd
./vectorAdd -tabla                       # tabla para N = 1M, 4M, 16M
nvcc -O2 -Xcompiler -Wall,-Wextra -o matMul src/matMul.cu && ./matMul
nvcc -O2 -Xcompiler -Wall,-Wextra -o vectorAddStream src/vectorAddStream.cu && ./vectorAddStream
./vectorAddStream 8                      # número de streams opcional
nvcc -O2 -o infoGPU src/infoGPU.cu && ./infoGPU
```

Flags usadas: `-O2` para optimizar y `-Xcompiler -Wall,-Wextra` para activar las advertencias del compilador anfitrión. Los tres programas compilaron sin advertencias. Todos los kernels se comprueban con `cudaGetLastError()` y todas las llamadas CUDA pasan por la macro `CUDA_CHECK`. Los punteros llevan prefijo `h_` (host) o `d_` (device).

## Parte 1

### Resultados de vectorAdd

Cada hilo suma un elemento (bloques de 256 hilos). El tiempo del kernel es el promedio de 20 ejecuciones después de una de calentamiento. El tiempo "GPU total" incluye la copia de A y B al device, el kernel y la copia de C al host, con memoria normal (`malloc`).

| N | CPU (ms) | Kernel GPU (ms) | GPU total con memcpy (ms) | Errores |
|---|---|---|---|---|
| 1 M | 2,45 | 0,051 | 5,00 | 0 |
| 4 M | 9,70 | 0,194 | 18,46 | 0 |
| 16 M | 38,80 | 0,769 | 74,34 | 0 |

El kernel es unas 48 veces más rápido que la CPU en 1 M y unas 50 veces en 16 M. El tiempo total, en cambio, es mayor que el de la CPU en los tres tamaños.

### Resultados de matMul

| N | CPU (ms) | GPU ingenua (ms) | GPU con tiling (ms) | Speedup tiling/ingenua | Error máx. vs CPU |
|---|---|---|---|---|---|
| 512 | 31,07 | 0,701 | 0,451 | 1,55× | 0 |
| 1024 | 219,20 | 5,570 | 3,498 | 1,59× | 0 |

El tiling usa tiles de 16×16 en memoria compartida con dos `__syncthreads()` por tile. Las entradas son múltiplos de 1/8, así que el resultado de la GPU y el de la CPU coinciden exactamente. Las repeticiones adicionales (capturas `cp2b` y `cp4b`) dieron speedups entre 1,55× y 1,60×, o sea que la medida es estable.

### Síncrono frente a streams (N = 16 M, memoria pinned)

| Versión | Tiempo (ms) |
|---|---|
| Síncrona (H2D + kernel + D2H completos) | 17,07 |
| Con 4 streams solapados | 12,97 |
| Reducción | 24,0 % |

Barrido del número de streams (una corrida por valor, capturas `cp4b` y `cp5b`):

| Streams | Tiempo (ms) | Reducción |
|---|---|---|
| 1 | 16,99 | −0,8 % |
| 2 | 14,23 | 15,6 % |
| 4 | 13,00 | 22,8 % |
| 8 | 12,47 | 26,3 % |
| 16 | 12,36 | 26,9 % |
| 32 | 12,68 | 25,9 % |

### Análisis

**Por qué el kernel de la GPU gana a la CPU con N grande.** El vectorAdd hace una suma por cada 12 bytes que mueve (lee dos floats y escribe uno), así que está limitado por memoria y no por cómputo. La CPU procesa los elementos casi en serie con unos pocos núcleos y su ancho de banda de memoria. La GPU lanza miles de hilos y la T4 tiene 320 GB/s de ancho de banda teórico. Para 16 M de elementos el kernel movió 192 MB en 0,769 ms, lo que da unos 250 GB/s, cerca del límite de la memoria. La CPU tardó 38,80 ms. Cuando N crece, el costo fijo de lanzar el kernel se reparte entre más trabajo y la ventaja se estabiliza en un valor cercano a 50×.

**Por qué el tiempo total con memcpy puede superar a la CPU.** Hay que copiar 128 MB hacia la GPU y 64 MB de vuelta por el bus PCIe, que es mucho más lento que la memoria de la GPU. Con memoria normal (`malloc`), el driver tiene que pasar por un búfer intermedio, y la medición dio unos 74 ms para 192 MB, es decir unos 2,6 GB/s. Con memoria pinned, en el programa de streams, las mismas copias completas tardaron 17 ms (unos 11 GB/s, razonable para PCIe 3.0 x16). En el vectorAdd el cómputo es tan barato que la transferencia domina todo: el kernel es menos del 2 % del tiempo total. En este laboratorio el total superó a la CPU incluso en N = 16 M (74,34 ms contra 38,80 ms), no solo para N pequeños, porque la memoria no era pinned. Para N pequeño el efecto es peor aún, ya que además pesan la latencia de lanzamiento y la creación del contexto. La GPU solo conviene en este problema si los datos ya viven en la GPU y se reutilizan en varios kernels, o si se usa memoria pinned y se solapan copias con cómputo.

**Sobre matMul.** En matMul el cómputo sí crece más rápido que los datos (N³ operaciones sobre N² datos), y por eso la GPU gana también con las copias. El tiling mejoró un 55–60 %, bastante menos de lo que se espera en teoría. La T4 tiene cachés L1 y L2 que ya absorben buena parte del patrón de accesos de la versión ingenua, y con N = 512 y 1024 las matrices caben en caché. El tiling ayudaría más con matrices más grandes.

**Diferencias con los ejemplos de la guía.** La guía usa como ejemplo una sincronización de unos 14,5 ms y una reducción de alrededor del 11,7 %. En esta T4 se obtuvo 17,07 ms y 24 %. Los valores dependen del bus PCIe y de la GPU, y Colab no es el mismo equipo. Además, la reducción del 24 % supera lo que se esperaría si el cómputo fuera el límite del solapamiento: como la T4 tiene 3 copy engines, también se solapan las copias H2D con las D2H y no solo con el kernel.

### Decisión 1: ¿streams o sincronía en vectorAdd?

Para vectorAdd se usó la versión con streams. Con 16 M de elementos los tres vectores suman 192 MB, y a unos 10–12 GB/s por PCIe eso son unos 16–19 ms de transferencia, mientras el kernel solo tarda unos 0,8 ms. Como el tiempo está dominado por las copias, el ahorro posible es grande: dividir el vector en trozos permite que la copia del trozo siguiente corra a la vez que se calcula el anterior y se devuelve el previo. En la medición real la reducción fue del 24 % (17,07 ms a 12,97 ms con 4 streams), y se pudo llegar a casi 27 % con 8 y 16 streams. No se puede ocultar más que la transferencia más larga, así que la mejora tiene un techo. Para matMul los streams convienen todavía más, porque ahí el cómputo (5,6 ms o 3,5 ms según la versión, para N = 1024) es comparable con el tiempo de copia de 3 matrices de 4 MB, y se puede ocultar casi toda la transferencia detrás del cálculo. En vectorAdd el beneficio es real pero acotado; el cómputo es tan pequeño que el cuello de botella sigue siendo el bus.

### Decisión 2: ¿cuántos streams?

Se usaron 4 streams por defecto y se midió un barrido de 1 a 32. Con 1 stream no hay solapamiento (−0,8 %, que es ruido de medición más el costo del stream). Con 2 la reducción sube a 15,6 %, con 4 a 22,8 % y con 8 llega a 26,3 %. A partir de ahí se estanca: 16 streams dan 26,9 % y 32 bajan a 25,9 %. La explicación es de hardware. La T4 reporta 3 copy engines, así que solo puede mover datos en un número limitado de direcciones a la vez, y todas las copias comparten el mismo ancho de banda del PCIe: más streams no añaden más ancho de banda, solo reparten el mismo. Además cada stream extra agrega sobrecarga de lanzamiento y trozos más pequeños, que aprovechan peor el bus. Por eso hay rendimientos decrecientes. Se dejó 4 como valor por defecto porque ya captura casi todo el beneficio (22,8 % frente al máximo de 26,9 %) con poca complejidad, y se permite cambiarlo con un argumento para repetir el experimento.

### Capturas

| Captura | Contenido |
|---|---|
| `capturas/cp1a_entorno_y_vectorAdd.jpg` | `nvidia-smi`, `nvcc --version` y primera ejecución de vectorAdd |
| `capturas/cp1b_vectorAdd_resultados.jpg` | Tabla de vectorAdd para 1 M, 4 M y 16 M |
| `capturas/cp2a_matMul_compilacion_y_tiempos.jpg` | Compilación sin advertencias y tiempos de matMul |
| `capturas/cp2b_matMul_repeticiones.jpg` | Repeticiones de matMul |
| `capturas/cp4a_vectorAddStream.jpg` | vectorAddStream: síncrono frente a streams |
| `capturas/cp4b_barrido_streams.jpg` | Barrido de 1 a 32 streams |
| `capturas/cp5a_infoGPU_copy_engines.jpg` | infoGPU: 3 copy engines, 40 SM, 320 GB/s |
| `capturas/cp5b_vectorAddStream_5_corridas.jpg` | Cinco corridas de vectorAddStream |

## Parte 2

El análisis comparativo está en [`docs/analisis-comparativo.md`](docs/analisis-comparativo.md). Incluye la caracterización de GPU, TPU, NPU y FPGA, los resultados de MLPerf Inference v4.1, los costos de 3 años para dos escenarios, la recomendación para tres perfiles, la comparación INT8 frente a FP32 con datos de Wu et al. (2020), las decisiones 3 y 4 y las referencias en formato APA.

## Historial de commits

Los commits están separados por tema (estructura, kernels, streams, análisis) para que se pueda seguir el avance con `git log`.
