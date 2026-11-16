# Análisis comparativo de aceleradores de IA: GPU, TPU, NPU y FPGA

**Autor:** Juan Diego Rojas Rey (código 1152512)
**Materia:** Arquitectura de Computadores, Ingeniería de Sistemas, UFPS
**Unidad 11 — Parte 2**

---

## Introducción

Un procesador de propósito general ya no basta para ejecutar redes neuronales con costos y consumos razonables, y por eso hoy conviven cuatro familias de aceleradores: las GPU, las TPU, las NPU y las FPGA. Las cuatro atacan el mismo problema, que es hacer muchísimas multiplicaciones y sumas de matrices por segundo, pero lo resuelven con ideas distintas. La GPU reparte el trabajo entre miles de hilos SIMT, la TPU lo ejecuta sobre un arreglo sistólico, la NPU integra un bloque fijo de multiplicación de matrices dentro de un SoC, y la FPGA permite dibujar el circuito exacto que necesita el modelo.

En la Parte 1 de esta unidad se midió en una GPU real (Tesla T4 de Google Colab) qué tanto pesan el cómputo y la transferencia por PCIe. El vectorAdd con 16 M de elementos tardó 0,77 ms en el kernel pero 74,10 ms en total con las copias, y el programa con streams logró reducir el tiempo de 17,07 ms a 12,97 ms. Ese resultado deja una idea que se repite en todo este documento: la velocidad pico de un chip casi nunca es lo que decide la elección, y lo que decide es el movimiento de datos, la potencia, el costo y el ecosistema de software.

Este documento compara los cuatro tipos de acelerador con datos de fabricantes y de MLPerf Inference v4.1, calcula costos para dos escenarios, recomienda una plataforma para tres perfiles de usuario y analiza la cuantización INT8. Donde un dato no es oficial sino calculado a partir de otros, se indica explícitamente. También se aclara cuando una cifra de la guía de la actividad no coincide con fuentes verificables.

## 1. Caracterización técnica

La Tabla 1 resume un representante de cada familia. Los valores salen de hojas de datos y documentación oficial. Un punto importante es que los picos de NVIDIA para la H100 suelen venir con esparsidad estructurada, que duplica la cifra; aquí se usan los valores densos.

**Tabla 1. Aceleradores representativos (valores de fabricante)**

| Familia | Dispositivo | Cómputo pico | Memoria | Potencia | Observación |
|---|---|---|---|---|---|
| GPU | NVIDIA H100 SXM | 989 TFLOPS FP16 (denso); 1.979 TOPS INT8 (denso) | 80 GB HBM3, 3,35 TB/s | hasta 700 W | Tensor Cores de 4.ª generación (NVIDIA, s. f.; SemiAnalysis, s. f.) |
| GPU | NVIDIA T4 | 65 TFLOPS FP16; 130 TOPS INT8 | 16 GB GDDR6, 320 GB/s | 70 W | La GPU usada en las mediciones de la Parte 1 (NVIDIA, 2018) |
| TPU | Google Cloud TPU v5e | 197 TFLOPS bf16; 393 TOPS INT8 | 16 GB HBM2, 819 GB/s | no publicada | Pods de hasta 256 chips, cada TensorCore con 4 MXU (Google Cloud, s. f.) |
| NPU | Apple Neural Engine (M3) | 18 TOPS (FP16) | memoria unificada del SoC, hasta 400 GB/s | no publicada para el bloque | 16 núcleos (AnandTech, 2023; Apple, 2023) |
| NPU | Qualcomm Hexagon (Snapdragon 8 Gen 3) | no publica TOPS en la ficha consultada | memoria del teléfono | no publicada | Declara 98 % más rendimiento y 40 % más rendimiento por vatio que la generación anterior (Qualcomm, 2023) |
| FPGA | AMD Alveo U55C | no hay un pico único; depende del diseño (9.024 DSP, 1,304 M LUT) | 16 GB HBM2, 460 GB/s | TDP 115 W, máximo 150 W | Se programa con HLS o RTL (AMD, 2023) |

Dos advertencias sobre la tabla. La primera es que los TOPS de una NPU y los TFLOPS de una GPU no son directamente comparables, porque dependen de la precisión, de si se cuenta esparsidad y de si el pico se puede sostener con datos reales. La segunda es que la guía de la actividad menciona un número de núcleos de Neural Engine y precios que no coinciden con las fuentes consultadas; en este documento se usan las cifras verificadas (el M3 trae un Neural Engine de 16 núcleos, y Apple y AnandTech coinciden en 18 TOPS).

La diferencia arquitectónica se entiende bien con lo que se vio en el laboratorio. Una GPU como la T4 tiene 40 SM, y el vectorAdd (un kernel limitado por memoria) usó muy poco de su cómputo, porque cada elemento hace una sola suma por cada 12 bytes movidos. Una TPU resuelve esto con un arreglo sistólico: los datos pasan de una celda a la vecina sin volver a la memoria, y por eso las unidades de matriz (MXU) reutilizan cada operando muchas veces (Jouppi et al., 2023). Es la misma idea del tiling con memoria compartida de matMul, llevada al hardware. La FPGA, en cambio, no tiene un conjunto de instrucciones: el diseñador describe un pipeline a medida y lo gana en latencia determinista, pero paga en tiempo de desarrollo y en frecuencia de reloj más baja.

## 2. Rendimiento en MLPerf Inference v4.1

MLPerf Inference es el estándar más usado para comparar sistemas de inferencia. Define escenarios (single-stream, multistream, server y offline) y exige una precisión mínima respecto al modelo de referencia, por ejemplo 99 % o 99,9 % (Reddi et al., 2020). La ronda v4.1 se publicó en agosto de 2024.

Hay tres limitaciones que conviene decir antes de la tabla. MLPerf no reporta latencia p99 directamente para BERT: reporta consultas por segundo bajo una restricción de latencia en el escenario server, así que no se puede citar un "p99 de BERT-Large" como número oficial. En esta ronda no hubo resultados de FPGA en el apartado de centros de datos que se consultó. Y NVIDIA presentó sistemas de 8 GPU, de modo que el valor por GPU se obtiene dividiendo entre 8 y es un cálculo propio, no una cifra publicada.

**Tabla 2. Resultados MLPerf Inference v4.1 (sistemas completos, división cerrada)**

| Carga | Sistema | Rendimiento | Potencia | Fuente |
|---|---|---|---|---|
| ResNet-50, server | 8× NVIDIA H200 SXM | 480.131 consultas/s | 5.013,79 W | IEEE Spectrum (2024) |
| ResNet-50, server | 6× Untether speedAI240 Slim (NPU) | 309.752 consultas/s | 985,52 W | IEEE Spectrum (2024) |
| BERT-Large (99 %), server | 8× NVIDIA H200 SXM | 57.609 consultas/s | no consultada | NVIDIA (2024) |
| BERT-Large (99 %), offline | 8× NVIDIA H200 SXM | 73.310 muestras/s | no consultada | NVIDIA (2024) |
| Stable Diffusion XL, server | 8× NVIDIA H200 SXM | 16,78 consultas/s | no consultada | NVIDIA (2024) |
| Stable Diffusion XL, offline | 8× NVIDIA H200 SXM | 17,42 muestras/s | no consultada | NVIDIA (2024) |

La H200 comparte el chip de la H100 y cambia la memoria por HBM3e, por lo que sus resultados sirven como referencia de la generación Hopper, aunque no son de la H100 exacta.

**Quién lidera en rendimiento.** La GPU de NVIDIA lidera en throughput absoluto y es la única con resultados en todas las cargas, incluyendo SDXL. En ResNet-50 el sistema de 8 H200 dio 480.131 consultas/s contra 309.752 del sistema de 6 NPU Untether, es decir, un 55 % más. Por GPU, el sistema NVIDIA da unas 60.000 consultas/s (480.131 / 8, cálculo propio). Un análisis de xpu.pub (2024) indica además que la H200 resultó cerca de 1,9 veces más rápida que la TPU Trillium en Stable Diffusion, aunque esa TPU era una vista previa y no una TPU v5e.

**Quién lidera en eficiencia.** Aquí cambia el orden. Las consultas por segundo por vatio fueron 95,76 para el sistema H200 y 314,30 para el de Untether, unas 3,3 veces más para la NPU (314,30 / 95,76 = 3,28). La razón es de diseño: la NPU de Untether usa computación en memoria (*at-memory compute*) y tarjetas de 75 W, mientras que una H200 SXM está pensada para consumir hasta 700 W y maximizar las cargas grandes de LLM. La NPU renuncia a flexibilidad (solo reportó ResNet-50) y gana en vatios. Esto coincide con la lectura de xpu.pub (2024): entre el 60 % y el 65 % del rendimiento con cerca del 20 % de la potencia.

La conclusión es que "mejor" depende de la métrica. Para un centro de datos que necesita cubrir muchos modelos, la GPU es la elección natural por su cobertura; para un nodo con presupuesto térmico fijo y un solo modelo de visión, una NPU o una FPGA bien diseñada puede ser más sensata.
