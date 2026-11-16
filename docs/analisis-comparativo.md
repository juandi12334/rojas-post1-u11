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

## 3. Costos, TDP y TCO a 3 años

Se compararon dos escenarios. En todos los cálculos se usan 26.280 horas (3 años × 8.760 h), operación continua y una tarifa eléctrica industrial de 8,91 centavos por kWh, que es el promedio de EIA de los 12 meses hasta julio de 2026 según Start Business by State (2026). No se incluyen refrigeración (PUE), servidor anfitrión, red ni personal, por lo que son costos mínimos de referencia.

### Escenario 1: startup en la nube con menos de 100 solicitudes por segundo

Una startup con ese volumen no compra hardware: alquila. Los precios por hora son un rango de mercado y varían mucho entre proveedores.

**Tabla 3. Costo de alquilar un acelerador 24/7 durante 3 años**

| Opción | Tarifa por hora | Cálculo | Total 3 años |
|---|---|---|---|
| H100 en proveedor económico | 2,50 USD | 2,50 × 26.280 | 65.700 USD |
| H100 en el rango alto citado | 3,50 USD | 3,50 × 26.280 | 91.980 USD |
| TPU v5e (precio de lista us-west1) | 1,20 USD | 1,20 × 26.280 | 31.536 USD |

La H100 se cotiza entre 2,50 USD por hora en un proveedor económico (GPUPerHour, s. f.) y entre 2,75 y 3,50 USD en otro análisis de mercado (Cyfuture, 2026). El precio de lista de la TPU v5e es de 1,20 USD por chip-hora (iThome, 2023). A 100 solicitudes por segundo, una sola H100 queda casi ociosa: si cada solicitud fuera una inferencia tipo ResNet-50, el sistema H200 de 8 GPU atendió 480.131 consultas/s, o sea unas 60.000 por GPU. Eso significa que la startup estaría usando menos del 0,2 % de la capacidad. Para ella conviene un acelerador pequeño, instancias bajo demanda con apagado automático, o la TPU v5e si su modelo corre en JAX o TensorFlow, porque el costo base es menos de la mitad.

### Escenario 2: borde, un nodo con el modelo en el dispositivo

En el borde el capital y la energía pesan más que el costo por hora. Solo se calcula la energía a TDP máximo, porque es la parte que se puede sustentar con fuentes.

**Tabla 4. Energía de un nodo continuo durante 3 años (8,91 ¢/kWh)**

| Dispositivo | Potencia | Energía (kWh) | Costo de energía |
|---|---|---|---|
| NVIDIA T4 | 70 W | 1.839,6 | 163,90 USD |
| AMD Alveo U55C | 150 W (máx.) | 3.942,0 | 351,22 USD |
| NVIDIA H100 SXM | 700 W | 18.396,0 | 1.639,08 USD |

Sobre el capital, el único precio verificado es el de la H100: entre 25.000 y 30.000 USD en versión PCIe y entre 35.000 y 40.000 USD en SXM (Fluence, 2026), con un rango general de 30.000 a más de 40.000 USD según otra fuente (GMI Cloud, 2026). No se consiguió un precio fiable y citable para la T4 ni para la U55C, por lo que no se inventó uno. El resultado útil es relativo: en 3 años la energía de una H100 (unos 1.639 USD) es pequeña frente a su precio de compra, así que en el borde el argumento contra la H100 no es la factura eléctrica, sino que 700 W exigen refrigeración y fuente que un gabinete industrial normalmente no tiene.

## 4. Recomendación por perfil

### Perfil A: laboratorio universitario con 10.000 USD, visión y NLP hasta 7B

Un modelo de 7.000 millones de parámetros ocupa unos 14 GB en FP16 y unos 7 GB en INT8, solo en pesos, sin contar la caché de atención ni las activaciones. Criterios técnicos:

1. **Capacidad de memoria.** Una GPU de 16 GB como la T4 queda justa en FP16, y una de 24 GB o más da margen para contexto y lotes.
2. **Ecosistema de software.** CUDA, PyTorch y TensorRT cubren visión y NLP con la menor fricción para estudiantes.
3. **Presupuesto.** Una H100 de 25.000 a 30.000 USD (Fluence, 2026) supera el presupuesto, y una FPGA exige tiempo de desarrollo que un laboratorio docente no tiene.

La recomendación es una estación con una o dos GPU de consumo o de estación de trabajo con al menos 24 GB, y usar créditos de nube para los picos: con 2,50 USD por hora, 10.000 USD equivalen a 4.000 horas de H100 (10.000 / 2,50). Limitaciones: sin NVLink no se pueden repartir bien modelos grandes, y el ajuste fino de 7B completo no cabe, por lo que habría que usar LoRA o cuantización. No se cotizó una GPU concreta por falta de una fuente de precio verificable.

### Perfil B: inspección en manufactura, menos de 5 ms por imagen, 50 líneas

Aquí manda la latencia determinista. La recomendación para este perfil se discute en detalle en la Decisión 3. En resumen, se elige una GPU central H100 si las cámaras ya están en red y el jitter de red se controla, y FPGA distribuida solo si el requisito de determinismo por línea es estricto. Criterios: latencia máxima y jitter, tolerancia a fallos, costo de capital y facilidad de mantener el modelo.

### Perfil C: entrenamiento en la nube de un modelo de 70B en 512 aceleradores

Los pesos de 70.000 millones de parámetros en bf16 ocupan 140 GB, y con estado del optimizador el entrenamiento necesita más de un terabyte repartido, por lo que lo que manda es la red entre chips. Criterios:

1. **Interconexión.** El NVLink de la H100 da 900 GB/s por GPU dentro del nodo (NVIDIA, s. f.); la TPU v4 escala a 4.096 chips con conmutadores ópticos (Jouppi et al., 2023); un pod de TPU v5e tiene 256 chips y 1,6 TB/s de ancho de banda de bisección (Google Cloud, s. f.).
2. **Escala.** 512 chips superan un pod de v5e, por lo que habría que cruzar la red del centro de datos y perder eficiencia.
3. **Ecosistema.** Si el equipo ya usa PyTorch, un clúster de H100 con NVLink e InfiniBand evita reescribir el código; si usa JAX, una TPU v4 es una opción real.

Limitaciones: la disponibilidad de H100 y su precio por hora varían mucho (de 1,38 a 11,68 USD según DeployBase, 2026), y las TPU solo se alquilan en Google Cloud, lo que genera dependencia de un proveedor.
