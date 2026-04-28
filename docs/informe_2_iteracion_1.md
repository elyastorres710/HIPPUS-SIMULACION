# Informe 2 Iteración 1

## 1. MÓDULO DE SEÑALES Y GENERACIÓN DE DATOS (01_generacion_datos.py)
Se han integrado parámetros de "realismo sucio" para evitar señales ideales:

- RUIDO ROSA (1/f): Implementación de ruido con decaimiento espectral para simular la vigilancia pupilar basal y el ruido biológico real.
- ARTEFACTOS DE PARPADEO: Inserción aleatoria de pérdida de señal (ceros) para testear la robustez del análisis ante fallos del sensor.
- CRITERIO MCLAREN (SOMNOLENCIA): Se configuró un solapamiento rítmico en controles sanos (banda 0.04-0.6 Hz) para simular hippus por fatiga.
- CASOS GRISES: Se ajustó la probabilidad de PNy débil en el grupo patológico para forzar errores necesarios y replicar la sensibilidad clínica real. 

## 2. MÓDULO DE ANÁLISIS DE DATOS (02_analisis_datos.py)
Refinamiento del procesamiento digital de señales:

- PUAL (Pupillary Unrest Alertness Level): Cálculo de energía espectral mediante FFT para detección de nistagmo pupilar (PNy).

## 3. MÓDULO DE VISUALIZACIÓN 3D (05_visualizacion.py)
- GRÁFICO 3D: Proyección en 3 ejes basada en el Top 3 del ranking (PUAL, Dfi, Frecuencia).
- CLÚSTERES: Visualización física de la separación de grupos y de los puntos de solapamiento.

## 4. MÓDULO DE EVALUACIÓN (06_evaluacion.py)
- Se alcanzó similitud con los resultados del estudio de referencia (Gufoni y Casani, 2023).
- SENSIBILIDAD ACTUAL: ~96.15% (Objetivo Gufoni: 93.3%)
- ESPECIFICIDAD ACTUAL: ~91.67% (Objetivo Gufoni: 94.0%) 







