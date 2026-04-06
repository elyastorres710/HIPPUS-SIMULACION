# Observaciones - 2026-04-02

## Observación Crítica: Error en Pipeline de Ruido

**Archivos:** `scripts/iteracion_2/evaluacion.py` y `scripts/iteracion_2/clasificacion.py`  
**Líneas:** 12-17 en ambos archivos

### Problema
Se está agregando ruido a las **medidas estadísticas ya calculadas** (Media, RMS, PUI, Dfi, etc.). Esto es metodológicamente incorrecto porque:

- Las medidas son valores **matemáticamente exactos** derivados de la señal
- Agregar ruido post-hoc simula una variabilidad que no existe en la realidad del proceso de medición
- Invalida la interpretación fisiológica de los resultados

### Flujo Incorrecto (Actual)
```
Señal cruda -> Extraer medidas -> Medidas exactas -> +20% ruido -> Clasificar (INCORRECTO)
```

### Flujo Correcto
```
Señal cruda -> +ruido de medición -> Señal con ruido -> Extraer medidas -> Clasificar (CORRECTO)
```

### Recomendación
El ruido debe aplicarse en la etapa de **generación de datos** (`generacion_datos.py`) o en el **procesamiento de señales** (`analisis.py`), nunca a las métricas finales.

---

## Observación de Orden: Ubicación de Archivos

**Archivos afectados:** `scripts/iteracion_2/clasificacion.py`, `scripts/iteracion_2/evaluacion.py`

### Problema
Los archivos creados están almacenados en la carpeta `iteracion_2/`, pero el flujo de trabajo actual aún corresponde a la **Iteración 1**:

```
Simulación -> Calculo de medidas -> Metricas -> Clasificacion -> Evaluacion
```

Este pipeline completo pertenece a la Iteración 1. La Iteración 2 debería iniciar solo después de completar y validar la Iteración 1.

### Recomendación
Mover los archivos de `scripts/iteracion_2/` a `scripts/iteracion_1/` hasta que se finalice completamente la primera iteración del proyecto.

---

## Observación de Resultados: Accuracy Sospechosamente Alta

**Archivos:** `scripts/iteracion_2/clasificacion.py`, `scripts/iteracion_2/evaluacion.py`

### Problema
Los resultados muestran accuracy de 1.0 (100%) en múltiples combinaciones de variables:

```
  combinacion                                         RF   SVM   KMeans
[Media, Desviacion, RMS, PUI]                        1.0   1.0   0.995
[PUI, Dfi, Frecuencia_Dom]                           1.0   1.0   0.995
...
```

Esto es estadísticamente poco realista y probablemente producto de:
- Sobre-simplificación del modelo generador de datos (`generacion_datos.py`)
- Falta de overlap entre clases en el espacio de features
- Datos sintéticos con separación lineal perfecta

### Recomendación
Es necesario generar visualizaciones de los datos para verificar la distribución real:
- **Scatter plots 2D:** para combinaciones de 2 variables
- **Scatter plots 3D:** para combinaciones de 3 variables
- **Reducción dimensional (PCA/t-SNE):** si se usan más de 3 variables, para verificar separabilidad

Las visualizaciones permitirán detectar si las clases están artificialmente separadas.

---

## Observación Positiva: Acierto Metodológico

**Archivos:** `scripts/iteracion_2/clasificacion.py`, `scripts/iteracion_2/evaluacion.py`

### Aspectos Correctos
La estructura del código presenta un **acierto metodológico importante** en la separación de responsabilidades:

1. **Clasificación con múltiples modelos y combinaciones:**
   - Probar múltiples clasificadores (RF, SVM, KMeans)
   - Evaluar todas las combinaciones posibles de features
   - Esto permite identificar sistemáticamente la mejor configuración

2. **Evaluación separada con la mejor configuración:**
   - Usar la mejor combinación encontrada (`mejor_combo`)
   - Aplicar el mejor clasificador (Random Forest)
   - Generar métricas detalladas (classification report, matriz de confusión)

### Ventaja
Esta separación facilita el análisis detallado de resultados y permite comparar el desempeño de diferentes métodos de forma estructurada.

### Mejora Pendiente
Queda profundizar la comparación de los diferentes métodos utilizando métricas de clasificación completas:
- Precision
- Recall
- F1-score
- AUC-ROC

Actualmente solo se reporta accuracy en `clasificacion.py`, lo cual limita el análisis comparativo.

---

## Pendientes

- [ ] Mover el pipeline de ruido a la señal cruda, no a las medidas extraídas
- [ ] Mover archivos de iteracion_2 a iteracion_1 (flujo aún pertenece a Iteración 1)
- [ ] Generar scatter plots 2D/3D para verificar separabilidad de clases
- [ ] (Siguiente iteración) Evaluar realismo del generador de datos sintéticos
