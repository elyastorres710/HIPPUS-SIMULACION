# Datos del Proyecto

Esta carpeta contiene todos los datos generados y utilizados durante el desarrollo del proyecto.

## Estructura (ejemplo)

- `raw/` - Datos originales tal como se generan o se reciben
- `processed/` - Datos después de limpieza y transformación
- `results/` - Resultados de modelos y análisis

## Flujo de Datos (ejemplo)

1. **Generación**: Los datos se crean en `raw/`
2. **Procesamiento**: Se limpian y transforman a `processed/`
3. **Resultados**: Los análisis y salidas se guardan en `results/`

## Uso (ejemplo)

Cada script de ejecución sabe dónde guardar y leer los datos según la etapa del proceso:

```python
# Datos generados
data_path = "data/raw/dataset_iteracion_1.csv"

# Datos procesados
processed_path = "data/processed/dataset_limpio.csv"

# Resultados
results_path = "data/results/metricas_evaluacion.json"
```
