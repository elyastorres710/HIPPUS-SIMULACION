# Herramientas de Análisis

Este módulo contiene funciones para el análisis estadístico y cálculo de métricas de las señales pupiliares.

## Funciones Disponibles (ejemplo)

- `estadisticas.py` - Cálculo de métricas estadísticas básicas
- `metricas.py` - Métricas específicas del dominio de hippus pupilar

## Uso

```python
from lib.analisis.estadisticas import calcular_estadisticas_basicas
from lib.analisis.metricas import calcular_indice_hippus

# Calcular estadísticas básicas
stats = calcular_estadisticas_basicas(senal)

# Calcular métricas específicas
indice_hippus = calcular_indice_hippus(senal, frecuencia=1000)
```
