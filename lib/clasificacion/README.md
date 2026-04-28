# Modelos de Clasificación

Este módulo contiene los modelos de machine learning y funciones de evaluación para la clasificación de migraña vestibular.

## Funciones Disponibles (ejemplo)

- `modelos.py` - Definición de clasificadores y configuración
- `evaluadores.py` - Funciones de evaluación y métricas de rendimiento

## Uso (ejemplo)

```python
from lib.clasificacion.modelos import crear_clasificador_svm
from lib.clasificacion.evaluadores import evaluar_rendimiento

# Crear clasificador
modelo = crear_clasificador_svm(kernel='rbf', C=1.0)

# Evaluar rendimiento
metricas = evaluar_rendimiento(modelo, X_test, y_test)
```
