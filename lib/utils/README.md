# Utilidades Generales

Este módulo contiene funcionalidades de soporte para el proyecto.

## Funciones Disponibles (ejemplo)

- `visualizacion.py` - Funciones para gráficos y reportes
- `configuracion.py` - Gestión de parámetros y configuración

## Uso

```python
from lib.utils.visualizacion import graficar_senal_pupilar
from lib.utils.configuracion import cargar_configuracion

# Graficar señal
graficar_senal_pupilar(senal, tiempo, titulo="Hippus Pupilar")

# Cargar configuración
config = cargar_configuracion("config.yaml")
```
