# Generadores de Señales

Este módulo contiene funciones para la generación de señales pupiliares y datos sintéticos.

## Funciones Disponibles (ejemplo)

- `signals.py` - Generación de señales pupiliares simuladas
- `noise.py` - Modelos de ruido y perturbaciones

## Uso

```python
from lib.generadores.signals import generar_senal_pupilar
from lib.generadores.noise import agregar_ruido_gaussiano

# Generar señal base
senal = generar_senal_pupilar(duracion=10, frecuencia=1000)

# Agregar ruido
senal_con_ruido = agregar_ruido_gaussiano(senal, sigma=0.1)
```
