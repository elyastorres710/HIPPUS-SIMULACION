"""
Módulo de análisis estadístico de señales pupilares.
"""

from .estadisticas import (
    calcular_media_pupilar,
    calcular_desviacion_estandar,
    calcular_rms,
    calcular_pui,
    calcular_dfi,
    calcular_velocidad_promedio,
    calcular_frecuencia_dominante,
    calcular_pual,        
    calcular_pual_ratio   
)

__all__ = [
    'calcular_media_pupilar',
    'calcular_desviacion_estandar',
    'calcular_rms',
    'calcular_pui',
    'calcular_dfi',
    'calcular_velocidad_promedio',
    'calcular_frecuencia_dominante',
    'calcular_pual',
    'calcular_pual_ratio'
]
