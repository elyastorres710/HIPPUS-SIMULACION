# Librerías Reutilizables

Esta carpeta contiene el código reutilizable que será utilizado por las diferentes iteraciones del proyecto.

## Estructura (ejemplo)

- `generadores/` - Funciones para generación de señales y datos
- `analisis/` - Herramientas de análisis estadístico y métricas
- `clasificacion/` - Modelos y algoritmos de machine learning
- `utils/` - Utilidades generales (configuración, visualización)

## Uso (ejemplo)

Las librerías son importadas por los scripts de ejecución:

```python
from lib.generadores.signals import generar_senal_pupilar
from lib.analisis.estadisticas import calcular_metricas
from lib.clasificacion.modelos import crear_clasificador
from lib.utils.visualizacion import graficar_resultados
```

## Desarrollo

Al agregar nuevas funciones:

1. Colócalas en el módulo apropiado según su funcionalidad
2. Sigue las directivas del proyecto (documentación, tipado, etc.)
3. Crea pruebas unitarias en la carpeta `tests/`
4. Actualiza la documentación si es necesario
