# Iteración 1 - Datos Sintéticos (ejemplo)

Esta carpeta contiene los scripts para la primera iteración del proyecto, enfocada en la generación de datos sintéticos (bogus) sin considerar el modelo matemático del sistema vestibular-pupilar.

## Scripts

- `generacion_datos.py` - Generación de datasets sintéticos
- `analisis.py` - Análisis exploratorio de los datos generados
- `clasificacion.py` - Aplicación de modelos de machine learning
- `evaluacion.py` - Evaluación de resultados y métricas

## Ejecución

```bash
# Ejecutar la iteración completa
python generacion_datos.py
python analisis.py
python clasificacion.py
python evaluacion.py
```

## Características

- **Datos**: Señales sintéticas generadas artificialmente
- **Enfoque**: Probar el pipeline de análisis sin modelo matemático real
- **Objetivo**: Validar la metodología y detectar problemas tempranos
