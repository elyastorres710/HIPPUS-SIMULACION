# Scripts de Ejecución

Esta carpeta contiene los scripts que ejecutan las diferentes iteraciones del proyecto.

## Estructura (ejemplo)

- `iteracion_1/` - Scripts para la primera fase (datos sintéticos bogus)
- `iteracion_2/` - Scripts para la segunda fase (con modelo matemático)

## Uso (ejemplo)

Cada carpeta de iteración contiene scripts independientes que ejecutan el ciclo completo:

1. **generacion_datos.py** - Crea los datasets de entrenamiento
2. **analisis.py** - Analiza los datos generados
3. **clasificacion.py** - Aplica modelos de machine learning
4. **evaluacion.py** - Evalúa los resultados obtenidos

## Ejecución

Para ejecutar una iteración completa:

```bash
cd scripts/iteracion_1
python generacion_datos.py
python analisis.py
python clasificacion.py
python evaluacion.py
```

Cada script es autocontenido y puede ser ejecutado de forma independiente.
