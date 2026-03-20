# Pruebas Unitarias

Esta carpeta contiene las pruebas unitarias para validar el funcionamiento de las librerías del proyecto.

## Estructura (ejemplo)

- `test_generadores.py` - Pruebas para funciones de generación de datos
- `test_analisis.py` - Pruebas para herramientas de análisis
- `test_clasificacion.py` - Pruebas para modelos de clasificación

## Uso

Para ejecutar todas las pruebas:

```bash
python -m pytest tests/
```

Para ejecutar pruebas específicas:

```bash
python -m pytest tests/test_generadores.py
python -m pytest tests/test_analisis.py -v
```

## Desarrollo

Al agregar nuevas funciones a las librerías:

1. Crea pruebas correspondientes en esta carpeta
2. Asegura que las pruebas cubran los casos principales
3. Ejecuta las pruebas antes de hacer commit
4. Mantén una cobertura de código del 80% como mínimo

## Convenciones

- Los nombres de archivos de prueba deben empezar con `test_`
- Las funciones de prueba deben empezar con `test_`
- Usa assertions claros y descriptivos
- Prueba casos normales, edge cases y casos de error
