# Estructura del Proyecto HIPPUS-SIMULACION

## Introducción

Este documento explica la estructura organizacional del proyecto HIPPUS-SIMULACION, diseñada para facilitar un desarrollo iterativo, modular y mantenible. La arquitectura propuesta sigue las mejores prácticas de ingeniería de software y se alinea perfectamente con las directivas establecidas para el proyecto.

## Filosofía de Diseño

### Separación de Responsabilidades

La estructura se basa en un principio fundamental: **separar el código que ejecuta (scripts) del código que reutiliza (librerías)**. Esta distinción nos permite:

- **Mantener la independencia** entre las diferentes iteraciones del proyecto
- **Reutilizar componentes** de manera eficiente entre ciclos
- **Facilitar el mantenimiento** y evolución del código base
- **Garantizar la trazabilidad** de los experimentos y resultados

### Modularidad por Funcionalidad

Cada componente del sistema tiene un propósito bien definido y vive en su propio espacio, lo que nos permite entender, modificar y probar cada pieza de forma aislada.

## Estructura de Carpetas

```
HIPPUS-SIMULACION/
├── scripts/                    # Código de ejecución por iteración
│   ├── iteracion_1/           # Primera fase del proyecto
│   │   ├── generacion_datos.py
│   │   ├── analisis.py
│   │   ├── clasificacion.py
│   │   └── evaluacion.py
│   └── iteracion_2/           # Segunda fase con modelo matemático
│       ├── generacion_datos.py
│       ├── analisis.py
│       ├── clasificacion.py
│       └── evaluacion.py
├── lib/                        # Código reutilizable
│   ├── generadores/           # Componentes de generación de datos
│   │   ├── signals.py         # Generación de señales pupiales
│   │   └── noise.py           # Modelos de ruido y perturbaciones
│   ├── analisis/              # Herramientas de análisis
│   │   ├── estadisticas.py    # Cálculo de métricas estadísticas
│   │   └── metricas.py        # Métricas específicas del dominio
│   ├── clasificacion/         # Modelos y algoritmos
│   │   ├── modelos.py         # Definición de clasificadores
│   │   └── evaluadores.py     # Funciones de evaluación
│   └── utils/                 # Utilidades generales
│       ├── visualizacion.py   # Gráficos y reportes
│       └── configuracion.py   # Gestión de parámetros
├── data/                       # Almacenamiento de datos
│   ├── raw/                   # Datos originales y generados
│   ├── processed/             # Datos procesados y limpios
│   └── results/               # Resultados de experimentos
├── docs/                      # Documentación del proyecto
│   ├── directivas.md          # Directrices de desarrollo
│   ├── estructura.md          # Este documento
│   └── reportes/              # Reportes por iteración
└── tests/                     # Pruebas unitarias
    ├── test_generadores.py
    ├── test_analisis.py
    └── test_clasificacion.py
```

## Detalle de Componentes

### Scripts: El Motor de Ejecución

Los scripts representan el **flujo de trabajo concreto** de cada iteración. Cada archivo en esta carpeta es un programa completo que:

1. **Importa las funciones necesarias** desde las librerías
2. **Configura los parámetros específicos** de la iteración
3. **Ejecuta el ciclo completo** de análisis
4. **Guarda los resultados** en las carpetas correspondientes

**¿Por qué esta separación?**

- **Reproducibilidad:** Cada iteración puede ser ejecutada de forma independiente
- **Experimentación:** Podemos modificar parámetros sin afectar otras iteraciones
- **Historial:** Mantenemos un registro exacto de qué se hizo en cada fase
- **Colaboración:** Differentes personas pueden trabajar en iteraciones distintas

### Lib: El Corazón Reutilizable

La carpeta `lib` contiene todo el **intelecto reusable** del proyecto. Aquí vive el código que:

- **Resuelve problemas específicos** del dominio
- **Puede ser utilizado** en múltiples iteraciones
- **Evoluciona independientemente** de los experimentos
- **Se puede probar** de forma aislada

**Subdivisiones por funcionalidad:**

- **generadores/**: Todo lo relacionado con crear datos
- **analisis/**: Herramientas para entender los datos
- **clasificacion/**: Modelos de machine learning
- **utils/**: Funcionalidades de soporte

### Data: La Memoria del Proyecto

Los datos tienen su propio ciclo de vida:

1. **raw/**: Datos tal como se generan o se reciben
2. **processed/**: Datos después de limpieza y transformación
3. **results/**: Salidas de modelos y análisis

Esta separación nos permite:
- **Entender la procedencia** de cada conjunto de datos
- **Reproducir pipelines** de procesamiento
- **Optimizar el almacenamiento** y acceso

## Flujo de Trabajo Típico

### Desarrollo de una Nueva Iteración

1. **Crear la carpeta de la iteración** en `scripts/`
2. **Desarrollar funciones específicas** si es necesario en `lib/`
3. **Escribir el script principal** que orquesta el proceso
4. **Ejecutar y validar** resultados
5. **Documentar aprendizajes** en `docs/reportes/`

### Reutilización de Código

Cuando una función desarrollada para una iteración puede ser útil en otras:

1. **Mover la función** a `lib/` en el módulo apropiado
2. **Crear pruebas unitarias** en `tests/`
3. **Actualizar los scripts** existentes para usar la nueva función
4. **Documentar** su uso y propósito

---

*Última actualización: Marzo 2026*
