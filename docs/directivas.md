# Directivas del Proyecto

> **Propósito del documento:** Establecer las directrices metodológicas y reglas de desarrollo para la construcción del sistema de clasificación de migraña vestibular basado en análisis de hippus pupilar.

---

## Objetivo del Proyecto

Desarrollar un sistema de **clasificación automática de migraña vestibular** utilizando técnicas de **aprendizaje de máquina explicables (XAI)**, fundamentado en el análisis del **hippus pupilar**. El sistema empleará **simulaciones de modelos matemáticos** que representan la dinámica del sistema vestibular-pupilar para generar datos de entrenamiento y validación.

---

## Metodología de Desarrollo por Iteraciones

El proyecto seguirá una metodología **iterativa e incremental**, donde cada iteración representa un ciclo completo de análisis, desarrollo y evaluación. Cada iteración generará conocimientos que alimentarán la siguiente fase del proyecto.

### Fases del Proyecto

**Iteración 1:** Desarrollo de sistema de señales sintéticas
- Enfoque principal: Generación de datos bogus sin modelo matemático del sistema vestibular-pupilar

**Iteración 2:** Desarrollo avanzado con modelo matemático  
- Enfoque principal: Integración del modelo matemático del sistema vestibular-pupilar

---

## Ciclo de Desarrollo

Cada iteración sigue un **ciclo estructurado** que garantiza la calidad y trazabilidad de los resultados:

**Flujo del Ciclo de Desarrollo:**

1. **Generación de Datos** -> 2.
2. **Análisis y Métricas** -> 3.
3. **Clasificación ML** -> 4.
4. **Evaluación de Resultados** -> 5.
5. **Presentación** -> 6.
6. **Implementación Siguiente Iteración** -> 7.
7. **Volver a Generación de Datos** -> 1

### Etapas del Ciclo

1. **Generación de Datos**
   - Creación de datasets de entrenamiento y clasificación
   - Validación de calidad y representatividad

2. **Análisis**
   - Análisis exploratorio de datos generados
   - Cálculo de métricas estadísticas y descriptivas

3. **Clasificación**
   - Aplicación de diversos algoritmos de aprendizaje de máquina
   - Optimización de hiperparámetros

4. **Evaluación**
   - Medición de rendimiento y precisión
   - Análisis de explicabilidad del modelo

5. **Presentación**
   - Documentación de resultados obtenidos
   - Visualización de métricas y conclusiones

6. **Implementación**
   - Planificación de la siguiente iteración
   - Incorporación de lecciones aprendidas

---

## Reglas de Desarrollo

### Principios de Diseño de Código

**Atomicidad**
- Descripción: Cada función debe tener un único propósito
- Aplicación: Funciones pequeñas y específicas

**Nombres Descriptivos**
- Descripción: Nombres que reflejen claramente su propósito
- Aplicación: Auto-documentación del código

**Inmutabilidad**
- Descripción: Sin efectos secundarios en estado global
- Aplicación: Funciones puras y predecibles

**Documentación Completa**
- Descripción: Docstrings detallados en español
- Aplicación: Propósito, parámetros y retornos

**Tipado Estricto**
- Descripción: Definición explícita de tipos
- Aplicación: Type hints para mayor robustez

**Manejo de Errores**
- Descripción: Captura y gestión de excepciones
- Aplicación: Código resiliente y robusto

### Estándares de Calidad

- **Documentación en español**: Toda la documentación debe estar en español
- **Variables Descriptivas**: Nombres que indiquen su contenido y propósito
- **Parametrización**: Constantes en archivos de configuración
- **Separación de Responsabilidades**: Lógica de negocio separada de presentación
- **Configuración por Parámetros**: Variables de configuración pasadas como parámetros
- **Independencia de Etapas**: Cada etapa ejecutable de forma individual
- **Visualización Clara**: Resultados presentados de forma comprensible

### Mejores Prácticas

```python
# Ejemplo de estructura de función según las directivas
def generar_datos_pupiales(
    cantidad_muestras: int,
    frecuencia_muestreo: float = 1000.0,
    ruido_base: float = 0.1
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Genera datos simulados de hippus pupilar para entrenamiento.
    
    Args:
        cantidad_muestras: Número de muestras a generar
        frecuencia_muestreo: Frecuencia en Hz para el muestreo
        ruido_base: Nivel de ruido base a añadir a la señal
        
    Returns:
        Tuple[datos, metadatos]: Array con datos y diccionario con metadatos
        
    Raises:
        ValueError: Si los parámetros de entrada no son válidos
    """
    # Implementación...
```

---

## Métricas de Éxito

- **Precisión del modelo**: > 85% en conjunto de prueba
- **Explicabilidad**: Métricas SHAP interpretables
- **Reproducibilidad**: Mismos resultados con misma semilla
- **Documentación**: 100% de funciones documentadas
- **Cobertura de pruebas**: > 80% del código

---

*Última actualización: Marzo 2026*
