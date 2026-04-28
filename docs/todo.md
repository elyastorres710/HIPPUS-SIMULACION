# Tareas pendientes
1. [x] Separación de grupo de entrenamiento y grupo de prueba
2. [x] Definir el modelo de aprendizaje automático para clasificar las señales (ej: k-means, SVM, Random Forest, etc.)
3. [x] Pruebas con diferentes combinaciones de medidas (ej: test1: utilizar RMS, amplitud, frecuencia fundamental; test2: agregar parámetros estadísticos; test 3: eliminar un parámetro y agregar PUI, etc.) con el o los modelos de clasificación que hayas implementado
4. [x] Presentar los resultados de las pruebas
5. [x] Evaluar los resultados de la clasificación con el grupo de prueba (gráficamente y matemáticamente utilizando métricas de evaluación [matriz de confusión + mediciones])

# Tareas completadas
1. [x] Definir medidas básicas para la extracción (por ejemplo: RMS, amplitud, parametros estadísticos, cuantificar el jitter, etc.)
2. [x] Definir las funciones y que parámetros recibirán (funcion (recibe señal) -> retorna medida ... una medida por función)
3. [x] Implementar las funciones
4. [x] generar un script que sea capaz de generar y almacenar múltiples señales
5. [x] generar un script que sea capaz de cargar múltiples señales y extraer las medidas

# Objetivos Específicos
1. [x] OE1: modelar la señal sintética del hippus pupilar
2. [x] OE2: Extraer los parámetros (medidas) de la señal sintética
3. [ ] OE3: Seleccionar y validar un modelo de aprendizaje automático para clasificar las señales

## observaciones:

1. una medida por función: cada función debe retornar un único valor numérico (la medida calculada).
    ej: calcular_media(signal) -> retorna un valor numérico
        calcular_desviacion_estandar(signal) -> retorna un valor numérico
        calcular_rms(signal) -> retorna un valor numérico
