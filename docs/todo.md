# Tareas pendientes

1. [ ] Definir medidas básicas para la extracción (por ejemplo: RMS, amplitud, parametros estadísticos, cuantificar el jitter, etc.)
2. [ ] Definir las funciones y que parámetros recibirán (funcion (recibe señal) -> retorna medida ... una medida por función)
3. [ ] Implementar las funciones
4. [ ] generar un script que sea capaz de generar y almacenar múltiples señales
5. [ ] generar un script que sea capaz de cargar múltiples señales y extraer las medidas


## observaciones:

1. una medida por función: cada función debe retornar un único valor numérico (la medida calculada).
    ej: calcular_media(signal) -> retorna un valor numérico
        calcular_desviacion_estandar(signal) -> retorna un valor numérico
        calcular_rms(signal) -> retorna un valor numérico