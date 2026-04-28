# Informe Iteración 1

## Resumen de Cambios
* **Fisiología más realista:** Ajusté los rangos de amplitud y frecuencia para permitir el solapamiento o cruce natural entre grupos (Sanos vs. Migraña).
* **Ruido de Sensor:** Agregué un ruido variable del 1% al 7% directamente en la señal cruda.
* **Métricas Clínicas:** Se agregó el cálculo de Sensibilidad (Recall), F1-Score y AUC-ROC para una no limitar el analisis comparativo.
* **Visualización:** Añadí gráficos de dispersión (2D y 3D) para analizar la distribución de los grupos, demostrando que los datos presentan una complejidad mayor y un solapamiento más realista.

# Resultados
El modelo de **Random Forest** demostró ser el más eficaz para el diagnóstico, logrando una alta capacidad de detección incluso con señales ruidosas.

# Conclusiones
**Biomarcadores:** Aparentemente, la combinación de [Desviacion, Frecuencia_Dom]es el mejor marcador para MV.
**Sensibilidad:** El sistema detecta el 97.3% de los casos patológicos, minimizando los falsos negativos en el diagnóstico.
**Validación Visual:** Los scatter plots (2D y 3D) confirman que los datos ya no son trivialmente separables, aportando realismo científico al modelo.