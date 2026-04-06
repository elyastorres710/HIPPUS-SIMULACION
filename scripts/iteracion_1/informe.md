# Informe Iteración 1

## Resumen de Cambios
* **Fisiología mas realista:** Se ajustaron los rangos de amplitud y frecuencia para permitir el solapamiento o cruce natural entre grupos (Sanos vs. Migraña).
* **Ruido de Sensor:** Agregué un ruido variable del 1% al 7% directamente en la señal cruda.
* **Métricas Clínicas:** Se agregó el cálculo de Sensibilidad (Recall), F1-Score y AUC-ROC para una no limitar el analisis comparativo.
* **Visualización:** Los scatter plots (2D y 3D) confirman que los datos son mas complejos y separables

# Resultados
El modelo de **Random Forest** demostró ser el más eficaz para el diagnóstico, logrando una alta capacidad de detección incluso con señales ruidosas.

# Conclusiones
**Biomarcadores:** Aparentemente, la combinación de [Desviacion, Frecuencia_Dom]es el mejor marcador para MV.
**Sensibilidad:** El sistema detecta el 97.3% de los casos patológicos, minimizando los falsos negativos en el diagnóstico.
