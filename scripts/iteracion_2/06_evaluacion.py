import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

PATH_RESULTADOS = 'data/processed/analisis_resultados_it2.csv'
PATH_RANKING = 'scripts/iteracion_2/metricas_completas_it2.csv'

# Lectura de los biomarcadores con mayor rendimiento
df_rank = pd.read_csv(PATH_RANKING)
df_rank.columns = df_rank.columns.str.strip()
biomarcadores_seleccionados = ast.literal_eval(df_rank.iloc[0]['Variables'])

data_clinica = pd.read_csv(PATH_RESULTADOS)
data_clinica.columns = data_clinica.columns.str.strip()

X = data_clinica[biomarcadores_seleccionados]
y = data_clinica['Diagnostico']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Configuración del modelo definitivo
modelo_final = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_final.fit(X_train, y_train)

predicciones = modelo_final.predict(X_test)

etiquetas_diagnosticas = ["Migraña Vestibular", "Control"]
matriz_conf = confusion_matrix(y_test, predicciones, labels=etiquetas_diagnosticas)

verdaderos_positivos = matriz_conf[0, 0] 
falsos_negativos = matriz_conf[0, 1] 
falsos_positivos = matriz_conf[1, 0] 
verdaderos_negativos = matriz_conf[1, 1] 

sensibilidad = (verdaderos_positivos / (verdaderos_positivos + falsos_negativos)) * 100 if (verdaderos_positivos + falsos_negativos) > 0 else 0
especificidad = (verdaderos_negativos / (verdaderos_negativos + falsos_positivos)) * 100 if (verdaderos_negativos + falsos_positivos) > 0 else 0

# Reporte para análisis en la tesis
print("\n--- REPORTE DE VALIDACIÓN CLÍNICA ---")
print(f"Sensibilidad obtenida: {sensibilidad:.2f}% (Referencia Gufoni: 93.3%)")
print(f"Especificidad obtenida: {especificidad:.2f}% (Referencia Gufoni: 94.0%)")
print(f"Biomarcadores empleados: {', '.join(biomarcadores_seleccionados)}")

# Generación del gráfico de la matriz de confusión
figura, eje = plt.subplots(figsize=(8, 6))
visualizacion = ConfusionMatrixDisplay(confusion_matrix=matriz_conf, display_labels=etiquetas_diagnosticas)
visualizacion.plot(cmap='Blues', ax=eje, values_format='g', colorbar=False)

plt.title(f"Matriz de Confusión - Iteración 2\n({', '.join(biomarcadores_seleccionados)})")
plt.xlabel("Diagnóstico Predictivo (Inteligencia Artificial)")
plt.ylabel("Diagnóstico Real (Condición del Sujeto)")

plt.tight_layout()
plt.show()