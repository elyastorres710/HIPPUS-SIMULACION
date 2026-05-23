import pandas as pd
import numpy as np
import itertools
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score
)

# Configuración de rutas
ruta_entrada = 'data/processed/analisis_resultados_it2.csv'
ruta_salida = 'scripts/iteracion_2/metricas_completas_it2.csv'

# Preparación de los datos
datos = pd.read_csv(ruta_entrada)
datos.columns = datos.columns.str.strip()
datos['clase'] = datos['Diagnostico'].map({'Control': 0, 'Migraña Vestibular': 1})

variables_analisis = ['Media', 'Desviacion', 'RMS', 'PUI', 'PUAL', 'PUAL_Ratio', 'Dfi', 'Velocidad_Media', 'Frecuencia_Dom']
X = datos[variables_analisis]
y = datos['clase']

# División de la muestra (Entrenamiento 70% / Prueba 30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ranking_combinaciones = []

# Evaluación de combinaciones de variables
for r in range(1, 4):
    for combinacion in itertools.combinations(variables_analisis, r):
        columnas = list(combinacion)
        
        modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo_rf.fit(X_train[columnas], y_train)
        
        predicciones = modelo_rf.predict(X_test[columnas])
        probabilidades = modelo_rf.predict_proba(X_test[columnas])[:, 1]
        
        ranking_combinaciones.append({
            'Variables': columnas,
            'Exactitud': round(accuracy_score(y_test, predicciones), 3),
            'Precision': round(precision_score(y_test, predicciones), 3),
            'Sensibilidad': round(recall_score(y_test, predicciones), 3),
            'F1_Score': round(f1_score(y_test, predicciones), 3),
            'AUC': round(roc_auc_score(y_test, probabilidades), 3)
        })

df_ranking = pd.DataFrame(ranking_combinaciones).sort_values(by='F1_Score', ascending=False)

print("\n--- Desempeño de Combinaciones de Biomarcadores (Ordenado por F1-Score) ---")
print(df_ranking.head(10).to_string(index=False))

os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
df_ranking.to_csv(ruta_salida, index=False)
print(f"\nDocumento de métricas guardado en: {ruta_salida}")