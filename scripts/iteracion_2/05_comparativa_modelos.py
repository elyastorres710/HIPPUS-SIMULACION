import pandas as pd
import numpy as np 
import os
import ast
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ruta_entrada_datos = 'data/processed/analisis_resultados_it2.csv'
ruta_entrada_ranking = 'scripts/iteracion_2/metricas_completas_it2.csv'
ruta_salida = 'scripts/iteracion_2/comparativa_algoritmos_it2.csv'

df = pd.read_csv(ruta_entrada_datos)
df.columns = df.columns.str.strip()
df['target'] = df['Diagnostico'].map({'Control': 0, 'Migraña Vestibular': 1})

# Selección automatizada de los mejores biomarcadores
if os.path.exists(ruta_entrada_ranking):
    df_rank = pd.read_csv(ruta_entrada_ranking)
    variables_optimas = ast.literal_eval(df_rank.iloc[0]['Variables'])
else:
    variables_optimas = ['Desviacion', 'Frecuencia_Dom']

X = df[variables_optimas]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelos_evaluacion = [
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('Support Vector Machine', SVC(probability=True, random_state=42)),
    ('K-Means', KMeans(n_clusters=2, random_state=42, n_init=10))
]

resultados_comparativa = []

for nombre_modelo, modelo in modelos_evaluacion:
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)

    # Ajuste de orientación para modelo no supervisado
    if nombre_modelo == 'K-Means':
        if accuracy_score(y_test, 1 - predicciones) > accuracy_score(y_test, predicciones):
            predicciones = 1 - predicciones

    resultados_comparativa.append({
        'Algoritmo': nombre_modelo,
        'Exactitud': round(accuracy_score(y_test, predicciones), 3),
        'Precision': round(precision_score(y_test, predicciones), 3),
        'Sensibilidad': round(recall_score(y_test, predicciones), 3),
        'F1_Score': round(f1_score(y_test, predicciones), 3)
    })

df_resultados = pd.DataFrame(resultados_comparativa)
print("\n--- Comparativa de Algoritmos de Clasificación ---")
print(df_resultados.to_string(index=False))

df_resultados.to_csv(ruta_salida, index=False)
print(f"\nDocumento de comparativa guardado en: {ruta_salida}")