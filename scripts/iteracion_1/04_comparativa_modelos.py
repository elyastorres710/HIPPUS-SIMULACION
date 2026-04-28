import pandas as pd
import numpy as np
import os
import ast
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#  Cargar datos y preparar etiquetas
df = pd.read_csv('data/processed/analisis_resultados.csv')
df.columns = df.columns.str.strip()

# Convertir diagnostico a valores numericos
df['target'] = df['Diagnostico'].map({'Control': 0, 'Migraña Vestibular': 1})

# Seleccionar variables y dividir grupos
import ast

if os.path.exists('scripts/iteracion_1/metricas_completas.csv'):
    df_rank = pd.read_csv('scripts/iteracion_1/metricas_completas.csv')
    # Saca el top 1 del ranking
    vars_top = ast.literal_eval(df_rank.iloc[0]['Variables'])
else:
    vars_top = ['Desviacion', 'Frecuencia_Dom']

X = df[vars_top]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#  Entrenamiento y test de cada algoritmo
lista_modelos = [
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('SVM', SVC(probability=True, random_state=42)),
    ('KMeans', KMeans(n_clusters=2, random_state=42, n_init=10))
]

res_final = []

for nombre, mod in lista_modelos:
    mod.fit(X_train, y_train)
    preds = mod.predict(X_test)

    # Ajuste manual para KMeans por si los clusters salen invertidos
    if nombre == 'KMeans':
        if accuracy_score(y_test, 1 - preds) > accuracy_score(y_test, preds):
            preds = 1 - preds

    # Guardar metricas una por una
    res_final.append({
        'Algoritmo': nombre,
        'Acc': round(accuracy_score(y_test, preds), 3),
        'Prec': round(precision_score(y_test, preds), 3),
        'Sens': round(recall_score(y_test, preds), 3),
        'F1': round(f1_score(y_test, preds), 3)
    })

#  Mostrar y guardar resultados
df_res = pd.DataFrame(res_final)
print("\n--- Tabla Comparativa de Metodos ---")
print(df_res.to_string(index=False))

df_res.to_csv('scripts/iteracion_1/comparativa_algoritmos.csv', index=False)
print("\nResultados guardados en: scripts/iteracion_1/comparativa_algoritmos.csv")  