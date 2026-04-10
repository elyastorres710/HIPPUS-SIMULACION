import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# 1 Cargar datos y ranking
PATH_RESULTADOS = 'data/processed/analisis_resultados.csv'
PATH_RANKING = 'scripts/iteracion_1/metricas_completas.csv'

if not os.path.exists(PATH_RANKING):
    print("Error: No se encontro el ranking. Corre primero el script 05.")
    exit()

# Leer automaticamente los mejores biomarcadores
df_rank = pd.read_csv(PATH_RANKING)
df_rank.columns = df_rank.columns.str.strip()
BIOMARCADORES_SELECCIONADOS = ast.literal_eval(df_rank.iloc[0]['Variables'])

# 2 Preparar datos
data_hippus = pd.read_csv(PATH_RESULTADOS)
data_hippus.columns = data_hippus.columns.str.strip()

X = data_hippus[BIOMARCADORES_SELECCIONADOS]
y = data_hippus['Diagnostico']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 3  Entrenamiento final
modelo_mv = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_mv.fit(X_train, y_train)

# 4  Prediccion y reporte
predicciones = modelo_mv.predict(X_test)

print(">>> REPORTE DE EVALUACION FINAL")
print(f"Variables utilizadas (Top Ranking): {BIOMARCADORES_SELECCIONADOS}")
print(classification_report(y_test, predicciones))

# 5  Matriz de Confusion
fig, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, 
    predicciones, 
    cmap='Blues', 
    ax=ax,
    colorbar=False
)

plt.title(f"Matriz de Confusion Final\nVariables: {', '.join(BIOMARCADORES_SELECCIONADOS)}")
plt.tight_layout()
plt.show()

