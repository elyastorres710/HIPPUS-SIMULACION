import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# CONFIGURACION
PATH_RESULTADOS = 'data/processed/analisis_resultados.csv'
PATH_RANKING = 'scripts/iteracion_1/metricas_completas.csv'

# Leer automáticamente los mejores biomarcadores del ranking
df_rank = pd.read_csv(PATH_RANKING)
df_rank.columns = df_rank.columns.str.strip()
BIOMARCADORES_SELECCIONADOS = ast.literal_eval(df_rank.iloc[0]['Variables'])

# PREPARACION DE DATOS
data_hippus = pd.read_csv(PATH_RESULTADOS)
data_hippus.columns = data_hippus.columns.str.strip()

X = data_hippus[BIOMARCADORES_SELECCIONADOS]
y = data_hippus['Diagnostico']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ENTRENAMIENTO DEL MODELO      
modelo_mv = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_mv.fit(X_train, y_train)

# PREDICCIÓN Y CÁLCULO DE MÉTRICAS
predicciones = modelo_mv.predict(X_test)

etiquetas = ["Migraña Vestibular", "Control"]
cm = confusion_matrix(y_test, predicciones, labels=etiquetas)

vp = cm[0, 0] 
fn = cm[0, 1] 
fp = cm[1, 0] 
vn = cm[1, 1] 

sensibilidad = (vp / (vp + fn)) * 100 if (vp + fn) > 0 else 0
especificidad = (vn / (vn + fp)) * 100 if (vn + fp) > 0 else 0

# TERMINAL
print("\nRESULTADOS DEL ANÁLISIS")
print(f"Sensibilidad: {sensibilidad:.2f}% (Gufoni: 93.3%)")
print(f"Especificidad: {especificidad:.2f}% (Gufoni: 94.0%)")
print(f"Variables: {', '.join(BIOMARCADORES_SELECCIONADOS)}")

# VISUALIZACIÓN DE LA MATRIZ
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=etiquetas)
disp.plot(cmap='Blues', ax=ax, values_format='g', colorbar=False)

plt.title(f"Matriz de Confusión\n({', '.join(BIOMARCADORES_SELECCIONADOS[:3])})")
plt.xlabel("Predicción IA")
plt.ylabel("Realidad Clínica")

plt.tight_layout()
plt.show()


