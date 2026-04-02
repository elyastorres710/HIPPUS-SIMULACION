import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Carga de datos
df = pd.read_csv('../../data/processed/analisis_resultados.csv')
df.columns = df.columns.str.strip()

# Aplicación de ruido (20% para máxima robustez)
np.random.seed(42)
X_raw = df.drop(columns=['Diagnostico'])
noise = np.random.normal(0, 0.20, X_raw.shape)
X_noisy = X_raw + (X_raw * noise)
y = df['Diagnostico']

# Selección manual de la combinación clínica ganadora
mejor_combo = ['Desviacion', 'PUI']
X_final = X_noisy[mejor_combo]
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.20, random_state=42)

# Modelo final (Random Forest)
modelo_final = RandomForestClassifier(random_state=42)
modelo_final.fit(X_train, y_train)
y_pred = modelo_final.predict(X_test)

# Reporte en terminal
print(f"--- EVALUACIÓN FINAL ---")
print(f"Variables seleccionadas: {mejor_combo}")
print(classification_report(y_test, y_pred))

# Generación de Matriz de Confusión
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues', ax=ax)
plt.title(f"Matriz de Confusión: {mejor_combo} (20% Ruido)")
plt.show()

