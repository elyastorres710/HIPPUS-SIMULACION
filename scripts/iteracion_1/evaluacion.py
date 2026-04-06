import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Configuración de archivos y parámetros
PATH_RESULTADOS = 'data/processed/analisis_resultados.csv'
BIOMARCADORES_SELECCIONADOS = ['Desviacion', 'PUI']

# Carga de datos procesados (con ruido integrado en origen)
data_hippus = pd.read_csv(PATH_RESULTADOS)
data_hippus.columns = data_hippus.columns.str.strip()

# Preparación de conjuntos de entrenamiento y test
caracteristicas = data_hippus[BIOMARCADORES_SELECCIONADOS]
diagnostico = data_hippus['Diagnostico']

X_train, X_test, y_train, y_test = train_test_split(
    caracteristicas, 
    diagnostico, 
    test_size=0.20, 
    random_state=42
)

# Entrenamiento del clasificador Random Forest
# Se utiliza el estado 42 para asegurar reproducibilidad en la tesis
modelo_mv = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_mv.fit(X_train, y_train)

# Predicción y validación del modelo
predicciones = modelo_mv.predict(X_test)

# Salida de métricas en consola
print(">>> RESULTADOS DE EVALUACIÓN CLÍNICA (Sujetos a ruido de captura)")
print(f"Variables analizadas: {BIOMARCADORES_SELECCIONADOS}")
print(classification_report(y_test, predicciones))

# Visualización de la Matriz de Confusión para el reporte final
fig, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, 
    predicciones, 
    cmap='Blues', 
    ax=ax,
    colorbar=False
)

plt.title(f"Clasificación MV vs Control\nBiomarcadores: {', '.join(BIOMARCADORES_SELECCIONADOS)}")
plt.tight_layout()
plt.show()

