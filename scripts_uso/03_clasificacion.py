import pandas as pd
import numpy as np
import itertools
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score

# CONFIGURACION DE RUTAS
RUTA_ENTRADA = 'data/processed/analisis_resultados.csv'
RUTA_METRICAS = 'scripts/iteracion_1/metricas_completas.csv'
DIR_RANKING = 'docs/resultados_iteracion_1/03_ranking/' 
os.makedirs(DIR_RANKING, exist_ok=True)

# CARGA DE DATOS Y PREPARACION DE LA MUESTRA
if not os.path.exists(RUTA_ENTRADA):
    print(f"Error: No se encontro la base de datos procesada en {RUTA_ENTRADA}")
    exit()

base_datos = pd.read_csv(RUTA_ENTRADA)
base_datos.columns = base_datos.columns.str.strip()

# Codificacion de la variable patologica para analisis estadistico
base_datos['clase'] = base_datos['Diagnostico'].map({'Control': 0, 'Migraña Vestibular': 1})

# LISTADO DE PARAMETROS PUPILARES DISPONIBLES
parametros = ['Media', 'Desviacion', 'RMS', 'PUI', 'PUAL', 'PUAL_Ratio', 'Dfi', 'Velocidad_Media', 'Frecuencia_Dom']
X = base_datos[parametros]
y = base_datos['clase']

# Division de la muestra: 70% para entrenamiento y 30% para validacion
X_entrenamiento, X_validacion, y_entrenamiento, y_validacion = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# EVALUACION DE COMBINACIONES DE BIOMARCADORES
# Analisis de subconjuntos (1, 2 o 3 variables) para determinar la eficiencia diagnostica
jerarquizacion = []

for r in range(1, 4):
    for combinacion in itertools.combinations(parametros, r):
        lista_vars = list(combinacion)
        
        # Configuracion del modelo de clasificacion
        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X_entrenamiento[lista_vars], y_entrenamiento)
        
        predicciones = modelo.predict(X_validacion[lista_vars])
        probabilidades = modelo.predict_proba(X_validacion[lista_vars])[:, 1]
        
        # Registro de rendimiento clinico por combinacion
        jerarquizacion.append({
            'Protocolo': lista_vars,
            'Puntaje_F1': round(f1_score(y_validacion, predicciones), 3),
            'AUC_Curva': round(roc_auc_score(y_validacion, probabilidades), 3),
            'Exactitud': round(accuracy_score(y_validacion, predicciones), 3),
            'Precision': round(precision_score(y_validacion, predicciones, zero_division=0), 3),
            'Sensibilidad': round(recall_score(y_validacion, predicciones, zero_division=0), 3)
        })

# Ordenamiento de resultados segun el Puntaje F1 (balance entre sensibilidad y especificidad)
tabla_final = pd.DataFrame(jerarquizacion).sort_values(by='Puntaje_F1', ascending=False)

# GUARDADO DE RESULTADOS
tabla_final.to_csv(RUTA_METRICAS, index=False)
# Exportacion de los 10 protocolos de mayor eficacia
tabla_final.head(10).to_csv(os.path.join(DIR_RANKING, "mejores_protocolos.csv"), index=False)

# GENERACION DE EVIDENCIA GRAFICA
# Estimacion de la importancia relativa de cada parametro en el diagnostico final
modelo_global = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_global.fit(X_entrenamiento, y_entrenamiento)

importancias = pd.Series(modelo_global.feature_importances_, index=parametros).sort_values()

plt.figure(figsize=(10, 6))
importancias.plot(kind='barh', color='skyblue')
plt.title('Importancia de los Parametros Pupilares en el Diagnostico')
plt.xlabel('Peso Estadistico')
plt.ylabel('Biomarcador')
plt.tight_layout()
plt.savefig(os.path.join(DIR_RANKING, "importancia_biomarcadores.png"), dpi=300)
plt.close()

print("\n Jerarquizacion de protocolos finalizada.")
