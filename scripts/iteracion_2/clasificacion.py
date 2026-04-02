import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# Carga de resultados
df = pd.read_csv('../../data/processed/analisis_resultados.csv')
df.columns = df.columns.str.strip()

# RUIDO
X_raw = df.drop(columns=['Diagnostico'])
noise = np.random.normal(0, 0.20, X_raw.shape) # Agrega un 5% de ruido aleatorio, asi le da mas variabilidad real de la pupila 
X_noisy = X_raw + (X_raw * noise)
y = df['Diagnostico']

X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.20, random_state=42)

# Definición de los 3 Modelos                      
modelo_rf = RandomForestClassifier(random_state=42)
modelo_svm = SVC(kernel='linear', random_state=42)
modelo_km = KMeans(n_clusters=2, random_state=42, n_init=10) # 2 grupos: Control y Migraña

# Algoritmo de combinaciones
medidas = ['Media', 'Desviacion', 'RMS', 'PUI', 'Dfi', 'Velocidad_Media', 'Frecuencia_Dom']
resultados = []

for r in range(1, len(medidas) + 1):
    for combo in itertools.combinations(medidas, r):
        sel = list(combo)
        
        # RF
        modelo_rf.fit(X_train[sel], y_train)
        acc_rf = modelo_rf.score(X_test[sel], y_test)
        
        # SVM
        modelo_svm.fit(X_train[sel], y_train)
        acc_svm = modelo_svm.score(X_test[sel], y_test)
        
        # K-MEANS (Se evalúa distinto por ser no supervisado)
        clusters = modelo_km.fit_predict(X_test[sel])
        # Convertimos clusters a etiquetas para comparar (simplificado)
        y_test_num = y_test.astype('category').cat.codes
        acc_km = max(accuracy_score(y_test_num, clusters), accuracy_score(y_test_num, 1-clusters))

        resultados.append({'combinacion': sel, 'RF': acc_rf, 'SVM': acc_svm, 'KMeans': acc_km})

# Ranking final
df_res = pd.DataFrame(resultados)
print("Ranking de Modelos y Combinaciones (con Ruido Realista):")
print(df_res.sort_values(by='RF', ascending=False).head(10))

# Guardar mejor combinacion
mejor_combo = df_res.sort_values(by='RF', ascending=False).iloc[0]['combinacion']
with open('mejor_combinacion.txt', 'w') as f:
    f.write(",".join(mejor_combo))