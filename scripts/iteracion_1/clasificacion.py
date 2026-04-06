import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Cargar los datos y limpiar nombres
datos = pd.read_csv('data/processed/analisis_resultados.csv')
datos.columns = datos.columns.str.strip()

# Pasar el diagnostico a numeros para que las metricas funcionen
datos['clase'] = datos['Diagnostico'].map({'Control': 0, 'Migraña Vestibular': 1})

# Configurar variables y dividir el dataset
variables = ['Media', 'Desviacion', 'RMS', 'PUI', 'Dfi', 'Velocidad_Media', 'Frecuencia_Dom']
X = datos[variables]
y = datos['clase']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

ranking = []

# Probar combinaciones de 1 a 3 variables
for r in range(1, 4):
    for combo in itertools.combinations(variables, r):
        cols = list(combo)
        
        # Entrenamiento del modelo Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train[cols], y_train)
        
        # Obtener predicciones y probabilidades
        pred = rf.predict(X_test[cols])
        prob = rf.predict_proba(X_test[cols])[:, 1]
        
        # Calculo de metricas de desempeño
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred)
        rec = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        auc = roc_auc_score(y_test, prob)
        
        ranking.append({
            'Variables': cols,
            'Acc': round(acc, 3),
            'Prec': round(prec, 3),
            'Rec': round(rec, 3),
            'F1': round(f1, 3),
            'AUC': round(auc, 3)
        })

# Ordenar por F1 que es la mas equilibrada y mostrar el top 10
df_final = pd.DataFrame(ranking).sort_values(by='F1', ascending=False)

print("\n--- Resultados Detallados (Ordenados por F1) ---")
print(df_final.head(10).to_string(index=False))

# Guardar el archivo para el informe final
df_final.to_csv('scripts/iteracion_1/metricas_completas.csv', index=False)
print("\nTabla completa guardada en: scripts/iteracion_1/metricas_completas.csv")