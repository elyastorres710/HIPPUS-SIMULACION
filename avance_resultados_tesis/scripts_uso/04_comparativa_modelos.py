import pandas as pd
import numpy as np
import os
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# CARGA DE DATOS     
df = pd.read_csv('data/processed/analisis_resultados.csv')
df.columns = df.columns.str.strip()
df['target'] = df['Diagnostico'].map({'Control': 0, 'Migraña Vestibular': 1})

# SELECCIÓN DE VARIABLES (Top 1 del ranking)
if os.path.exists('scripts/iteracion_1/metricas_completas.csv'):
    df_rank = pd.read_csv('scripts/iteracion_1/metricas_completas.csv')
    vars_top = ast.literal_eval(df_rank.iloc[0]['Variables'])
else:
    vars_top = ['Desviacion', 'Frecuencia_Dom']

X = df[vars_top]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ENTRENAMIENTO DE MODELOS
lista_modelos = [
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('SVM', SVC(probability=True, random_state=42)),
    ('KMeans', KMeans(n_clusters=2, random_state=42, n_init=10))
]

res_final = []
for nombre, mod in lista_modelos:
    mod.fit(X_train, y_train)
    preds = mod.predict(X_test)
    if nombre == 'KMeans':
        if accuracy_score(y_test, 1 - preds) > accuracy_score(y_test, preds):
            preds = 1 - preds

    res_final.append({
        'Algoritmo': nombre,
        'Acc': round(accuracy_score(y_test, preds), 3),
        'Prec': round(precision_score(y_test, preds), 3),
        'Sens': round(recall_score(y_test, preds), 3),
        'F1': round(f1_score(y_test, preds), 3)
    })

df_res = pd.DataFrame(res_final)

# EXPORTACIÓN A DOCS 
output_dir = "docs/resultados_iteracion_1/04_comparativa_modelos/"
os.makedirs(output_dir, exist_ok=True)
df_res.to_csv(output_dir + "comparativa_algoritmos.csv", index=False)

# GRÁFICO COMPARATIVO (Acc, Prec, Sens, F1)
plt.figure(figsize=(12, 7))
# graficar las 4 métricas juntas
df_plot = df_res.melt(id_vars='Algoritmo', value_vars=['Acc', 'Prec', 'Sens', 'F1'], 
                      var_name='Metrica', value_name='Valor')

sns.barplot(data=df_plot, x='Algoritmo', y='Valor', hue='Metrica', palette='viridis')

plt.title('Comparativa Técnica de Algoritmos (Métricas Completas)')
plt.ylim(0, 1.1)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

plt.savefig(output_dir + "grafico_comparativa_modelos.png", dpi=300)
plt.close()

print(f"\n--- Tabla Comparativa ---")
print(df_res.to_string(index=False))
print(f"\n[OK] Tabla y gráfico con 4 métricas guardados en: {output_dir}")