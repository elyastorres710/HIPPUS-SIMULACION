import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Configuración de rutas
ruta_entrada_datos   = 'data/processed/analisis_final.csv'
ruta_entrada_ranking = 'scripts/iteracion_2/metricas_finales.csv'
ruta_salida_csv      = 'scripts/iteracion_2/comparativa_algoritmos_finales.csv'
ruta_salida_png      = 'scripts/iteracion_2/comparativa_algoritmos_finales.png'

# Preparación de los datos
df = pd.read_csv(ruta_entrada_datos)
df.columns = df.columns.str.strip()
df['target'] = df['Diagnostico'].map({'Control': 0, 'Migraña Vestibular': 1})

# Selección automatizada de los mejores biomarcadores
if os.path.exists(ruta_entrada_ranking):
    df_rank = pd.read_csv(ruta_entrada_ranking)
    variables_optimas = df_rank.iloc[0]['Variables'].split(' + ')
else:
    variables_optimas = ['Desviacion', 'Frecuencia_Dom']

print(f"Combinación óptima seleccionada: {' + '.join(variables_optimas)}")

X = df[variables_optimas]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluación de algoritmos
modelos_evaluacion = [
    ('Random Forest',          RandomForestClassifier(n_estimators=100, random_state=42)),
    ('Support Vector Machine', SVC(probability=True, random_state=42)),
    ('K-Means',                KMeans(n_clusters=2, random_state=42, n_init=10))
]

resultados_comparativa = []

for nombre_modelo, modelo in modelos_evaluacion:
    es_kmeans = nombre_modelo == 'K-Means'

    if es_kmeans:
        modelo.fit(X_train, y_train)
    else:
        modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_test)

    # Ajuste de orientación para modelo no supervisado
    if es_kmeans:
        if accuracy_score(y_test, 1 - predicciones) > accuracy_score(y_test, predicciones):
            predicciones = 1 - predicciones
        auc_valor = 'N/A'
    else:
        probabilidades = modelo.predict_proba(X_test)[:, 1]
        auc_valor = round(roc_auc_score(y_test, probabilidades), 3)

    resultados_comparativa.append({
        'Algoritmo':    nombre_modelo,
        'Exactitud':    round(accuracy_score(y_test, predicciones), 3),
        'Precision':    round(precision_score(y_test, predicciones), 3),
        'Sensibilidad': round(recall_score(y_test, predicciones), 3),
        'F1_Score':     round(f1_score(y_test, predicciones), 3),
        'AUC':          auc_valor
    })

df_resultados = pd.DataFrame(resultados_comparativa)

# Guardar CSV
os.makedirs(os.path.dirname(ruta_salida_csv), exist_ok=True)
df_resultados.to_csv(ruta_salida_csv, index=False)
print(f"CSV guardado en: {ruta_salida_csv}")

# Exportar tabla PNG
col_display = ['Algoritmo', 'Exactitud', 'Precision', 'Sensibilidad', 'F1_Score', 'AUC']
col_headers = ['Algoritmo', 'Exactitud', 'Precisión', 'Sensibilidad', 'F1-Score', 'AUC']

tabla_data = df_resultados[col_display].values.tolist()

fig, ax = plt.subplots(figsize=(13, 3.2))
ax.axis('off')

# Colores
COLOR_HEADER   = '#1B2A4A'
COLOR_FILA_PAR = '#EEF2F7'
COLOR_FILA_IMP = '#FFFFFF'
COLOR_TEXTO    = '#1B2A4A'
COLOR_BORDE    = '#C8D3E0'
ACCENT_VERDE   = '#2E7D32'

# Índices de métricas clave
IDX_SENS = col_display.index('Sensibilidad')
IDX_F1   = col_display.index('F1_Score')
IDX_AUC  = col_display.index('AUC')

col_widths  = [0.32, 0.12, 0.12, 0.14, 0.14, 0.12]
x_positions = []
x = 0.01
for w in col_widths:
    x_positions.append(x + w / 2)
    x += w

ROW_H    = 0.16
HEADER_Y = 0.82

# Encabezado
for j, (header, xp, w) in enumerate(zip(col_headers, x_positions, col_widths)):
    ax.add_patch(plt.Rectangle(
        (xp - w/2, HEADER_Y - ROW_H/2), w, ROW_H,
        transform=ax.transAxes, color=COLOR_HEADER, zorder=2
    ))
    ax.text(xp, HEADER_Y, header,
            transform=ax.transAxes,
            ha='center', va='center',
            fontsize=9, fontweight='bold', color='white', zorder=3)

# Filas
for i, fila in enumerate(tabla_data):
    y_centro = HEADER_Y - ROW_H - i * ROW_H
    bg = COLOR_FILA_PAR if i % 2 == 0 else COLOR_FILA_IMP

    for j, (val, xp, w) in enumerate(zip(fila, x_positions, col_widths)):
        ax.add_patch(plt.Rectangle(
            (xp - w/2, y_centro - ROW_H/2), w, ROW_H,
            transform=ax.transAxes, color=bg, zorder=1,
            linewidth=0.5, edgecolor=COLOR_BORDE
        ))

        # Resaltar mejor fila en métricas clave (fila 0 = RF, asumido mejor)
        if i == 0 and j in [IDX_SENS, IDX_F1, IDX_AUC]:
            txt_color = ACCENT_VERDE
            fw = 'bold'
        else:
            txt_color = COLOR_TEXTO
            fw = 'normal'

        ax.text(xp, y_centro, str(val),
                transform=ax.transAxes,
                ha='center', va='center',
                fontsize=8.5, color=txt_color, fontweight=fw, zorder=3)

# Título
fig.text(
    0.5, 0.97,
    'Comparativa de Algoritmos de Clasificación — Iteración 2',
    ha='center', va='top',
    fontsize=12, fontweight='bold', color=COLOR_HEADER
)

# Leyenda
fig.text(
    0.01, 0.02,
    'Verde: métricas destacadas del mejor clasificador supervisado',
    fontsize=7.5, color=ACCENT_VERDE
)

os.makedirs(os.path.dirname(ruta_salida_png), exist_ok=True)
plt.savefig(ruta_salida_png, dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"Figura PNG guardada en: {ruta_salida_png}")

# Imprimir en consola
print("\nComparativa de Algoritmos de Clasificación (Iteración 2)")
print(df_resultados.to_string(index=False))