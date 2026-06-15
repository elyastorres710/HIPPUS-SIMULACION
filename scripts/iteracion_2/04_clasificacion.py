import pandas as pd
import numpy as np
import itertools
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Configuración de rutas
ruta_entrada = "data/processed/analisis_final.csv"
ruta_salida_csv = "scripts/iteracion_2/metricas_finales.csv"
ruta_salida_png = "scripts/iteracion_2/ranking_finales.png"

# Preparación de los datos
datos = pd.read_csv(ruta_entrada)
datos.columns = datos.columns.str.strip()
datos['clase'] = datos['Diagnostico'].map({'Control': 0, 'Migraña Vestibular': 1})

variables_analisis = [
    'Media', 'Desviacion', 'RMS', 'PUI',
    'PUAL', 'PUAL_Ratio', 'Dfi', 'Velocidad_Media', 'Frecuencia_Dom'
]

X = datos[variables_analisis]
y = datos['clase']

# División de la muestra (Entrenamiento 80% / Prueba 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Evaluación de combinaciones de variables (hasta 5 biomarcadores)
ranking_combinaciones = []
total = sum(len(list(itertools.combinations(variables_analisis, r))) for r in range(1, 6))
procesadas = 0

print(f"Evaluando {total} combinaciones (1 a 5 variables)...")

for r in range(1, 6):
    for combinacion in itertools.combinations(variables_analisis, r):
        columnas = list(combinacion)

        modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo_rf.fit(X_train[columnas], y_train)

        predicciones   = modelo_rf.predict(X_test[columnas])
        probabilidades = modelo_rf.predict_proba(X_test[columnas])[:, 1]

        ranking_combinaciones.append({
            'Variables':     ' + '.join(columnas),
            'N_Variables':   len(columnas),
            'Exactitud':     round(accuracy_score(y_test, predicciones), 3),
            'Precision':     round(precision_score(y_test, predicciones), 3),
            'Sensibilidad':  round(recall_score(y_test, predicciones), 3),
            'F1_Score':      round(f1_score(y_test, predicciones), 3),
            'AUC':           round(roc_auc_score(y_test, probabilidades), 3)
        })

        procesadas += 1
        if procesadas % 50 == 0:
            print(f"  Progreso: {procesadas}/{total}")

# Ranking: ordenado por Sensibilidad,F1_Score, AUC, Precision, Exactitud
df_ranking = pd.DataFrame(ranking_combinaciones)

df_top10 = (
    df_ranking
    .sort_values(
        by=['Sensibilidad', 'F1_Score', 'AUC', 'Precision', 'Exactitud'],
        ascending=False
    )
    .head(10)
    .reset_index(drop=True)
)
df_top10.index += 1  # Ranking del 1 al 10

# Guardar CSV completo
os.makedirs(os.path.dirname(ruta_salida_csv), exist_ok=True)
df_ranking.sort_values(
    by=['Sensibilidad', 'F1_Score', 'AUC'],
    ascending=False
).to_csv(ruta_salida_csv, index=False)
print(f"\nCSV completo guardado en: {ruta_salida_csv}")

# Exportar tabla PNG
col_display = ['Variables', 'N_Variables', 'Sensibilidad', 'F1_Score', 'AUC', 'Precision', 'Exactitud']
col_headers = ['Combinación de Biomarcadores', 'N', 'Sensibilidad', 'F1-Score', 'AUC', 'Precisión', 'Exactitud']

tabla_data = df_top10[col_display].values.tolist()
n_filas = len(tabla_data)
n_cols  = len(col_headers)

fig, ax = plt.subplots(figsize=(16, 5.5))
ax.axis('off')

# Colores
COLOR_HEADER   = '#1B2A4A'
COLOR_FILA_PAR = '#EEF2F7'
COLOR_FILA_IMP = '#FFFFFF'
COLOR_TEXTO    = '#1B2A4A'
COLOR_BORDE    = '#C8D3E0'
ACCENT_VERDE   = '#2E7D32'
ACCENT_AMARILLO= '#F57F17'

# Métricas a resaltar (índices en col_display)
IDX_SENS = col_display.index('Sensibilidad')
IDX_F1   = col_display.index('F1_Score')
IDX_AUC  = col_display.index('AUC')

col_widths = [0.38, 0.05, 0.10, 0.10, 0.12, 0.12, 0.10]
x_positions = []
x = 0.01
for w in col_widths:
    x_positions.append(x + w / 2)
    x += w

ROW_H    = 0.082
HEADER_Y = 0.88

# Encabezado
for j, (header, xp, w) in enumerate(zip(col_headers, x_positions, col_widths)):
    ax.add_patch(plt.Rectangle(
        (xp - w/2, HEADER_Y - ROW_H/2), w, ROW_H,
        transform=ax.transAxes, color=COLOR_HEADER, zorder=2
    ))
    ax.text(xp, HEADER_Y, header,
            transform=ax.transAxes,
            ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='white', zorder=3)

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

        # Color especial para métricas clave en fila 1
        if i == 0 and j in [IDX_SENS, IDX_F1, IDX_AUC]:
            txt_color = ACCENT_VERDE
            fw = 'bold'
        elif j == 0:
            txt_color = COLOR_TEXTO
            fw = 'normal'
        else:
            txt_color = COLOR_TEXTO
            fw = 'normal'

        ax.text(xp, y_centro, str(val),
                transform=ax.transAxes,
                ha='center', va='center',
                fontsize=7.8, color=txt_color, fontweight=fw, zorder=3)

    # Número de ranking a la izquierda
    ax.text(0.002, y_centro, f"#{i+1}",
            transform=ax.transAxes,
            ha='left', va='center',
            fontsize=7.5, color='#888888', fontweight='bold')

# Título
fig.text(
    0.5, 0.97,
    'Ranking Top 10 — Combinaciones de Biomarcadores (Iteración 2)',
    ha='center', va='top',
    fontsize=12, fontweight='bold', color=COLOR_HEADER
)

# Leyenda
fig.text(
    0.01, 0.02,
    'Verde: métricas destacadas del mejor clasificador',
    fontsize=7.5, color=ACCENT_VERDE
)

os.makedirs(os.path.dirname(ruta_salida_png), exist_ok=True)
plt.savefig(ruta_salida_png, dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"Figura PNG guardada en: {ruta_salida_png}")

# Imprimir en consola
print("\nTOP 10 Combinaciones de Biomarcadores (Iteración 2)")
print(df_top10[col_display].to_string())