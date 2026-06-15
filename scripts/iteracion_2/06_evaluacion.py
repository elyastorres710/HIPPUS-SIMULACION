import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)

# Configuración de rutas
PATH_RESULTADOS = 'data/processed/analisis_final.csv'
PATH_RANKING    = 'scripts/iteracion_2/metricas_finales.csv'
PATH_SALIDA_PNG = 'scripts/iteracion_2/evaluacion_2biomarc.png'
PATH_SALIDA_CSV = 'scripts/iteracion_2/evaluacion_2biomarc.csv'

# Lectura de biomarcadores óptimos
df_rank = pd.read_csv(PATH_RANKING)
df_rank.columns = df_rank.columns.str.strip()
biomarcadores_seleccionados = ['Desviacion', 'Frecuencia_Dom']

data_clinica = pd.read_csv(PATH_RESULTADOS)
data_clinica.columns = data_clinica.columns.str.strip()
data_clinica['clase'] = data_clinica['Diagnostico'].map({'Control': 0, 'Migraña Vestibular': 1})

X = data_clinica[biomarcadores_seleccionados]
y = data_clinica['clase']

print(f"Biomarcadores seleccionados: {' + '.join(biomarcadores_seleccionados)}")
print(f"Total de sujetos: {len(y)}  |  Controles: {(y==0).sum()}  |  Patológicos: {(y==1).sum()}")

# Validación cruzada estratificada (k-fold, k=10)
N_FOLDS = 10
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

scoring = {
    'accuracy':  'accuracy',
    'precision': 'precision',
    'recall':    'recall',
    'f1':        'f1',
    'roc_auc':   'roc_auc'
}

modelo_cv     = RandomForestClassifier(n_estimators=100, random_state=42)
cv_resultados = cross_validate(modelo_cv, X, y, cv=kf, scoring=scoring, return_train_score=False)

metricas_cv = {
    'Exactitud':    cv_resultados['test_accuracy'],
    'Precision':    cv_resultados['test_precision'],
    'Sensibilidad': cv_resultados['test_recall'],
    'F1_Score':     cv_resultados['test_f1'],
    'AUC':          cv_resultados['test_roc_auc']
}

print(f"\nValidación Cruzada Estratificada ({N_FOLDS}-Fold)")
print(f"{'Métrica':<15} {'Media':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 50)
for nombre, valores in metricas_cv.items():
    print(f"{nombre:<15} {valores.mean()*100:>7.2f}% {valores.std()*100:>7.2f}% {valores.min()*100:>7.2f}% {valores.max()*100:>7.2f}%")

# Bootstrap con intervalos de confianza (n=1000)
N_BOOTSTRAP = 1000
ALPHA       = 0.95
rng         = np.random.default_rng(42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
modelo_final = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_final.fit(X_train, y_train)

predicciones   = modelo_final.predict(X_test)
probabilidades = modelo_final.predict_proba(X_test)[:, 1]

bootstrap_metricas = {'Exactitud': [], 'Precision': [], 'Sensibilidad': [], 'F1_Score': [], 'AUC': [], 'Especificidad': []}

n_test = len(y_test)
for _ in range(N_BOOTSTRAP):
    indices = rng.integers(0, n_test, size=n_test)
    y_b     = np.array(y_test)[indices]
    pred_b  = predicciones[indices]
    prob_b  = probabilidades[indices]

    if len(np.unique(y_b)) < 2:
        continue

    bootstrap_metricas['Exactitud'].append(accuracy_score(y_b, pred_b))
    bootstrap_metricas['Precision'].append(precision_score(y_b, pred_b, zero_division=0))
    bootstrap_metricas['Sensibilidad'].append(recall_score(y_b, pred_b, zero_division=0))
    bootstrap_metricas['F1_Score'].append(f1_score(y_b, pred_b, zero_division=0))
    bootstrap_metricas['AUC'].append(roc_auc_score(y_b, prob_b))
    cm_b = confusion_matrix(y_b, pred_b)
    if cm_b.shape == (2, 2):
        tn_b, fp_b, fn_b, tp_b = cm_b.ravel()
        bootstrap_metricas['Especificidad'].append(tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0)

alpha_tail = (1 - ALPHA) / 2
print(f"\nBootstrap IC {int(ALPHA*100)}% (n={N_BOOTSTRAP})")
print(f"{'Métrica':<15} {'Media':>8} {'IC Inf':>8} {'IC Sup':>8}")
print("-" * 42)

resumen_bootstrap = {}
for nombre, valores in bootstrap_metricas.items():
    arr    = np.array(valores)
    media  = arr.mean()
    ic_inf = np.percentile(arr, alpha_tail * 100)
    ic_sup = np.percentile(arr, (1 - alpha_tail) * 100)
    resumen_bootstrap[nombre] = (media, ic_inf, ic_sup)
    print(f"{nombre:<15} {media*100:>7.2f}% [{ic_inf*100:.2f}% – {ic_sup*100:.2f}%]")

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, probabilidades, pos_label=1)
auc_valor   = roc_auc_score(y_test, probabilidades)
pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv('scripts/iteracion_2/roc_comb2.csv', index=False)


# Matriz de confusión
etiquetas_diagnosticas = ['Control', 'Migraña Vestibular']
y_test_labels = y_test.map({0: 'Control', 1: 'Migraña Vestibular'})
pred_labels   = pd.Series(predicciones).map({0: 'Control', 1: 'Migraña Vestibular'})
matriz_conf   = confusion_matrix(y_test_labels, pred_labels, labels=etiquetas_diagnosticas)

vp = matriz_conf[1, 1]
fn = matriz_conf[1, 0]
fp = matriz_conf[0, 1]
vn = matriz_conf[0, 0]
sensibilidad_puntual  = vp / (vp + fn) * 100 if (vp + fn) > 0 else 0
especificidad_puntual = vn / (vn + fp) * 100 if (vn + fp) > 0 else 0

print(f"\nReporte de Validación Clínica (split 80/20)")
print(f"Sensibilidad:  {sensibilidad_puntual:.2f}%  (Referencia Gufoni: 93.3%)")
print(f"Especificidad: {especificidad_puntual:.2f}%  (Referencia Gufoni: 94.0%)")

# Figura compuesta
COLOR_HEADER = '#1B2A4A'
COLOR_ACCENT = '#2E7D32'
colores      = ['#4472C4', '#ED7D31', '#2E7D32', '#FFC000', '#7030A0']

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Panel superior izquierdo: Validación cruzada
ax1 = fig.add_subplot(gs[0, 0])
nombres_metricas = list(metricas_cv.keys())
medias  = [metricas_cv[m].mean() * 100 for m in nombres_metricas]
errores = [metricas_cv[m].std() * 100  for m in nombres_metricas]

barras = ax1.bar(nombres_metricas, medias, yerr=errores, capsize=5,
                 color=colores, edgecolor='white', linewidth=0.8, error_kw={'linewidth': 1.5})
ax1.set_ylim(70, 100)
ax1.set_ylabel('Valor (%)', fontsize=9, color=COLOR_HEADER)
ax1.set_title(f'Validación Cruzada ({N_FOLDS}-Fold)', fontsize=10, fontweight='bold', color=COLOR_HEADER)
ax1.tick_params(axis='x', labelsize=8)
ax1.tick_params(axis='y', labelsize=8)
ax1.spines[['top', 'right']].set_visible(False)
for barra, media in zip(barras, medias):
    ax1.text(barra.get_x() + barra.get_width()/2, media + 2,
             f'{media:.1f}%', ha='center', va='bottom', fontsize=7.5, color=COLOR_HEADER)

# Panel superior central: Bootstrap IC
ax2 = fig.add_subplot(gs[0, 1])
nombres_bs  = list(resumen_bootstrap.keys())
medias_bs   = [resumen_bootstrap[m][0] * 100 for m in nombres_bs]
ic_inf_bs   = [resumen_bootstrap[m][1] * 100 for m in nombres_bs]
ic_sup_bs   = [resumen_bootstrap[m][2] * 100 for m in nombres_bs]
errores_inf = [m - i for m, i in zip(medias_bs, ic_inf_bs)]
errores_sup = [s - m for m, s in zip(medias_bs, ic_sup_bs)]

ax2.bar(nombres_bs, medias_bs,
        yerr=[errores_inf, errores_sup], capsize=5,
        color=colores, edgecolor='white', linewidth=0.8, error_kw={'linewidth': 1.5})
ax2.set_ylim(70, 100)
ax2.set_ylabel('Valor (%)', fontsize=9, color=COLOR_HEADER)
ax2.set_title(f'Bootstrap IC {int(ALPHA*100)}% (n={N_BOOTSTRAP})', fontsize=10, fontweight='bold', color=COLOR_HEADER)
ax2.tick_params(axis='x', labelsize=8)
ax2.tick_params(axis='y', labelsize=8)
ax2.spines[['top', 'right']].set_visible(False)
for i, (nombre, media) in enumerate(zip(nombres_bs, medias_bs)):
    ic_i = ic_inf_bs[i]
    ic_s = ic_sup_bs[i]
    ax2.text(i, media + 2, f'{media:.1f}%', ha='center', va='bottom', fontsize=7.5, color=COLOR_HEADER)
    ax2.text(i, ic_i - 4, f'[{ic_i:.1f}–{ic_s:.1f}]', ha='center', va='top', fontsize=6, color='#666666')

# Panel superior derecho: Curva ROC
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(fpr, tpr, color='#4472C4', lw=2, label=f'AUC = {auc_valor:.3f}')
ax3.plot([0, 1], [0, 1], color='#AAAAAA', lw=1, linestyle='--', label='Clasificador aleatorio')
ax3.fill_between(fpr, tpr, alpha=0.08, color='#4472C4')
ax3.set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=8.5, color=COLOR_HEADER)
ax3.set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad)', fontsize=8.5, color=COLOR_HEADER)
ax3.set_title('Curva ROC', fontsize=10, fontweight='bold', color=COLOR_HEADER)
ax3.legend(fontsize=8.5, loc='lower right')
ax3.spines[['top', 'right']].set_visible(False)
ax3.tick_params(labelsize=8)

# Panel inferior izquierdo: Matriz de confusión
ax4 = fig.add_subplot(gs[1, 0])
disp = ConfusionMatrixDisplay(confusion_matrix=matriz_conf, display_labels=etiquetas_diagnosticas)
disp.plot(cmap='Blues', ax=ax4, values_format='g', colorbar=False)
ax4.set_title('Matriz de Confusión', fontsize=10, fontweight='bold', color=COLOR_HEADER)
ax4.set_xlabel('Diagnóstico Predictivo (IA)', fontsize=8.5, color=COLOR_HEADER)
ax4.set_ylabel('Diagnóstico Real', fontsize=8.5, color=COLOR_HEADER)
ax4.tick_params(labelsize=8)

# Panel inferior central: Tabla resumen comparativa con Gufoni
ax5 = fig.add_subplot(gs[1, 1:])
ax5.axis('off')

media_sens = resumen_bootstrap['Sensibilidad'][0] * 100
ic_inf_s   = resumen_bootstrap['Sensibilidad'][1] * 100
ic_sup_s   = resumen_bootstrap['Sensibilidad'][2] * 100

media_esp  = resumen_bootstrap['Especificidad'][0] * 100
ic_inf_e   = resumen_bootstrap['Especificidad'][1] * 100
ic_sup_e   = resumen_bootstrap['Especificidad'][2] * 100

tabla_resumen = [
    ['Métrica',       'Iteración 2 (Media ± IC 95%)',                        'Referencia Gufoni'],
    ['Sensibilidad',  f'{media_sens:.1f}% [{ic_inf_s:.1f}–{ic_sup_s:.1f}%]', '93.3%'],
    ['Especificidad', f'{media_esp:.1f}% [{ic_inf_e:.1f}–{ic_sup_e:.1f}%]',  '94.0%'],
    ['AUC',           f'{auc_valor:.3f}',                                      'N/R'],
    ['F1-Score',      f'{resumen_bootstrap["F1_Score"][0]*100:.1f}%',          'N/R'],
]

col_widths_tabla = [0.28, 0.42, 0.28]
col_x      = [0.02, 0.30, 0.72]
ROW_H      = 0.14
HEADER_Y_T = 0.88

for j, (header, xp, w) in enumerate(zip(tabla_resumen[0], col_x, col_widths_tabla)):
    ax5.add_patch(plt.Rectangle(
        (xp, HEADER_Y_T - ROW_H/2), w, ROW_H,
        transform=ax5.transAxes, color=COLOR_HEADER, zorder=2
    ))
    ax5.text(xp + w/2, HEADER_Y_T, header,
             transform=ax5.transAxes,
             ha='center', va='center',
             fontsize=8.5, fontweight='bold', color='white', zorder=3)

for i, fila in enumerate(tabla_resumen[1:]):
    y_c = HEADER_Y_T - ROW_H - i * ROW_H
    bg  = '#EEF2F7' if i % 2 == 0 else '#FFFFFF'
    for j, (val, xp, w) in enumerate(zip(fila, col_x, col_widths_tabla)):
        ax5.add_patch(plt.Rectangle(
            (xp, y_c - ROW_H/2), w, ROW_H,
            transform=ax5.transAxes, color=bg, zorder=1,
            linewidth=0.5, edgecolor='#C8D3E0'
        ))
        color_val = COLOR_ACCENT if j == 1 else COLOR_HEADER
        ax5.text(xp + w/2, y_c, val,
                 transform=ax5.transAxes,
                 ha='center', va='center',
                 fontsize=8, color=color_val, zorder=3)

ax5.set_title('Resumen Comparativo con Referencia Clínica', fontsize=10, fontweight='bold', color=COLOR_HEADER)

fig.suptitle(
    f'Evaluación del Modelo de Clasificación — Iteración 2\nBiomarcadores: {" + ".join(biomarcadores_seleccionados)}',
    fontsize=13, fontweight='bold', color=COLOR_HEADER, y=0.98
)

os.makedirs(os.path.dirname(PATH_SALIDA_PNG), exist_ok=True)
plt.savefig(PATH_SALIDA_PNG, dpi=180, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print(f"\nFigura guardada en: {PATH_SALIDA_PNG}")

# Guardar CSV resumen
filas_csv = []
for nombre in metricas_cv:
    filas_csv.append({
        'Metrica':      nombre,
        'CV_Media':     round(metricas_cv[nombre].mean(), 3),
        'CV_Std':       round(metricas_cv[nombre].std(), 3),
        'BS_Media':     round(resumen_bootstrap[nombre][0], 3),
        'BS_IC_inf_95': round(resumen_bootstrap[nombre][1], 3),
        'BS_IC_sup_95': round(resumen_bootstrap[nombre][2], 3)
    })

df_resumen = pd.DataFrame(filas_csv)
os.makedirs(os.path.dirname(PATH_SALIDA_CSV), exist_ok=True)
df_resumen.to_csv(PATH_SALIDA_CSV, index=False)
print(f"CSV resumen guardado en: {PATH_SALIDA_CSV}")