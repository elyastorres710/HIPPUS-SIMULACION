import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# CONFIGURACION DE RUTAS
PATH_RESULTADOS = 'data/processed/analisis_resultados.csv'
DIR_EVAL = "docs/resultados_iteracion_1/06_evaluacion/"
os.makedirs(DIR_EVAL, exist_ok=True)

# VALORES DE REFERENCIA BIBLIOGRAFICA (PAPERS)
S_PAPER_23, E_PAPER_23 = 93.00, 94.00
S_PAPER_26, E_PAPER_26 = 91.00, 99.00

# CARGA DE DATOS
if not os.path.exists(PATH_RESULTADOS):
    print(f"Error: No se encontro el archivo en {PATH_RESULTADOS}")
    exit()

df = pd.read_csv(PATH_RESULTADOS)
y_real = np.where(df['Diagnostico'] == 'Migraña Vestibular', 1, 0)

# CRITERIOS CLINICOS (GUFONI ET AL.)
df['Pred_2023'] = np.where((df['Amplitud'] >= 0.5) & (df['Frecuencia_Dom'].between(0.04, 2.0)), 1, 0)
df['Pred_2026'] = np.where(df['PUAL'] > 0.393, 1, 0)

# MODELO DE INTELIGENCIA ARTIFICIAL (IA)
biomarcadores = ['Amplitud', 'PUAL', 'Frecuencia_Dom', 'Dfi', 'PUAL_Ratio', 'RMS']
X = df[biomarcadores]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_real, test_size=0.2, random_state=42, stratify=y_real
)

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)
y_pred_ia = modelo.predict(X_test)

# CALCULOS DE DESEMPEÑO
def calcular_metricas(cm):
    tp, fn, fp, tn = cm[1,1], cm[1,0], cm[0,1], cm[0,0]
    return (tp / (tp + fn)) * 100, (tn / (tn + fp)) * 100

s23, e23 = calcular_metricas(confusion_matrix(y_real, df['Pred_2023']))
s26, e26 = calcular_metricas(confusion_matrix(y_real, df['Pred_2026']))
s_ia, e_ia = calcular_metricas(confusion_matrix(y_test, y_pred_ia))

# GENERACION DE GRAFICOS (TABLAS DE CONTINGENCIA)
def guardar_grafico(cm, titulo, nombre):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Sano', 'Paciente VM'], 
                yticklabels=['Sano', 'Paciente VM'], cbar=False)
    plt.title(f"Validacion: {titulo}")
    plt.ylabel('Estado Real')
    plt.xlabel('Diagnostico Sistema')
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_EVAL, nombre), dpi=300)
    plt.close()

guardar_grafico(confusion_matrix(y_real, df['Pred_2023']), "Gufoni 2023", "contingencia_2023.png")
guardar_grafico(confusion_matrix(y_real, df['Pred_2026']), "Gufoni 2026", "contingencia_2026.png")
guardar_grafico(confusion_matrix(y_test, y_pred_ia), "Propuesta IA", "contingencia_ia.png")

# REPORTE FINAL SIMPLIFICADO (FORMATO HUMANO)
reporte_path = os.path.join(DIR_EVAL, "reporte_final.txt")

with open(reporte_path, "w") as f:
    f.write("INFORME DE RESULTADOS: EVALUACION HIPPUS PUPILAR\n")
    f.write("----------------------------------------------\n\n")
    
    f.write("1. VALIDACION DE LA SIMULACION (VS PAPERS)\n")
    f.write(f"Comparacion con Gufoni 2023:\n")
    f.write(f"   - Sensibilidad: Obtenida {s23:.2f}% (Esperada {S_PAPER_23}%)\n")
    f.write(f"   - Especificidad: Obtenida {e23:.2f}% (Esperada {E_PAPER_23}%)\n\n")
    
    f.write(f"Comparacion con Gufoni 2026 (PUAL):\n")
    f.write(f"   - Sensibilidad: Obtenida {s26:.2f}% (Esperada {S_PAPER_26}%)\n")
    f.write(f"   - Especificidad: Obtenida {e26:.2f}% (Esperada {E_PAPER_26}%)\n\n")
    
    f.write("2. RENDIMIENTO DE LA IA (PROPUESTA)\n")
    f.write(f"Resultados sobre Muestra Ciega (n=200):\n")
    f.write(f"   - Sensibilidad: {s_ia:.2f}%\n")
    f.write(f"   - Especificidad: {e_ia:.2f}%\n\n")
    
    f.write("3. ANALISIS DE MEJORA CLINICA\n")
    f.write(f"   - Mejora neta vs Gufoni 2023: +{s_ia - s23:.2f}% Sens. / +{e_ia - e23:.2f}% Espec.\n")
    f.write(f"   - Mejora neta vs Gufoni 2026: +{s_ia - s26:.2f}% Sens. / {e_ia - e26:.2f}% Espec.\n\n")
    

print(f"Reporte y graficos actualizados en: {DIR_EVAL}")    