import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# CONFIGURACION DE RUTAS
RUTA_ENTRADA = 'data/processed/analisis_resultados_final.csv'
DIR_VISUALIZACION = "docs/resultados_iteracion_1/05_visualizacion/"
os.makedirs(DIR_VISUALIZACION, exist_ok=True)

# CARGA DE DATOS
if not os.path.exists(RUTA_ENTRADA):
    print(f"Error: No se localizo el archivo en {RUTA_ENTRADA}")
    exit()

datos = pd.read_csv(RUTA_ENTRADA)
y_real = np.where(datos['Diagnostico'] == 'Migraña Vestibular', 1, 0)
biomarcadores = ['Amplitud', 'PUAL', 'Frecuencia_Dom']
X = datos[biomarcadores]

# DIVISION DE MUESTRA Y ANALISIS COMPUTACIONAL
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X, y_real, test_size=0.2, random_state=42, stratify=y_real
)

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_entrenamiento, y_entrenamiento)
y_prediccion = modelo.predict(X_prueba)

# CATEGORIZACION CLINICA SEGUN DESEMPEÑO
analisis_df = X_prueba.copy()
analisis_df['Real'] = y_prueba
analisis_df['Pred'] = y_prediccion

def asignar_categoria(fila):
    real, pred = fila['Real'], fila['Pred']
    if real == 1 and pred == 1: return 'Verdadero Positivo (VM Detectado)', '#e74c3c', 'o'
    if real == 0 and pred == 0: return 'Verdadero Negativo (Control Correcto)', '#2ecc71', 'o'
    if real == 0 and pred == 1: return 'Falso Positivo (Error: Sano como VM)', '#f1c40f', 'D'
    if real == 1 and pred == 0: return 'Falso Negativo (Error: VM no detectado)', '#9b59b6', 's'
    return 'Indefinido', 'gray', 'x'

categorias = analisis_df.apply(lambda r: pd.Series(asignar_categoria(r)), axis=1)
analisis_df[['Categoria', 'Color', 'Marcador']] = categorias

# REPRESENTACION GRAFICA 3D
plt.rcParams['toolbar'] = 'toolbar2'

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# CONFIGURACION DEL ANGULO DE VISION
ax.view_init(elev=15, azim=-120)

# Creacion de puntos por categoria (estilo suave)
for cat in analisis_df['Categoria'].unique():
    sub = analisis_df[analisis_df['Categoria'] == cat]
    ax.scatter(
        sub['Amplitud'], 
        sub['Frecuencia_Dom'], 
        sub['PUAL'],
        c=sub['Color'].iloc[0],
        marker=sub['Marcador'].iloc[0],
        label=cat,
        s=50,
        edgecolors='none',
        alpha=0.8
    )

# Rotulacion de ejes y formato
ax.set_title('Espacio Diagnostico 3D: Analisis de Clasificacion', pad=20)
ax.set_xlabel('Amplitud (mm)')
ax.set_ylabel('Frecuencia (Hz)')
ax.set_zlabel('PUAL (mm)')

# Activacion de rejilla de referencia
ax.grid(True)

# Ubicacion de la leyenda
ax.legend(title="Interpretacion Clinica", loc='center left', bbox_to_anchor=(1.1, 0.5))

# Exportacion de imagen 
ruta_imagen = os.path.join(DIR_VISUALIZACION, "espacio_3D_diagnostico_perspectiva.png")
plt.savefig(ruta_imagen, dpi=300, bbox_inches='tight')

print(f"Imagen guardada en: {ruta_imagen}")
print("Desplegando ventana interactiva...")

plt.show()
