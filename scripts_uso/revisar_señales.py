import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# CONFIGURACION DE RUTAS
# Se actualiza la ruta para apuntar al archivo validado de Gufoni
PATH_RAW = 'data/raw/dataset_final_validacion_gufoni.csv'
DIR_DOCS = "docs/resultados_iteracion_1/01_generacion/"
os.makedirs(DIR_DOCS, exist_ok=True)

# Verificacion de existencia del archivo de datos
if not os.path.exists(PATH_RAW):
    print(f"Error: No se localizo el archivo {PATH_RAW}.")
    print("Verificar el nombre del archivo en la carpeta data/raw/ antes de continuar.")
    exit()

# CARGA DE DATOS Y PARAMETROS TECNICOS
df = pd.read_csv(PATH_RAW)
frecuencia_muestreo = 60.0  
# El vector de tiempo se genera excluyendo la columna de diagnostico
tiempo = np.arange(df.shape[1] - 1) / frecuencia_muestreo  

# SELECCION ALEATORIA DE SUJETOS POR CATEGORIA
# Identificacion de indices para segmentacion diagnostica
indices_control = df[df['Diagnostico'] == 'Control'].index
indices_mv = df[df['Diagnostico'] == 'Migraña Vestibular'].index

# Seleccion al azar de un individuo por grupo para inspeccion
id_control = np.random.choice(indices_control)
id_mv = np.random.choice(indices_mv)

# Extraccion de series temporales (conversion a float para procesamiento)
ejemplo_control = df.iloc[id_control, :-1].values.astype(float)
ejemplo_mv = df.iloc[id_mv, :-1].values.astype(float)

# GENERACION DE EVIDENCIA GRAFICA
plt.figure(figsize=(12, 8))

# Visualizacion: Grupo Control
plt.subplot(2, 1, 1)
plt.plot(tiempo, ejemplo_control, color='#2ecc71', lw=1.5, label=f'Sujeto Control ID: {id_control}')
plt.title(f'Morfologia de Señal Pupilar: Grupo Control (ID: {id_control})', fontweight='bold')
plt.ylabel('Diametro (mm)')
plt.grid(True, alpha=0.3, linestyle='--')
# Ajuste de escala para mejor apreciacion de la señal
plt.ylim(np.min(ejemplo_control)-0.5, np.max(ejemplo_control)+0.5)
plt.legend(loc='upper right')

# Visualizacion: Grupo Migraña Vestibular
plt.subplot(2, 1, 2)
plt.plot(tiempo, ejemplo_mv, color='#e74c3c', lw=1.5, label=f'Sujeto MV ID: {id_mv}')
plt.title(f'Morfologia de Señal Pupilar: Grupo Migraña Vestibular (ID: {id_mv})', fontweight='bold')
plt.xlabel('Tiempo (segundos)')
plt.ylabel('Diametro (mm)')
plt.grid(True, alpha=0.3, linestyle='--')
plt.ylim(np.min(ejemplo_mv)-0.5, np.max(ejemplo_mv)+0.5)
plt.legend(loc='upper right')

plt.tight_layout()

# EXPORTACION Y DESPLIEGUE
ruta_guardado = os.path.join(DIR_DOCS, "verificacion_señales_aleatorias.png")
plt.savefig(ruta_guardado, dpi=300)

print(f"\n[OK] Grafico generado utilizando la base de datos: {os.path.basename(PATH_RAW)}")
print(f"[OK] Archivo guardado en: {ruta_guardado}")
print("Desplegando ventana de inspeccion visual interactiva...")
plt.show()
