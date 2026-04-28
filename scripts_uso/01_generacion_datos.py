import os
import numpy as np
import pandas as pd
import sys

# Inyectar ruta de librería
sys.path.append(os.path.abspath("."))
from lib.generadores.signals import generar_señal_pupilar

# Configuración
fs = 60.0
t = np.linspace(0, 30, int(fs * 30))
n_sujetos = 1000
np.random.seed(42) # Para que tu tesis sea replicable
dataset = []

print("Generando señales")

# Definir los grupos
for i in range(n_sujetos):
    es_patologico = i >= 500
    diag = "Migraña Vestibular" if es_patologico else "Control"
    
    # La función ahora aplica el filtro y el jitter internamente
    pupila = generar_señal_pupilar(es_patologico, t, fs)
    
    fila = np.append(pupila, diag)
    dataset.append(fila)
    
# Exportar Dataset
columnas = [f"t_{j}" for j in range(len(t))] + ["Diagnostico"]
df = pd.DataFrame(dataset, columns=columnas)

os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/dataset_raw.csv", index=False)

# Exportar Resumen para Documentación
output_dir = "docs/resultados_iteracion_1/01_generacion/"
os.makedirs(output_dir, exist_ok=True)

with open(output_dir + "resumen_generacion.txt", "w") as f:
    f.write("RESUMEN DE DATOS GENERADOS (MEJORADOS)\n")
    f.write("="*35 + "\n")
    f.write(f"Sujetos: {n_sujetos}\n")
    f.write(f"Frecuencia: {fs} Hz\n")
    f.write(f"Duracion: {t[-1]} segundos\n")

print(f"Dataset guardado en data/raw/dataset_raw.csv")
print(f"Resumen guardado en {output_dir}")

