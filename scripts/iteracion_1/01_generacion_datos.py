import os
import numpy as np
import pandas as pd
import sys

# signals
sys.path.append(os.path.abspath("."))
from lib.generadores.signals import generar_señal_pupilar

# Configuracion
fs = 60.0
t = np.linspace(0, 30, int(fs * 30))
n_sujetos = 1000
np.random.seed(42)
dataset = []

# Definir los grupos
for i in range(n_sujetos):
    es_patologico = i >= 500
    diag = "Migraña Vestibular" if es_patologico else "Control"
    
    # Creacion de la señal biologica + ruido + artefactos
    pupila = generar_señal_pupilar(es_patologico, t, fs)
    fila = np.append(pupila, diag)
    dataset.append(fila)
    
# Exportar
columnas = [f"t_{j}" for j in range(len(t))] + ["Diagnostico"]
df = pd.DataFrame(dataset, columns=columnas)

os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/dataset_raw.csv", index=False)
