import os
import numpy as np
import pandas as pd

# Configuracion
fs = 60.0
t = np.linspace(0, 30, int(fs * 30))
n_sujetos = 1000
np.random.seed(42)
dataset = []

 # Definir los grupos
for i in range(n_sujetos):
    es_enfermo = i >= 500
    diag = "Migraña Vestibular" if es_enfermo else "Control"
    
    if not es_enfermo:
        # Control: Algunos sanos tienen hippus leve (Falsos Positivos potenciales)
        amp = np.random.uniform(0.05, 0.15)
        # Frecuencia que a veces entra en el rango de Gufoni
        f = np.random.uniform(0.4, 0.7) 
        onda = amp * np.sin(2 * np.pi * f * t)
    else:
        # MV: Algunos enfermos tienen hippus muy debil (Falsos Negativos potenciales)
        amp = np.random.uniform(0.08, 0.30) 
        f = np.random.uniform(0.1, 0.5)
        onda = amp * np.sin(2 * np.pi * f * t)

    # RUIDO
    ruido_blanco = np.random.normal(0, np.random.uniform(0.05, 0.12), len(t))
    
    # Derivada Biológica
    deriva = np.cumsum(np.random.normal(0, 0.002, len(t))) 
    
    pupila = 4.0 + onda + ruido_blanco + deriva
    
    fila = np.append(pupila, diag)
    dataset.append(fila)

# Guardar
columnas = [f"t_{j}" for j in range(len(t))] + ["Diagnostico"]
df = pd.DataFrame(dataset, columns=columnas)
os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/dataset_raw.csv", index=False)
print("Generacion de datos realizada")