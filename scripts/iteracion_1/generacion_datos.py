import os
import sys
import numpy as np
import pandas as pd

# Configuracion de entorno
sys.path.append(os.getcwd())
from lib.generadores.signals import generar_señal_pupilar

# Parametros globales de adquisicion
FS = 60.0             # Sampling rate (Hz)
DURATION = 30.0       # Window length (s)
N_TOTAL = 1000        # Sample size
N_CONTROL = 500       # Control group size
TIME = np.linspace(0, DURATION, int(FS * DURATION))

def main():
    """
    Generacion de dataset sintetico para la Iteracion 1.
    Estructura: [Series temporales t_n | Etiqueta Diagnostica]
    """
    dataset = []

    for i in range(N_TOTAL):
        # Division balanceada de grupos
        es_patologico = i >= N_CONTROL
        label = "Migraña Vestibular" if es_patologico else "Control"
        
        # Simulacion de señal segun grupo
        signal = generar_señal_pupilar(TIME, es_patologico=es_patologico)
        
        # Consolidacion de registro
        row = np.append(signal, label)
        dataset.append(row)

    # Definicion de headers y persistencia
    cols = [f"t_{j}" for j in range(len(TIME))] + ["Diagnostico"]
    df = pd.DataFrame(dataset, columns=cols)

    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, "dataset_iteracion_1.csv")
    df.to_csv(file_path, index=False)
    
    print(f"Dataset persistido en: {file_path}")

if __name__ == "__main__":
    main()