import os
import sys
import numpy as np
import pandas as pd

# Cargar libreria local
sys.path.append(os.getcwd())
from lib.generadores.signals import generar_señal_pupilar

# Configuracion de la señal (30 segundos a 60Hz)
fs = 60.0
t = np.linspace(0, 30.0, int(fs * 30.0))
n_sujetos = 1000

def main():
    dataset = []
    np.random.seed(42)

    for i in range(n_sujetos):
        # Dividir grupos 50/50
        es_enfermo = i >= 500
        label = "Migraña Vestibular" if es_enfermo else "Control"
        
        # Crear señal limpia del hippus
        s = generar_señal_pupilar(t, es_patologico=es_enfermo)
        
        # Aplicar ruido variable por sujeto (entre 1% y 7%)
        std_ruido = np.random.uniform(0.01, 0.07)
        ruido = np.random.normal(0, std_ruido, len(s))
        final = s + ruido
        
        # Consolidar datos con la etiqueta de diagnostico
        fila = np.append(final, label)
        dataset.append(fila)

    # Definir columnas y crear dataframe
    cols = [f"t_{j}" for j in range(len(t))] + ["Diagnostico"]
    df = pd.DataFrame(dataset, columns=cols)

    # Guardar el archivo en la carpeta de datos crudos
    if not os.path.exists("data/raw"):
        os.makedirs("data/raw")
    
    ruta = "data/raw/dataset_iteracion_1.csv"
    df.to_csv(ruta, index=False)
    
    print(f"Archivo guardado en: {ruta}")

if __name__ == "__main__":
    main()