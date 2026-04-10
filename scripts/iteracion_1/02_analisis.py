import sys
import os
import pandas as pd
import numpy as np

# Inyección de ruta raíz para localización de módulos
sys.path.append(os.path.abspath("."))

from lib.analisis.estadisticas import (
    calcular_media_pupilar,
    calcular_desviacion_estandar,
    calcular_rms,
    calcular_pui,
    calcular_pual_fft,  #PARAMETRO NUEVO.
    calcular_dfi,
    calcular_velocidad_promedio,
    calcular_frecuencia_dominante
)

def ejecutar_procesamiento_datos():
    ruta_entrada = "data/raw/dataset_raw.csv"
    ruta_salida = "data/processed/analisis_resultados.csv"
    fs = 60.0  # Frecuencia de muestreo en Hz
    
    if not os.path.exists(ruta_entrada):
        print(f"Error: No se encontró el archivo en {ruta_entrada}")
        return

    # Carga de datos de señales
    df = pd.read_csv(ruta_entrada)
    diagnosticos = df["Diagnostico"]
    señales = df.drop(columns=["Diagnostico"]).values

    resultados = []

    # Extracción de biomarcadores por paciente
    for i in range(len(señales)):
        señal_actual = señales[i]
        
        metricas = {
            "Media": calcular_media_pupilar(señal_actual),
            "Desviacion": calcular_desviacion_estandar(señal_actual),
            "RMS": calcular_rms(señal_actual),
            "PUI": calcular_pui(señal_actual),
            "PUAL": calcular_pual_fft(señal_actual, fs),
            "Dfi": calcular_dfi(señal_actual),
            "Velocidad_Media": calcular_velocidad_promedio(señal_actual, fs),
            "Frecuencia_Dom": calcular_frecuencia_dominante(señal_actual, fs),
            "Diagnostico": diagnosticos[i]
        }
        resultados.append(metricas)

    # Exportación a formato CSV procesado
    df_final = pd.DataFrame(resultados)
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    df_final.to_csv(ruta_salida, index=False)
    
    print(f"Dataset procesado exitosamente en: {ruta_salida}")

if __name__ == "__main__":
    ejecutar_procesamiento_datos()