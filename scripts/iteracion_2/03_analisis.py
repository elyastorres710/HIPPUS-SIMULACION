import sys
import os
import pandas as pd
import numpy as np

# Integración de la ruta principal para módulos
sys.path.append(os.path.abspath("."))

from lib.analisis.estadisticas import (
    calcular_media_pupilar,
    calcular_desviacion_estandar,
    calcular_rms,
    calcular_pui,
    calcular_pual,
    calcular_pual_ratio,
    calcular_dfi,
    calcular_velocidad_promedio,
    calcular_frecuencia_dominante
)

def ejecutar_procesamiento_datos():
    ruta_entrada = "data/raw/dataset_final.csv"
    ruta_salida = "data/processed/analisis_final.csv"
    frecuencia_muestreo = 60.0  # Hz
    
    if not os.path.exists(ruta_entrada):
        print(f"Error: No se localizó el archivo base en {ruta_entrada}")
        return

    # Carga de la base de datos
    df = pd.read_csv(ruta_entrada)
    diagnosticos = df["Diagnostico"]
    señales_pupilares = df.drop(columns=["Diagnostico"]).values

    resultados_clinicos = []

    # Extracción de parámetros por cada paciente simulado
    for i in range(len(señales_pupilares)):
        señal_actual = señales_pupilares[i]
        
        metricas = {
            "Media": calcular_media_pupilar(señal_actual),
            "Desviacion": calcular_desviacion_estandar(señal_actual),
            "RMS": calcular_rms(señal_actual),
            "PUI": calcular_pui(señal_actual),
            "PUAL": calcular_pual(señal_actual, frecuencia_muestreo),
            "PUAL_Ratio": calcular_pual_ratio(señal_actual, frecuencia_muestreo),
            "Dfi": calcular_dfi(señal_actual),
            "Velocidad_Media": calcular_velocidad_promedio(señal_actual, frecuencia_muestreo),
            "Frecuencia_Dom": calcular_frecuencia_dominante(señal_actual, frecuencia_muestreo),
            "Diagnostico": diagnosticos[i]
        }
        resultados_clinicos.append(metricas)

    # Exportación de las variables extraídas
    df_final = pd.DataFrame(resultados_clinicos)
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    df_final.to_csv(ruta_salida, index=False)
    
    print(f"Procesamiento de señales finalizado. Archivo guardado en: {ruta_salida}")

if __name__ == "__main__":
    ejecutar_procesamiento_datos() 