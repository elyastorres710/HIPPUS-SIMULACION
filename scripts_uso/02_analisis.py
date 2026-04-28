import sys
import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, detrend

# Configuración de directorios para la localización de librerías de análisis
ruta_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ruta_raiz not in sys.path:
    sys.path.append(ruta_raiz)

# Importación de parámetros clínicos y biomarcadores pupilares
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

def aplicar_filtrado_banda_clinico(señal, frecuencia_muestreo):
    """
    Aplica un filtro Butterworth de 4to orden para aislar la banda del Hippus.
    Rango de interés: 0.05 Hz a 2.5 Hz para eliminar ruidos de baja frecuencia.
    """
    nyquist = 0.5 * frecuencia_muestreo
    limites = [0.05 / nyquist, 2.5 / nyquist]
    b, a = butter(4, limites, btype='band')
    return filtfilt(b, a, señal)

def procesar_estudio_pupilar():
    """
    Ejecuta el análisis masivo de registros para la caracterización de biomarcadores.
    Implementa segmentación temporal y estimación robusta de la amplitud oscilatoria.
    """
    # Rutas de archivos (Ajustar según el dataset de validación final)
    ruta_entrada = "data/raw/dataset_raw.csv" 
    ruta_salida = "data/processed/analisis_resultados.csv"
    
    fs = 60.0
    VENTANA_ANALISIS = 256 # Tamaño del segmento para análisis espectral

    if not os.path.exists(ruta_entrada):
        print(f"Error: No se localizó la base de datos en {ruta_entrada}")
        return

    # Carga de registros y etiquetas diagnósticas
    datos_pacientes = pd.read_csv(ruta_entrada)
    diagnosticos = datos_pacientes["Diagnostico"]
    registros_crudos = datos_pacientes.drop(columns=["Diagnostico"]).values

    analisis_clinico_total = []

    print("Iniciando caracterización de biomarcadores y estimación de amplitud...")

    for i in range(len(registros_crudos)):
        señal_original = registros_crudos[i]
        
        # 1. Tratamiento de la señal: Filtrado de banda
        señal_preprocesada = aplicar_filtrado_banda_clinico(señal_original, fs)
        
        # 2. Segmentación del registro para análisis local
        segmentos = [señal_preprocesada[j : j + VENTANA_ANALISIS] 
                     for j in range(0, len(señal_preprocesada), VENTANA_ANALISIS) 
                     if len(señal_preprocesada[j : j + VENTANA_ANALISIS]) == VENTANA_ANALISIS]
        
        metricas_segmentadas = []
        for ventana in segmentos:
            # Corrección de tendencia lineal para estabilizar la línea base del segmento
            ventana_estabilizada = detrend(ventana, type='linear')
            
            # Cálculo de la Amplitud Pico a Pico Estimada:
            # Se utiliza el factor de 2.828 sobre la desviación estándar para 
            # representar fielmente la oscilación pupilar clínica.
            amplitud_estimada = np.std(ventana_estabilizada) * 2.828
            
            metricas_segmentadas.append({
                "Media": calcular_media_pupilar(señal_original[0:VENTANA_ANALISIS]),
                "Amplitud": amplitud_estimada, 
                "Desviacion": calcular_desviacion_estandar(ventana_estabilizada),
                "RMS": calcular_rms(ventana_estabilizada),
                "PUI": calcular_pui(ventana_estabilizada),
                "PUAL": calcular_pual(ventana_estabilizada, fs),
                "PUAL_Ratio": calcular_pual_ratio(ventana_estabilizada, fs),
                "Dfi": calcular_dfi(ventana_estabilizada),
                "Velocidad_Media": calcular_velocidad_promedio(ventana_estabilizada, fs),
                "Frecuencia_Dom": calcular_frecuencia_dominante(ventana_estabilizada, fs)
            })
        
        # Consolidación de métricas por sujeto
        resultados_sujeto = pd.DataFrame(metricas_segmentadas)
        
        # Se obtiene el promedio de los biomarcadores, manteniendo la amplitud máxima registrada
        perfil_paciente = resultados_sujeto.mean().to_dict()
        perfil_paciente["Amplitud"] = resultados_sujeto["Amplitud"].max()
        perfil_paciente["Diagnostico"] = diagnosticos[i]
        
        analisis_clinico_total.append(perfil_paciente)

    # Exportación de la base de datos procesada
    base_procesada = pd.DataFrame(analisis_clinico_total)
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    base_procesada.to_csv(ruta_salida, index=False)

    print(f"Análisis finalizado. Base de datos clínica actualizada en: {ruta_salida}")

if __name__ == "__main__":
    procesar_estudio_pupilar()