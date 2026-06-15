import os
import sys
import copy
import numpy as np
import pandas as pd

# Configuración del entorno y dependencias
sys.path.append(os.path.abspath("."))
from lib.generadores.JohanssonBalkenius import ConfigurableJBS, CUSTOM_CONFIG

# Parámetros generales de la simulación
T_TOTAL = 30.0         # Duración de la grabación en segundos
DT = 0.002             # Paso de tiempo (2ms)
FREQ_MUESTREO = 60.0  # Frecuencia final de los datos guardados
L_BASE = 0.5           # Nivel de luz base constante
N_SUJETOS = 1000       # Cantidad de casos a simular
SEED = 42

np.random.seed(SEED)


# ARQUITECTURA BASE DE LA RED 
config_base = copy.deepcopy(CUSTOM_CONFIG)

# Configuración de parámetros internos fijos para la Iteración 2
config_base['default_jitter_epsilon'] = 0.05
config_base['default_tau_rec'] = 0.5
config_base['default_U'] = 0.8
config_base['default_sigma'] = 0.6
config_base['default_tau_jitter'] = 0.2

# Estructura de perturbación externa de forma sináptica
config_base['boxes'].append({'name': 'disturbance', 'alpha': 0.0, 'beta': 0, 'gamma': 0})
config_base['connections'].append({'from': 'disturbance', 'to': 'EWpg_l', 'tipo': 'inhibitory'})
config_base['connections'].append({'from': 'disturbance', 'to': 'EWpg_r', 'tipo': 'inhibitory'})

# Guardar el valor por defecto de epsilon y tau del framework para las variaciones
EPSILON_BASE = config_base['default_epsilon']
TAU_BASE = config_base['default_tau']


def preparar_sistema_sujeto(L_background=0.5, T_stabilize=15.0, es_patologico=False):
    """
    Configura y estabiliza el modelo optimizando la inicialización 
    y reduciendo el tiempo de transitorio para acelerar la ejecución masiva.
    """
    # Se clona la estructura base ya armada para evitar sobrecarga de copias profundas pesadas
    config_sujeto = config_base.copy()

    # Aplicar la variación aleatoria poblacional (sujetos únicos)
    config_sujeto['default_epsilon'] = EPSILON_BASE * (1 + np.random.normal(0, 0.1))
    config_sujeto['default_tau']     = TAU_BASE * (1 + np.random.normal(0, 0.1))

    # Instanciar el simulador de forma directa
    system = ConfigurableJBS(config_sujeto)
    system.config('enable_bilateral_noise', True)
    system.config('enable_stochastic_noise', True)
    system.enable_history = False
    system.define_input("disturbance", ['disturbance'])

    # Estabilización optimizada (15 segundos bastan para remover el transitorio numérico)
    pasos_estabilizacion = int(round(T_stabilize / system.dt))
    for _ in range(pasos_estabilizacion):
        ruido_estab = np.random.normal(0, 0.15)
        system.step_simulation(L=L_background, inputs={'disturbance': ruido_estab})
    system.t = 0
    return system


# Bucle principal optimizado para generar el conjunto de datos
dataset = []
print(f"Iniciando simulación optimizada de la Iteración 2: {N_SUJETOS} sujetos.")

# Factor de remuestreo fijo calculado antes del bucle
downsample_factor = int((1.0 / FREQ_MUESTREO) / DT)
pasos_grabacion = int(T_TOTAL / DT)

for i in range(N_SUJETOS):
    es_patologico = i >= (N_SUJETOS // 2)
    
    if es_patologico:
        ruido_magnitud = np.random.uniform(0.5, 1.2) # Magnitud alta para Migraña      #Rangos finales
        diagnostico = "Migraña Vestibular"
    else:
        ruido_magnitud = np.random.uniform(0.1, 0.6) # Magnitud baja para Control      #Rangos finales
        diagnostico = "Control" 

    modelo = preparar_sistema_sujeto(L_background=L_BASE, T_stabilize=15.0, es_patologico=es_patologico)
    registros_pupila = []

    # Simulación principal de la prueba (30 segundos)
    for _ in range(pasos_grabacion):
        # Aquí se inyecta la inestabilidad en la simulación paso a paso
        valor_ruido_actual = np.random.normal(0, ruido_magnitud)
        modelo.step_simulation(L=L_BASE, inputs={'disturbance': valor_ruido_actual})
        registros_pupila.append(modelo.get_output("pupil_left"))

    # Aplicar remuestreo a 60Hz
    señal_final = registros_pupila[::downsample_factor]

    # Consolidar fila
    fila = np.append(señal_final, diagnostico)
    dataset.append(fila)

    if (i + 1) % 100 == 0:
        print(f"Progreso: {i + 1}/{N_SUJETOS} sujetos procesados de forma eficiente.")

# Guardado en formato CSV
columnas = [f"t_{j}" for j in range(len(dataset[0]) - 1)] + ["Diagnostico"]
df = pd.DataFrame(dataset, columns=columnas)

output_dir = "data/raw"
os.makedirs(output_dir, exist_ok=True)
df.to_csv(os.path.join(output_dir, "dataset_final.csv"), index=False)

print(f"\nProceso finalizado con éxito. Dataset guardado en: {os.path.join(output_dir, 'dataset_final.csv')}")


