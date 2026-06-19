import os
import sys
import copy
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

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
N_PROCESOS = min(cpu_count(), 16)  # Usar hasta 16 procesos o todos los disponibles

np.random.seed(SEED)


# ARQUITECTURA BASE DE LA RED (Se ejecuta una sola vez fuera del bucle masivo)
config_base = copy.deepcopy(CUSTOM_CONFIG)

# Configuración de parámetros internos fijos para la Iteración 2
config_base['default_jitter_epsilon'] = 0.05
config_base['default_tau_rec'] = 0.5
config_base['default_U'] = 0.8
config_base['default_sigma'] = 0.6
config_base['default_tau_jitter'] = 0.2

# Estructurar la perturbación externa de forma sináptica
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
    # Copia profunda necesaria para aislar listas anidadas ('boxes', 'connections')
    # y evitar que ConfigurableJBS las mute y corrompa config_base en el mismo worker
    config_sujeto = copy.deepcopy(config_base)

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


def procesar_sujeto(args):
    """
    Función que procesa un solo sujeto de forma independiente.
    Diseñada para ejecutarse en paralelo usando multiprocessing.
    
    Args:
        args: Tupla (indice_sujeto, seed_local, T_TOTAL, DT, FREQ_MUESTREO, L_BASE, N_SUJETOS)
    
    Returns:
        fila: Array con la señal pupilar y diagnóstico
    """
    indice_sujeto, seed_local, T_TOTAL, DT, FREQ_MUESTREO, L_BASE, N_SUJETOS = args
    
    # Establecer semilla única para este proceso
    np.random.seed(seed_local)
    
    # Definición de grupos clínicos
    es_patologico = indice_sujeto >= (N_SUJETOS // 2)
    if es_patologico:
        ruido_magnitud = np.random.uniform(0.5, 1.2) # Magnitud alta para Migraña
        diagnostico = "Migraña Vestibular"
    else:
        ruido_magnitud = np.random.uniform(0.1, 0.6) # Magnitud baja para Control
        diagnostico = "Control"

    # Inicialización veloz del sujeto
    modelo = preparar_sistema_sujeto(L_background=L_BASE, T_stabilize=15.0, es_patologico=es_patologico)
    registros_pupila = []

    # Simulación principal de la prueba (30 segundos)
    pasos_grabacion = int(T_TOTAL / DT)
    for _ in range(pasos_grabacion):
        # Aquí se inyecta la inestabilidad en la simulación paso a paso
        valor_ruido_actual = np.random.normal(0, ruido_magnitud)
        modelo.step_simulation(L=L_BASE, inputs={'disturbance': valor_ruido_actual})
        registros_pupila.append(modelo.get_output("pupil_left"))

    # Aplicar remuestreo a 60Hz
    downsample_factor = int((1.0 / FREQ_MUESTREO) / DT)
    señal_final = registros_pupila[::downsample_factor]

    # Consolidar fila
    fila = np.append(señal_final, diagnostico)
    return fila


def main():
    """Función principal que coordina la ejecución paralela."""
    print(f"Iniciando simulación paralela de la Iteración 2: {N_SUJETOS} sujetos.")
    print(f"Usando {N_PROCESOS} procesos para paralelización.")
    
    # Preparar argumentos para cada sujeto
    # Cada sujeto recibe una semilla única para reproducibilidad
    args_list = []
    for i in range(N_SUJETOS):
        # Semilla única basada en la semilla global + índice
        seed_local = SEED + i * 1000
        args_list.append((i, seed_local, T_TOTAL, DT, FREQ_MUESTREO, L_BASE, N_SUJETOS))
    
    # Ejecutar en paralelo usando multiprocessing Pool
    print("Iniciando procesamiento paralelo...")
    with Pool(processes=N_PROCESOS) as pool:
        # Usar imap para mostrar progreso
        results = []
        for i, result in enumerate(pool.imap(procesar_sujeto, args_list)):
            results.append(result)
            if (i + 1) % 100 == 0:
                print(f"Progreso: {i + 1}/{N_SUJETOS} sujetos procesados.")
    
    # Combinar resultados
    dataset = results
    
    # Guardado en formato CSV
    columnas = [f"t_{j}" for j in range(len(dataset[0]) - 1)] + ["Diagnostico"]
    df = pd.DataFrame(dataset, columns=columnas)

    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "dataset_final.csv")
    df.to_csv(output_file, index=False)

    print(f"\nProceso finalizado con éxito. Dataset guardado en: {output_file}")
    print(f"Total de sujetos procesados: {len(dataset)}")


if __name__ == "__main__":
    # Necesario para multiprocessing en Windows/Linux
    main()
