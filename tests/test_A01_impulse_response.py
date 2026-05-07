"""
Test A01: Respuesta al impulso con sistema entrenado.

Este test combina:
1. Entrenamiento prolongado del sistema (basado en test_A00_training.py)
2. Prueba de respuesta a pulsos de luz (basado en test_02_JB_pulsed_light.py)

El sistema primero se entrena durante 10 minutos con espectro completo de luminancia,
luego se prueba con pulsos de luz a diferentes intensidades para evaluar
cómo el aprendizaje afecta la respuesta pupilar.

Objetivo: Evaluar el efecto del aprendizaje del CB y AMY en la respuesta al reflejo de luz.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import copy

# Agregar directorio padre al path para importar lib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.generadores.JohanssonBalkenius import ConfigurableJBS

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

# Parámetros de entrenamiento (de test_A00_training.py)
DT = 0.002  # 2 ms
T_TRAINING = 600.0  # 10 minutos de entrenamiento
N_STEPS_TRAINING = int(round(T_TRAINING / DT))

# Parámetros de entrenamiento para espectro completo de luminancia
L_BASE = 0.5  # luminancia base
L_NOISE_STD = 0.1  # desviación estándar para explorar espectro completo
CHANGE_INTERVAL = 5.0  # intervalo para cambios de nivel base (segundos)
ASYMMETRY_STD = 0.05  # asimetría para entrenamiento robusto

# Parámetros de prueba de pulsos (de test_02_JB_pulsed_light.py)
PULSE_PREVIEW_TIME = 0.5  # 500 ms de visualización previa al pulso
PULSE_DURATION = 0.200  # 200 ms
PRE_PULSE_TIME = 10.0  # tiempo antes del pulso (para alcanzar estado estacionario)
POST_PULSE_TIME = 10.0  # tiempo después del pulso
PULSE_POSTVIEW_TIME = 5.0  # 5 segundos de visualización después del pulso

T_TOTAL = PRE_PULSE_TIME + PULSE_DURATION + POST_PULSE_TIME

# Niveles de luminosidad para pulsos (como fracción de 1.0)
LUMINANCE_LEVELS = [0.2, 0.6, 1.0]

# Para curva de amplitud vs intensidad
INTENSITY_RANGE = np.linspace(0.0, 1.0, 20)  # 20 intensidades de 0 a 100%

# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

def train_system(system, T_training):
    """
    Entrena el sistema con espectro completo de luminancia (de test_A00_training.py).
    
    Args:
        system: Sistema JBS a entrenar
        T_training: Tiempo de entrenamiento en segundos
        
    Returns:
        system: Sistema entrenado
    """
    n_steps = int(round(T_training / DT))
    
    # Variables para ruido diferencial
    L_base = L_BASE
    last_change_time = -CHANGE_INTERVAL
    
    print(f"Entrenando sistema por {T_training}s ({n_steps} pasos a dt={DT*1000:.1f}ms)...")
    pupil_left = system.get_output("pupil_left")
    pupil_right = system.get_output("pupil_right")

    for i in range(n_steps):
        t = system.t
        
        # Ruido diferencial lento
        if t - last_change_time >= CHANGE_INTERVAL:
            L_base = np.clip(L_base + np.random.normal(0, 0.2), 0.0, 1.0)
            last_change_time = t
        
        # Espectro completo de luminancia
        L_fast = np.clip(np.random.normal(L_base, L_NOISE_STD), 0.0, 1.0)
        
        # Asimetría en entrada luminica
        asymmetry = np.random.normal(0, ASYMMETRY_STD)
        L_left = np.clip(L_fast + asymmetry, 0.0, 1.0)
        L_right = np.clip(L_fast - asymmetry, 0.0, 1.0)
        
        # Aplicar entrada óptica usando el método del sistema
        alpha_left = system.light_input(L_left, pupil_left)
        alpha_right = system.light_input(L_right, pupil_right)
        system.set_input("retina_left", alpha_left)
        system.set_input("retina_right", alpha_right)
        
        # Entradas corticales
        avg_luminance = (L_left + L_right) / 2.0
        # cortex_excitatory = avg_luminance * 2.0
        # cortex_excitatory = np.clip(cortex_excitatory, 0.0, 2.0)
        # Normalización simple para entrada cortical
        cortex_excitatory = avg_luminance * 2.0 - 1.0  # Mapeo [0,1] → [-1,1]
        system.set_input("cortex_excitatory", cortex_excitatory)
        
        if avg_luminance > 0.5:
            system.set_input("cortex_emotional", 2)
        else:
            system.set_input("cortex_emotional", 0)


        # Paso de simulación
        system.step()
        
        pupil_left = system.get_output("pupil_left")
        pupil_right = system.get_output("pupil_right")
    
    print("Entrenamiento completado.")
    return system

def create_stabilized_system(system, L_background=0.0, T_stabilize=10.0):
    """
    Estabiliza el sistema con luminancia de fondo constante (de test_02_JB_pulsed_light.py).
    
    Args:
        system: Sistema JBS a estabilizar
        L_background: Luminancia de fondo
        T_stabilize: Tiempo de estabilización
        
    Returns:
        system: Sistema estabilizado
        time_arr: Vector de tiempos
        pupil_l_arr: Diámetro pupilar izquierdo
        pupil_r_arr: Diámetro pupilar derecho
    """
    n_steps = int(round(T_stabilize / DT))

    time_arr = np.empty(n_steps)
    pupil_l_arr = np.empty(n_steps)
    pupil_r_arr = np.empty(n_steps)
    
    # Establecer baseline cortical
    system.set_cortical_baseline()
    pupil_left = system.get_output("pupil_left")
    pupil_right = system.get_output("pupil_right")

    for i in range(n_steps):
        alpha_left = system.light_input(L_background, pupil_left)
        alpha_right = system.light_input(L_background, pupil_right)
        system.set_input("retina_left", alpha_left)
        system.set_input("retina_right", alpha_right)
        
        system.step()

        pupil_left = system.get_output("pupil_left")
        pupil_right = system.get_output("pupil_right")
        
        time_arr[i] = system.t
        pupil_l_arr[i] = pupil_left
        pupil_r_arr[i] = pupil_right

    return system, time_arr, pupil_l_arr, pupil_r_arr

def simulate_pulse_response(L_level, system):
    """
    Simula la respuesta pupilar a un pulso de luz (de test_02_JB_pulsed_light.py).
    
    Args:
        L_level: Intensidad del pulso (0.0 a 1.0)
        system: Sistema JBS estabilizado
        
    Returns:
        time_arr: Vector de tiempos
        pupil_l_arr: Diámetro pupilar izquierdo
        pupil_r_arr: Diámetro pupilar derecho
    """
    t_pulse_start = system.t
    n_steps = int(round((PULSE_DURATION + POST_PULSE_TIME) / DT))
    
    time_arr = np.empty(n_steps)
    pupil_l_arr = np.empty(n_steps)
    pupil_r_arr = np.empty(n_steps)

    pupil_left = system.get_output("pupil_left")
    pupil_right = system.get_output("pupil_right")
    
    # Establecer baseline cortical
    system.set_cortical_baseline()
    
    for i in range(n_steps):
        if t_pulse_start <= system.t < t_pulse_start + PULSE_DURATION:
            L = L_level
        else:
            L = 0.0
        
        alpha_left = system.light_input(L, pupil_left)
        alpha_right = system.light_input(L, pupil_right)
        system.set_input("retina_left", alpha_left)
        system.set_input("retina_right", alpha_right)
        
        system.step()

        pupil_left = system.get_output("pupil_left")
        pupil_right = system.get_output("pupil_right")
        
        time_arr[i] = system.t
        pupil_l_arr[i] = pupil_left
        pupil_r_arr[i] = pupil_right
    
    return time_arr, pupil_l_arr, pupil_r_arr

# ---------------------------------------------------------------------------
# Ejecución principal
# ---------------------------------------------------------------------------

def main():
    """Función principal del test."""
    print("=== Test A01: Respuesta al Impulso con Sistema Entrenado ===\n")
    
    # 1. Crear y entrenar sistema
    print("1. Creando y entrenando sistema...")
    system = ConfigurableJBS()
    
    # Mostrar estado inicial de cajas plásticas
    print("Estado inicial de cajas plásticas:")
    plastic_info_initial = system.get_plastic_boxes_info()
    for box_name, info in plastic_info_initial.items():
        print(f"  {box_name}: pesos={info['weights']}, salida={info['current_output']:.4f}")
    
    # Entrenar sistema
    system = train_system(system, T_TRAINING)
    
    # Mostrar estado después del entrenamiento
    print("\nEstado después del entrenamiento:")
    plastic_info_trained = system.get_plastic_boxes_info()
    for box_name, info in plastic_info_trained.items():
        print(f"  {box_name}: pesos={info['weights']}, salida={info['current_output']:.4f}")
    
    # 2. Estabilizar sistema para prueba de pulsos
    new_system = ConfigurableJBS()   # 2.1 crea nuevo sistema virgen con parámetros óptimos
    new_system.load_weights(plastic_info_trained) # 2.2 carga pesos de entrenamiento
    print("\n2. Estabilizando sistema para prueba de pulsos...") 
    stable_system, time_arr_base, pupil_l_arr_base, pupil_r_arr_base = create_stabilized_system(# 2.3 estabiliza con el sistema pre-enmtrenado (pesos cargados)
        new_system, L_background=0.0, T_stabilize=PRE_PULSE_TIME
    )
    
    # 3. Simular pulsos con diferentes intensidades
    print("\n3. Simulando pulsos con diferentes intensidades...")
    results = {}
    
    for L_level in INTENSITY_RANGE:
        print(f"Simulando pulso al {L_level*100:.0f}% de luminosidad...")
        # Copiar el sistema estabilizado
        system_copy = copy.deepcopy(stable_system)
        time_arr, pupil_l_arr, pupil_r_arr = simulate_pulse_response(L_level, system_copy)
        
        results[L_level] = {
            'time': np.concatenate([time_arr_base, time_arr]),
            'pupil_left': np.concatenate([pupil_l_arr_base, pupil_l_arr]),
            'pupil_right': np.concatenate([pupil_r_arr_base, pupil_r_arr]),
        }
    
    # 4. Extraer métricas (amplitud de reflejo)
    print("\n4. Analizando amplitudes de respuesta...")
    
    sim_time_start = PRE_PULSE_TIME - PULSE_PREVIEW_TIME
    sim_time_end = PRE_PULSE_TIME + PULSE_DURATION + PULSE_POSTVIEW_TIME
    base_time_start = PRE_PULSE_TIME - PULSE_PREVIEW_TIME
    base_time_end = PRE_PULSE_TIME
    
    amplitudes = []
    for L_level in INTENSITY_RANGE:
        data = results[L_level]
        # Mínimo: durante el pulso
        pulse_mask = (data['time'] >= sim_time_start) & (data['time'] <= sim_time_end)
        # Basal: antes del pulso
        base_mask = (data['time'] >= base_time_start) & (data['time'] <= base_time_end)
        
        pupil_signal = data['pupil_left'][pulse_mask]
        base_pupil = np.mean(data['pupil_left'][base_mask])
        
        min_pupil = np.min(pupil_signal)
        amplitude = base_pupil - min_pupil
        amplitudes.append(amplitude)
    
    # 5. Generar gráficos
    print("\n5. Generando gráficos...")
    
    # Obtener posiciones para niveles específicos
    idx_levels = [np.argmin(np.abs(INTENSITY_RANGE - level)) for level in LUMINANCE_LEVELS]
    idx_50 = np.argmin(np.abs(INTENSITY_RANGE - 0.5))
    
    time_50 = results[INTENSITY_RANGE[idx_50]]['time']
    pupil_50 = results[INTENSITY_RANGE[idx_50]]['pupil_left']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Panel 1: Respuesta temporal (zoom)
    ax1 = axes[0, 0]
    colors = ['lightblue', 'orange', 'red']
    labels = ['20% luminosidad', '60% luminosidad', '100% luminosidad']
    t_pulse_start = PRE_PULSE_TIME
    
    for idx, L_level in enumerate(LUMINANCE_LEVELS):
        idx_in_range = idx_levels[idx]
        data = results[INTENSITY_RANGE[idx_in_range]]
        time_relative = data['time'] - t_pulse_start
        ax1.plot(time_relative, data['pupil_left'],
                color=colors[idx], linewidth=2, label=labels[idx])
    
    ax1.axvspan(0, PULSE_DURATION, alpha=0.2, color='yellow', label='Pulso de luz')
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(PULSE_DURATION, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Tiempo (s)')
    ax1.set_ylabel('Diámetro pupilar (mm)')
    ax1.set_title('Respuesta Temporal del PLR (Sistema Entrenado)')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-PULSE_PREVIEW_TIME, PULSE_POSTVIEW_TIME)
    
    # Panel 2: Serie temporal completa al 50%
    ax2 = axes[0, 1]
    time_50_relative = time_50 - PRE_PULSE_TIME
    ax2.plot(time_50_relative, pupil_50, color='purple', linewidth=1.5)
    ax2.axvspan(0, PULSE_DURATION, alpha=0.2, color='yellow')
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(PULSE_DURATION, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Diámetro pupilar (mm)')
    ax2.set_title('Respuesta Completa al 50% Intensidad')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(min(time_50_relative), max(time_50_relative))
    
    # Panel 3: Amplitud vs intensidad
    ax3 = axes[1, 0]
    ax3.plot(INTENSITY_RANGE * 100, amplitudes, 'o-', color='darkblue', linewidth=2, markersize=6)
    ax3.set_xlabel('Intensidad de luz (%)')
    ax3.set_ylabel('Amplitud de reflejo (mm)')
    ax3.set_title('Amplitud del PLR vs Intensidad (Sistema Entrenado)')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 100)
    
    # Panel 4: Información del sistema
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    info_text = f"""Información del Sistema Entrenado:
    
Tiempo de entrenamiento: {T_TRAINING/60:.1f} minutos
Luminancia base: {L_BASE}
Ruido STD: {L_NOISE_STD}

Pesos aprendidos:
"""
    for box_name, info in plastic_info_trained.items():
        weights_str = ", ".join([f"{w:.4f}" for w in info['weights'].values()])
        info_text += f"  {box_name}: [{weights_str}]\n"
    
    info_text += f"""
Parámetros de prueba:
Duración pulso: {PULSE_DURATION*1000:.0f} ms
Tiempo estabilización: {PRE_PULSE_TIME:.1f} s
Rango intensidades: 0% - 100% [{system._config['L_MIN']:.2e} cd/m^2 - {system._config['L_MAX']:.2e} cd/m^2]
"""
    
    ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Guardar gráfico
    plot_file = "data/test/A01_impulse_response_trained.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Gráfico guardado en: {plot_file}")
    
    plt.show()

    print("\n=== Test A01 Completado ===")

if __name__ == "__main__":
    main()
