"""
Prueba: Respuesta pupilar a activación cortical (panic response).

Topología: Configuración por defecto del sistema J&B 2018.

Estímulo: Condiciones lumínicas estables (50% de luminancia) con escalones
de activación cortical en diferentes tiempos:
    - Panel 1: cortex_excitatory activado
    - Panel 2: cortex_emotional activado
    - Panel 3: cortex_novelty activado
    - Panel 4: Todos los corticales activados simultáneamente (panic)

Objetivo: Visualizar la respuesta del diámetro pupilar a diferentes patrones
de activación cortical, demostrando el efecto de dilatación por vía simpática.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os

# Agregar directorio padre al path para importar lib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.generadores.JohanssonBalkenius import JohanssonBalkeniusSystem, DefaultJBS

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

DT = 0.002  # 2 ms
STABILIZATION_TIME = 10.0  # tiempo de estabilización antes de escalones
STEP_DURATION = 5.0  # duración de cada escalón cortical
RECOVERY_TIME = 5.0  # tiempo de recuperación después del escalón

T_TOTAL = STABILIZATION_TIME + STEP_DURATION + RECOVERY_TIME
N_STEPS = int(round(T_TOTAL / DT))

# Luminancia constante (condiciones estables)
L_CONSTANT = 0.5  # 50% de luminancia

# Configuraciones de activación cortical
CORTICAL_CONFIGS = [
    {'name': 'Excitatory', 'excitatory': 1.0, 'emotional': 0.0, 'novelty': 0.0},
    {'name': 'Emotional', 'excitatory': 0.0, 'emotional': 1.0, 'novelty': 0.0},
    {'name': 'Novelty', 'excitatory': 0.0, 'emotional': 0.0, 'novelty': 1.0},
    {'name': 'Panic (todos)', 'excitatory': 1.0, 'emotional': 1.0, 'novelty': 1.0},
]

# ---------------------------------------------------------------------------
# Ejecutar simulaciones
# ---------------------------------------------------------------------------

results = {}

for config in CORTICAL_CONFIGS:
    # Crear sistema fresco para cada configuración
    system = JohanssonBalkeniusSystem("default")
    
    # Configurar inputs/outputs estándar
    DefaultJBS.setup_standard_inputs(system)
    DefaultJBS.setup_standard_outputs(system)
    
    # Arrays para guardar resultados
    time_arr = np.empty(N_STEPS)
    pupil_l_arr = np.empty(N_STEPS)
    pupil_r_arr = np.empty(N_STEPS)
    
    # Diámetro inicial
    d_left = DefaultJBS.pupil_diameter(0.0, 0.0)
    d_right = DefaultJBS.pupil_diameter(0.0, 0.0)
    
    print(f"Simulando activación cortical: {config['name']}...")
    
    for i in range(N_STEPS):
        t = system.t
        
        # Luminancia constante
        L = L_CONSTANT
        
        # Closed-loop óptico
        alpha_left = DefaultJBS.optical_input(L, d_left)
        alpha_right = DefaultJBS.optical_input(L, d_right)
        system.set_input("retina_left", alpha_left)
        system.set_input("retina_right", alpha_right)
        
        # Activación cortical según configuración
        if STABILIZATION_TIME <= t < STABILIZATION_TIME + STEP_DURATION:
            DefaultJBS.set_cortical_baseline(
                system,
                excitatory=config['excitatory'],
                emotional=config['emotional'],
                novelty=config['novelty']
            )
        else:
            DefaultJBS.set_cortical_baseline(
                system,
                excitatory=0.0,
                emotional=0.0,
                novelty=0.0
            )
        
        # Paso de simulación
        system.step()
        
        # Muestrear (t=0 es inicio del escalón cortical)
        time_arr[i] = t - STABILIZATION_TIME
        pupil_l_arr[i] = system.get_output("pupil_left")
        pupil_r_arr[i] = system.get_output("pupil_right")
        
        # Actualizar diámetros para próximo paso
        d_left = pupil_l_arr[i]
        d_right = pupil_r_arr[i]
    
    results[config['name']] = {
        'time': time_arr,
        'pupil_left': pupil_l_arr,
        'pupil_right': pupil_r_arr,
    }

print("Hecho.")

# ---------------------------------------------------------------------------
# Gráfico
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, config in enumerate(CORTICAL_CONFIGS):
    ax = axes[idx]
    data = results[config['name']]
    
    # Graficar ambos ojos (deberían superponerse por simetría)
    ax.plot(data['time'], data['pupil_left'], 
            color='indigo', linewidth=2, label='Ojo izquierdo')
    ax.plot(data['time'], data['pupil_right'], 
            color='darkorange', linewidth=2, linestyle=':', label='Ojo derecho')
    
    # Marcar escalón cortical
    ax.axvspan(0, STEP_DURATION, alpha=0.2, color='red', label='Activación cortical')
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(STEP_DURATION, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Tiempo (s)', fontsize=11)
    ax.set_ylabel('Diámetro pupilar (mm)', fontsize=11)
    ax.set_title(f'Activación: {config["name"]}', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, STEP_DURATION + 1)  # Zoom en área de interés
    
    # Grilla de tiempo cada 500ms
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.grid(True, which='minor', alpha=0.1, linestyle=':')

plt.suptitle(
    'Respuesta pupilar a activación cortical (Panic Response)\n'
    'Luminancia constante al 50% | Escalón cortical de 5s',
    fontsize=14, fontweight='bold', y=0.995
)

plt.tight_layout()
plt.savefig('panic_response.png', dpi=150)
print("Figura guardada → panic_response.png")
plt.close()
