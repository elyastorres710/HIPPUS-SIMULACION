"""
Prueba: Respuesta del reflejo de luz a pulsos de 250ms a diferentes intensidades.

Estímulo: Pulso de luz de 250ms a tres niveles de luminosidad:
    - 20% (baja)
    - 60% (media)
    - 100% (alta)

Objetivo: Visualizar la respuesta transitoria del diámetro pupilar al pulso de luz,
comparando las tres intensidades en un gráfico sincronizado.

Topología: Configuración por defecto del sistema J&B 2018.
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
PULSE_DURATION = 0.250  # 250 ms

PRE_PULSE_TIME = 10.0  # tiempo antes del pulso (para alcanzar estado estacionario)
POST_PULSE_TIME = 10.0  # tiempo después del pulso (para ver recuperación)

T_TOTAL = PRE_PULSE_TIME + PULSE_DURATION + POST_PULSE_TIME
N_STEPS = int(round(T_TOTAL / DT))

# Niveles de luminosidad (como fracción de 1.0)
LUMINANCE_LEVELS = [0.2, 0.6, 1.0]

# Para curva de amplitud vs intensidad
INTENSITY_RANGE = np.linspace(0.0, 1.0, 20)  # 20 intensidades de 0 a 100%

# ---------------------------------------------------------------------------
# Ejecutar simulaciones
# ---------------------------------------------------------------------------

results = {}

for L_level in LUMINANCE_LEVELS:
    # Crear sistema fresco para cada simulación
    system = JohanssonBalkeniusSystem("default")
    
    # Configurar inputs/outputs estándar usando DefaultJBS
    DefaultJBS.setup_standard_inputs(system)
    DefaultJBS.setup_standard_outputs(system)
    
    # Arrays para guardar resultados
    time_arr = np.empty(N_STEPS)
    pupil_l_arr = np.empty(N_STEPS)
    pupil_r_arr = np.empty(N_STEPS)
    source_arr = np.empty(N_STEPS)  # luminancia de fuente
    
    # Diámetro inicial (estado basal)
    d_left = DefaultJBS.pupil_diameter(0.0, 0.0)
    d_right = DefaultJBS.pupil_diameter(0.0, 0.0)
    
    print(f"Simulando pulso al {L_level*100:.0f}% de luminosidad...")
    
    for i in range(N_STEPS):
        t = system.t
        # Determinar luminancia de fuente (pulso rectangular)
        if PRE_PULSE_TIME <= t < PRE_PULSE_TIME + PULSE_DURATION:
            L = L_level
        else:
            L = 0.0
        
        # Closed-loop óptico
        alpha_left = DefaultJBS.optical_input(L, d_left)
        alpha_right = DefaultJBS.optical_input(L, d_right)
        system.set_input("retina_left", alpha_left)
        system.set_input("retina_right", alpha_right)
        
        # Corticales en 0 (reflejo autonómico puro)
        DefaultJBS.set_cortical_baseline(system, excitatory=0.0, emotional=0.0, novelty=0.0)
        
        # Paso de simulación
        system.step()
        
        # Muestrear (t=0 es inicio del pulso)
        time_arr[i] = t - PRE_PULSE_TIME
        source_arr[i] = L
        pupil_l_arr[i] = system.get_output("pupil_left")
        pupil_r_arr[i] = system.get_output("pupil_right")
        
        # Actualizar diámetros para próximo paso
        d_left = pupil_l_arr[i]
        d_right = pupil_r_arr[i]
    
    results[L_level] = {
        'time': time_arr,
        'pupil_left': pupil_l_arr,
        'pupil_right': pupil_r_arr,
        'source': source_arr
    }

print("Hecho.")

# ---------------------------------------------------------------------------
# Calcular amplitud de reflejo para múltiples intensidades
# ---------------------------------------------------------------------------

print("Calculando curva de amplitud vs intensidad...")

amplitudes = []
for L_level in INTENSITY_RANGE:
    # Crear sistema fresco
    system = JohanssonBalkeniusSystem("default")
    DefaultJBS.setup_standard_inputs(system)
    DefaultJBS.setup_standard_outputs(system)
    
    # Diámetro inicial
    d_left = DefaultJBS.pupil_diameter(0.0, 0.0)
    d_right = DefaultJBS.pupil_diameter(0.0, 0.0)
    
    # Simular
    d_basal = None
    d_min = None
    
    for i in range(N_STEPS):
        t = system.t
        if PRE_PULSE_TIME <= t < PRE_PULSE_TIME + PULSE_DURATION:
            L = L_level
        else:
            L = 0.0
        
        alpha_left = DefaultJBS.optical_input(L, d_left)
        alpha_right = DefaultJBS.optical_input(L, d_right)
        system.set_input("retina_left", alpha_left)
        system.set_input("retina_right", alpha_right)
        DefaultJBS.set_cortical_baseline(system)
        
        system.step()
        
        d_left = system.get_output("pupil_left")
        d_right = system.get_output("pupil_right")
        
        # Registrar diámetro basal (promedio de 100ms antes del pulso)
        if -0.1 <= (t - PRE_PULSE_TIME) < 0:
            if d_basal is None:
                d_basal = d_left
        
        # Registrar diámetro mínimo durante pulso
        if 0 <= (t - PRE_PULSE_TIME) <= PULSE_DURATION:
            if d_min is None or d_left < d_min:
                d_min = d_left
    
    # Amplitud = basal - mínimo (constricción)
    if d_basal is not None and d_min is not None:
        amplitude = d_basal - d_min
    else:
        amplitude = 0.0
    
    amplitudes.append(amplitude)

print("Hecho.")

# ---------------------------------------------------------------------------
# Gráfico
# ---------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# --- Panel 1: Respuesta temporal (zoom -1 a 4s) ---
colors = ['lightblue', 'orange', 'red']
labels = ['20% luminosidad', '60% luminosidad', '100% luminosidad']

for idx, L_level in enumerate(LUMINANCE_LEVELS):
    data = results[L_level]
    ax1.plot(data['time'], data['pupil_left'], 
            color=colors[idx], linewidth=2, label=labels[idx])

# Marcar inicio y fin del pulso
ax1.axvspan(0, PULSE_DURATION, alpha=0.2, color='yellow', label='Pulso de luz')
ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax1.axvline(PULSE_DURATION, color='gray', linestyle=':', alpha=0.5)

ax1.set_xlabel('Tiempo (s)', fontsize=12)
ax1.set_ylabel('Diámetro pupilar (mm)', fontsize=12)
ax1.set_title('Respuesta temporal del PLR', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-1, 4)  # Zoom en área de interés

# Grilla de tiempo cada 200ms
ax1.xaxis.set_major_locator(MultipleLocator(1.0))
ax1.xaxis.set_minor_locator(MultipleLocator(0.2))
ax1.grid(True, which='minor', alpha=0.1, linestyle=':')

# --- Panel 2: Amplitud vs intensidad ---
ax2.plot(INTENSITY_RANGE * 100, amplitudes, 'o-', color='darkblue', linewidth=2, markersize=6)
ax2.set_xlabel('Intensidad de luz (%)', fontsize=12)
ax2.set_ylabel('Amplitud de reflejo (mm)', fontsize=12)
ax2.set_title('Amplitud del PLR vs intensidad de luz', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 100)

plt.tight_layout()
plt.savefig('pulsed_light_response.png', dpi=150)
print("Figura guardada → pulsed_light_response.png")
plt.close()
