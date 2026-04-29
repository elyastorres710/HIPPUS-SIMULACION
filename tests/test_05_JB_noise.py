"""
Prueba: Reflejo de luz pupilar con luminancia estacionaria fija
y ruido en 3 niveles diferentes (bajo, medio, alto).

Topología bilateral (lateralizada L/R):
    Parasimpático: Retinas → PTA → EWpg → CG  → Esfínter (constricción)
    Simpático:     PVN     → IML → SCG         → Dilatador (dilatación)
    Loop predictor cerebellar: CB ↔ EWpg vía EWpg_pred (resta lateralizada).

Acople óptico (extensión de J&B 2018):
    El flujo lumínico que llega a la retina es proporcional al área de la
    apertura pupilar (∝ d²), cerrando el loop óptico-mecánico.

Objetivo: Cuantificar visualmente la respuesta del sistema al ruido
en la señal de luminancia (luminancia base fija al 50%).
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# Agregar directorio padre al path para importar lib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.generadores.JohanssonBalkenius import JohanssonBalkeniusSystem, DefaultJBS

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

DT = 0.002  # 2 ms
T_TOTAL = 30.0  # 30 segundos de simulación
N_STEPS = int(round(T_TOTAL / DT))

# Límites del diámetro pupilar (usados en graficado)
D_MIN, D_MAX = 2.0, 8.0

# Luminancia base fija
L_BASE = 0.5  # 50% de luminancia

# Niveles de ruido (desviación estándar)
NOISE_LEVELS = [0.01, 0.05, 0.10]  # bajo, medio, alto

def get_noisy_luminance(t: float, base_level: float, noise_std: float) -> float:
    """
    Luminancia de la fuente con ruido Gaussiano.
    
    Args:
        t (float): Tiempo en segundos.
        base_level (float): Nivel base de luminancia.
        noise_std (float): Desviación estándar del ruido.
    
    Returns:
        float: Luminancia con ruido (u.a.).
    """
    noise = np.random.normal(0, noise_std)
    return np.clip(base_level + noise, 0.0, 1.0)

# ---------------------------------------------------------------------------
# Ejecutar simulaciones
# ---------------------------------------------------------------------------

results = {}

for noise_std in NOISE_LEVELS:
    # Crear sistema fresco para cada nivel de ruido
    system = JohanssonBalkeniusSystem("default")
    
    # Configurar inputs/outputs estándar usando DefaultJBS
    DefaultJBS.setup_standard_inputs(system)
    DefaultJBS.setup_standard_outputs(system)
    
    # Arrays para guardar resultados
    time_arr = np.empty(N_STEPS)
    source_arr = np.empty(N_STEPS)   # luminancia de fuente con ruido
    retinal_l_arr = np.empty(N_STEPS)   # entrada efectiva a retina izquierda
    retinal_r_arr = np.empty(N_STEPS)   # entrada efectiva a retina derecha
    pupil_l_arr = np.empty(N_STEPS)
    pupil_r_arr = np.empty(N_STEPS)
    
    # Diámetros iniciales
    d_left = DefaultJBS.pupil_diameter(0.0, 0.0)
    d_right = DefaultJBS.pupil_diameter(0.0, 0.0)
    
    # Semilla para reproducibilidad del ruido
    np.random.seed(42)
    
    print(f"Simulando luminancia base {L_BASE*100:.0f}% con ruido σ={noise_std:.2f}...")
    
    for i in range(N_STEPS):
        t = system.t
        
        # Luminancia con ruido
        L = get_noisy_luminance(t, L_BASE, noise_std)
        
        # Closed-loop óptico
        alpha_left = DefaultJBS.optical_input(L, d_left)
        alpha_right = DefaultJBS.optical_input(L, d_right)
        system.set_input("retina_left", alpha_left)
        system.set_input("retina_right", alpha_right)
        
        # Corticales en 0 (reflejo autonómico puro)
        DefaultJBS.set_cortical_baseline(system)
        
        # Paso de simulación
        system.step()
        
        # Muestrear
        time_arr[i] = t
        source_arr[i] = L
        retinal_l_arr[i] = alpha_left
        retinal_r_arr[i] = alpha_right
        pupil_l_arr[i] = system.get_output("pupil_left")
        pupil_r_arr[i] = system.get_output("pupil_right")
        
        # Actualizar diámetros para próximo paso
        d_left = pupil_l_arr[i]
        d_right = pupil_r_arr[i]
    
    results[noise_std] = {
        'time': time_arr,
        'source': source_arr,
        'retinal_left': retinal_l_arr,
        'retinal_right': retinal_r_arr,
        'pupil_left': pupil_l_arr,
        'pupil_right': pupil_r_arr,
    }

print("Hecho.")

# ---------------------------------------------------------------------------
# Gráfico
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.flatten()

colors = ['lightgreen', 'orange', 'red']
labels = ['Bajo (σ=0.01)', 'Medio (σ=0.05)', 'Alto (σ=0.10)']

for idx, noise_std in enumerate(NOISE_LEVELS):
    data = results[noise_std]
    
    # --- Panel izquierdo: Luminancia de fuente vs entrada retinal ---
    ax = axes[2 * idx]
    ax.plot(data['time'], data['source'], color='goldenrod', linewidth=0.5, alpha=0.7,
            label='Luminancia con ruido')
    ax.axhline(L_BASE, color='darkgoldenrod', linestyle='--', alpha=0.5, 
               label=f'Nivel base ({L_BASE:.2f})')
    ax.plot(data['time'], data['retinal_left'], color='darkred', linewidth=1.0,
            label='Entrada retinal (ojo izq.)')
    ax.plot(data['time'], data['retinal_right'], color='maroon', linewidth=1.0, linestyle=':',
            label='Entrada retinal (ojo der.)')
    ax.set_ylabel('Luminancia (u.a.)', fontsize=10)
    ax.set_title(f'Ruido: {labels[idx]}', fontsize=11, fontweight='bold')
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # --- Panel derecho: Diámetro pupilar ---
    ax = axes[2 * idx + 1]
    ax.plot(data['time'], data['pupil_left'], color='indigo', linewidth=1.0,
            label='Ojo izquierdo')
    ax.plot(data['time'], data['pupil_right'], color='darkorange', linewidth=1.0, linestyle=':',
            label='Ojo derecho')
    ax.axhline(3.0, color='gray', linestyle=':', alpha=0.6, label='≈ luz normal (3 mm)')
    ax.set_ylabel('Diámetro pupilar (mm)', fontsize=10)
    ax.set_title(f'Diámetro pupilar: {labels[idx]}', fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Límites compartidos del eje x
    if idx == 2:  # última fila
        ax.set_xlabel('Tiempo (s)', fontsize=11)
    ax.set_xlim(0, T_TOTAL)

plt.suptitle(
    'Johansson & Balkenius (2018) — Respuesta al ruido en luminancia estacionaria\n'
    f'Luminancia base fija al {L_BASE*100:.0f}% | Corticales en 0 (reflejo autonómico puro)',
    fontsize=12, fontweight='bold', y=0.995
)

plt.tight_layout()
plt.savefig('pupil_noise_response.png', dpi=150)
print("Figura guardada → pupil_noise_response.png")
plt.close()
