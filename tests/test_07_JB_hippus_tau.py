"""
Prueba: Oscilaciones tipo "pupillary hippus" variando tau de las conexiones.

Topología bilateral (lateralizada L/R):
    Parasimpático: Retinas → PTA → EWpg → CG  → Esfínter (constricción)
    Simpático:     PVN     → IML → SCG         → Dilatador (dilatación)
    Loop predictor cerebellar: CB ↔ EWpg vía EWpg_pred (resta lateralizada).

Acople óptico (extensión de J&B 2018):
    El flujo lumínico que llega a la retina es proporcional al área de la
    apertura pupilar (∝ d²), cerrando el loop óptico-mecánico.

Objetivo: Investigar cómo el parámetro tau (retraso de transmisión) de TODAS
las conexiones afecta la dinámica del sistema y puede generar oscilaciones
tipo "pupillary hippus" en condiciones de luminancia constante con ruido.
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

# Luminancia constante (sin ruido)
L_CONSTANT = 0.5  # 50% de luminancia

# Valores de tau para TODAS las conexiones
# El valor por defecto es dt = 0.002 (2 ms)
# Valores más altos pueden generar oscilaciones en todo el sistema
TAU_VALUES = [0.002, 0.020, 0.20, 0.40]  # 2ms, 20ms, 200ms, 400ms

# ---------------------------------------------------------------------------
# Ejecutar simulaciones
# ---------------------------------------------------------------------------

results = {}

for tau in TAU_VALUES:
    # Crear sistema fresco para cada valor de tau
    system = JohanssonBalkeniusSystem("default")
    
    # Modificar tau de TODAS las conexiones
    system.set_all_tau(tau)
    
    # Configurar inputs/outputs estándar usando DefaultJBS
    DefaultJBS.setup_standard_inputs(system)
    DefaultJBS.setup_standard_outputs(system)
    
    # Arrays para guardar resultados
    time_arr = np.empty(N_STEPS)
    source_arr = np.empty(N_STEPS)   # luminancia de fuente
    retinal_l_arr = np.empty(N_STEPS)   # entrada efectiva a retina izquierda
    retinal_r_arr = np.empty(N_STEPS)   # entrada efectiva a retina derecha
    pupil_l_arr = np.empty(N_STEPS)
    pupil_r_arr = np.empty(N_STEPS)
    
    # Diámetros iniciales
    d_left = DefaultJBS.pupil_diameter(0.0, 0.0)
    d_right = DefaultJBS.pupil_diameter(0.0, 0.0)
    
    print(f"Simulando con tau de TODAS las conexiones = {tau:.3f}s...")
    
    for i in range(N_STEPS):
        t = system.t
        
        # Luminancia constante con ruido gaussiano
        L = np.clip(np.random.normal(L_CONSTANT, 0.1), 0.0, 1.0)
        
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
    
    results[tau] = {
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

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(len(TAU_VALUES) + 1, 1, hspace=0.4)

# --- Panel 1: Luminancia constante (compartido para todas las simulaciones) ---
ax0 = fig.add_subplot(gs[0])
# Usamos datos de la primera simulación (todas tienen la misma luminancia)
data_first = results[TAU_VALUES[0]]
ax0.plot(data_first['time'], data_first['source'], color='goldenrod', linewidth=2,
         label='Luminancia constante')
ax0.fill_between(data_first['time'], data_first['source'], alpha=0.15, color='goldenrod')
ax0.set_ylabel('Luminancia (u.a.)', fontsize=11)
ax0.set_title('Estímulo: Luminancia constante (sin ruido)', fontsize=12, fontweight='bold')
ax0.set_ylim(-0.05, 1.15)
ax0.legend(loc='upper right', fontsize=10)
ax0.grid(True, alpha=0.3)
ax0.set_xlim(0, T_TOTAL)

# --- Paneles 2-5: Diámetro pupilar para diferentes taus ---
colors = ['lightblue', 'orange', 'red', 'purple']
labels = [f'τ={TAU_VALUES[0]*1000:.0f}ms (muy bajo)', f'τ={TAU_VALUES[1]*1000:.0f}ms (bajo)', f'τ={TAU_VALUES[2]*1000:.0f}ms (normal)', f'τ={TAU_VALUES[3]*1000:.0f}ms (alto)']

for idx, tau in enumerate(TAU_VALUES):
    ax = fig.add_subplot(gs[idx + 1], sharex=ax0)
    data = results[tau]
    
    ax.plot(data['time'], data['pupil_left'], color='indigo', linewidth=1.5,
            label='Ojo izquierdo')
    ax.plot(data['time'], data['pupil_right'], color='darkorange', linewidth=1.5, linestyle=':',
            label='Ojo derecho')
    ax.axhline(3.0, color='gray', linestyle=':', alpha=0.6, label='≈ luz normal (3 mm)')
    ax.set_ylabel('Diámetro pupilar (mm)', fontsize=11)
    ax.set_title(f'Diámetro pupilar: τ={tau*1000:.0f}ms', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, T_TOTAL)
    
    # Solo el último panel tiene etiqueta en eje x
    if idx == len(TAU_VALUES) - 1:
        ax.set_xlabel('Tiempo (s)', fontsize=11)

plt.suptitle(
    'Johansson & Balkenius (2018) — Oscilaciones tipo "pupillary hippus"\n'
    'Variando tau de TODAS las conexiones | '
    f'Luminancia constante al {L_CONSTANT*100:.0f}% con ruido (σ=0.1) | Corticales en 0',
    fontsize=12, fontweight='bold', y=0.995
)

plt.tight_layout()
plt.savefig('pupil_hippus_tau.png', dpi=150)
print("Figura guardada → pupil_hippus_tau.png")
plt.close()
