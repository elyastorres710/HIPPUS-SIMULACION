"""
Prueba: Grid 4x4 de oscilaciones tipo "pupillary hippus" variando
epsilon y tau simultáneamente.

Topología bilateral (lateralizada L/R):
    Parasimpático: Retinas → PTA → EWpg → CG  → Esfínter (constricción)
    Simpático:     PVN     → IML → SCG         → Dilatador (dilatación)
    Loop predictor cerebellar: CB ↔ EWpg vía EWpg_pred (resta lateralizada).

Acople óptico (extensión de J&B 2018):
    El flujo lumínico que llega a la retina es proporcional al área de la
    apertura pupilar (∝ d²), cerrando el loop óptico-mecánico.

Objetivo: Investigar la interacción entre epsilon (decaimiento de las cajas)
y tau (retraso de transmisión) en la generación de oscilaciones tipo "pupillary
hippus". Grid 5x5 mostrando solo el estado estacionario (últimos 10 segundos).
Valores centrales: epsilon=0.2, tau=0.02 (por defecto). Extremos: 50% y 2x del valor central.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import os

# Agregar directorio padre al path para importar lib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.generadores.JohanssonBalkenius import JohanssonBalkeniusSystem, DefaultJBS

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

DT = 0.002  # 2 ms
T_TOTAL = 30.0  # 30 segundos de simulación
T_STEADY_START = 20.0  # Inicio del estado estacionario (últimos 10s)
N_STEPS = int(round(T_TOTAL / DT))
IDX_STEADY = int(round(T_STEADY_START / DT))

# Límites del diámetro pupilar (usados en graficado)
D_MIN, D_MAX = 2.0, 8.0

# Luminancia constante con ruido
L_CONSTANT = 0.5  # 50% de luminancia
NOISE_STD = 0.1   # desviación estándar del ruido

# Valores de epsilon (filas del grid)
# Centro: 0.2 (por defecto), extremos: 50% y 2x del valor central
EPSILON_VALUES = [0.02, 0.01, 0.20, 1.0, 2.0]

# Valores de tau (columnas del grid)
# Centro: 0.02 (por defecto), extremos: 50% y 2x del valor central
TAU_VALUES = [0.002, 0.01, 0.020, 0.100, 0.20]

# ---------------------------------------------------------------------------
# Ejecutar simulaciones
# ---------------------------------------------------------------------------

results = {}  # (epsilon, tau) -> datos

for eps in EPSILON_VALUES:
    for tau in TAU_VALUES:
        # Crear sistema fresco para cada combinación
        system = JohanssonBalkeniusSystem("default")
        
        # Modificar parámetros
        system.set_all_epsilon(eps)
        system.set_all_tau(tau)
        
        # Configurar inputs/outputs estándar usando DefaultJBS
        DefaultJBS.setup_standard_inputs(system)
        DefaultJBS.setup_standard_outputs(system)
        
        # Arrays para guardar resultados
        time_arr = np.empty(N_STEPS)
        pupil_l_arr = np.empty(N_STEPS)
        pupil_r_arr = np.empty(N_STEPS)
        
        # Diámetros iniciales
        d_left = DefaultJBS.pupil_diameter(0.0, 0.0)
        d_right = DefaultJBS.pupil_diameter(0.0, 0.0)
        
        print(f"Simulando ε={eps:.3f}, τ={tau:.3f}s...")
        
        for i in range(N_STEPS):
            t = system.t
            
            # Luminancia constante con ruido gaussiano
            L = np.clip(np.random.normal(L_CONSTANT, NOISE_STD), 0.0, 1.0)
            
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
            pupil_l_arr[i] = system.get_output("pupil_left")
            pupil_r_arr[i] = system.get_output("pupil_right")
            
            # Actualizar diámetros para próximo paso
            d_left = pupil_l_arr[i]
            d_right = pupil_r_arr[i]
        
        # Guardar solo estado estacionario (últimos 10s)
        results[(eps, tau)] = {
            'time': time_arr[IDX_STEADY:],
            'pupil_left': pupil_l_arr[IDX_STEADY:],
            'pupil_right': pupil_r_arr[IDX_STEADY:],
        }

print("Hecho.")

# ---------------------------------------------------------------------------
# Gráfico: Grid 4x4
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(5, 5, figsize=(18, 16), sharex=True, sharey=True)
axes = axes.flatten()

# Etiquetas para filas y columnas
row_labels = [f'ε={eps:.3f}' for eps in EPSILON_VALUES]
col_labels = [f'τ={tau*1000:.0f}ms' for tau in TAU_VALUES]

for idx_eps, eps in enumerate(EPSILON_VALUES):
    for idx_tau, tau in enumerate(TAU_VALUES):
        ax = axes[idx_eps * 5 + idx_tau]
        data = results[(eps, tau)]
        
        # Plot diámetro pupilar (ojo izquierdo)
        ax.plot(data['time'], data['pupil_left'], color='indigo', linewidth=1.0,
                label='Ojo izq.')
        ax.plot(data['time'], data['pupil_right'], color='darkorange', linewidth=1.0,
                linestyle=':', label='Ojo der.')
        
        # Línea de referencia
        ax.axhline(3.0, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
        
        # Título de la celda (solo en primera fila para tau)
        if idx_eps == 0:
            ax.set_title(f'τ={tau*1000:.0f}ms', fontsize=10, fontweight='bold')
        
        # Etiqueta de epsilon en el borde izquierdo
        if idx_tau == 0:
            ax.set_ylabel(f'ε={eps:.3f}', fontsize=11, fontweight='bold', rotation=90, labelpad=20)
        
        # Grid ligero
        ax.grid(True, alpha=0.2, linewidth=0.5)
        
        # Límites
        ax.set_xlim(T_STEADY_START, T_TOTAL)
        ax.set_ylim(D_MIN - 0.5, D_MAX + 0.5)

# Etiquetas de ejes externos
for i in range(5):
    axes[20 + i].set_xlabel('Tiempo (s)', fontsize=10)

# Agregar etiqueta general para el eje Y
fig.text(0.02, 0.5, 'Diámetro pupilar (mm)', va='center', rotation='vertical', fontsize=12, fontweight='bold')

# Título general
plt.suptitle(
    'Johansson & Balkenius (2018) — Grid ε vs τ: Oscilaciones tipo "pupillary hippus"\n'
    f'Grid 5x5 | Centro: ε=0.2, τ=0.02 (por defecto) | '
    f'Estado estacionario (últimos {T_TOTAL - T_STEADY_START:.0f}s) | '
    f'Luminancia {L_CONSTANT*100:.0f}% ± σ={NOISE_STD} | Corticales en 0',
    fontsize=12, fontweight='bold', y=0.995
)

plt.tight_layout()
plt.savefig('pupil_hippus_grid.png', dpi=150)
print("Figura guardada → pupil_hippus_grid.png")
plt.close()
