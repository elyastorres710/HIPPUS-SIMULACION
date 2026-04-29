"""
Prueba: Reflejo de luz pupilar con 3 escalones de luminancia de fuente
durante 30 segundos, con acople óptico-mecánico cerrado (closed-loop).

Topología bilateral (lateralizada L/R):
    Parasimpático: Retinas → PTA → EWpg → CG  → Esfínter (constricción)
    Simpático:     PVN     → IML → SCG         → Dilatador (dilatación)
    Loop predictor cerebellar: CB ↔ EWpg vía EWpg_pred (resta lateralizada).

Acople óptico (extensión de J&B 2018, no implementada en el paper original):
    El flujo lumínico que llega a la retina es proporcional al área de la
    apertura pupilar (∝ d²), no a la luminancia bruta de la fuente. Esto
    cierra el loop óptico-mecánico:
        más luz → constricción → menor área → menos luz a la retina → equilibrio
    sin esta retroalimentación el modelo opera en circuito abierto y satura
    rápidamente. Ver función `optical_input` para detalles del modelo L2
    (cuadrático normalizado por diámetro pupilar normal).

Diámetro pupilar (mm) — aproximación lineal:
    d = 5.0 - 4.0 * CG.o + 1.0 * SCG.o,  limitado a [2, 8] mm
    Calculado por ojo (d_left, d_right).

Nota: la escala absoluta de luminancia debería calibrarse contra datos
empíricos. El objetivo aquí es comportamiento cualitativo.
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

DT  = 0.002   # 2 ms  — más fino que los 20 ms del artículo para mejor precisión RK4
TAU = 0.020   # 20 ms — retraso de transmisión (por defecto del artículo)

# Constante de tiempo del nodo restador EWpg_pred (predictor cerebellar
# lateralizado). Determina la dinámica del loop CB↔EWpg y, a futuro, el
# rango en que pueden emerger oscilaciones tipo "pupillary hippus".
EPS_SUB = 1.0

# ---------------------------------------------------------------------------
# Estímulo: luminancia de la FUENTE (no entrada retinal directa)
# ---------------------------------------------------------------------------
# La luminancia de la fuente es lo que el ambiente emite. Lo que llega a la
# retina depende además del área de apertura pupilar (función optical_input).
#
#  t=0s  → tenue     (0.10)
#  t=5s  → moderada  (0.50)
#  t=15s → brillante (0.90)
#  t=25s → tenue de nuevo (0.10)

SOURCE_LUMINANCE_SCHEDULE = [
    (0.0,  0.10),
    (5.0,  0.50),
    (15.0, 0.90),
    (25.0, 0.10),
]

def get_source_luminance(t: float) -> float:
    """
    Luminancia de la fuente lumínica en el tiempo t.

    Esto es la luz emitida por el ambiente, NO lo que llega a la retina
    (ver optical_input para la modulación por apertura pupilar).

    Args:
        t (float): Tiempo en segundos.

    Returns:
        float: Luminancia de fuente (u.a.).
    """
    level = SOURCE_LUMINANCE_SCHEDULE[0][1]
    for t_on, intensity in SOURCE_LUMINANCE_SCHEDULE:
        if t >= t_on:
            level = intensity
    return level

# ---------------------------------------------------------------------------
# Acople óptico-mecánico (closed-loop)
# ---------------------------------------------------------------------------
# Las funciones optical_input y pupil_diameter ahora se obtienen de DefaultJBS

# Límites del diámetro pupilar (usados en graficado)
D_MIN, D_MAX = 2.0, 8.0

# ---------------------------------------------------------------------------
# Ejecutar simulación
# ---------------------------------------------------------------------------

system = JohanssonBalkeniusSystem("default")

# Configurar inputs/outputs estándar usando DefaultJBS
DefaultJBS.setup_standard_inputs(system)
DefaultJBS.setup_standard_outputs(system)

T_TOTAL = 30.0
N_STEPS = int(round(T_TOTAL / DT))

# Cada retina recibe luz a través de SU ojo (anatómicamente correcto):
#   left_retinae_l/r  ← d_left
#   right_retinae_l/r ← d_right
# En este test el estímulo es simétrico, así que d_left = d_right en todo
# instante. La distinción importa para experimentos futuros (p.ej. parche
# en un ojo, reflejo consensual).

# Pre-asignar arrays para velocidad — todos los núcleos del trayecto
# parasimpático/simpático se muestrean lateralizados.
time_arr      = np.empty(N_STEPS)
source_arr    = np.empty(N_STEPS)   # luminancia de fuente (lo que emite el ambiente)
retinal_l_arr = np.empty(N_STEPS)   # entrada efectiva a retina izquierda (post-pupila)
retinal_r_arr = np.empty(N_STEPS)   # entrada efectiva a retina derecha
pta_l_arr     = np.empty(N_STEPS)
pta_r_arr     = np.empty(N_STEPS)
ewpg_l_arr    = np.empty(N_STEPS)
ewpg_r_arr    = np.empty(N_STEPS)
cg_l_arr      = np.empty(N_STEPS)
cg_r_arr      = np.empty(N_STEPS)
scg_l_arr     = np.empty(N_STEPS)
scg_r_arr     = np.empty(N_STEPS)
pupil_l_arr   = np.empty(N_STEPS)
pupil_r_arr   = np.empty(N_STEPS)

# Diámetros iniciales: estado de reposo del sistema (CG=SCG=0 ⇒ d=5mm).
d_left  = DefaultJBS.pupil_diameter(0.0, 0.0)
d_right = DefaultJBS.pupil_diameter(0.0, 0.0)

print(f"Ejecutando simulación de {T_TOTAL}s ({N_STEPS} pasos a dt={DT*1000:.1f}ms)...")

for i in range(N_STEPS):
    t = system.t

    # --- Closed-loop óptico-mecánico ---
    # La entrada efectiva a cada retina depende de (a) la luminancia de la
    # fuente, y (b) el área de apertura pupilar de SU ojo (modelo L2).
    # Usamos el diámetro del paso anterior (o el inicial en i=0): esto es
    # físicamente correcto porque la pupila no puede reaccionar antes de
    # que la luz la golpee.
    L = get_source_luminance(t)
    alpha_left  = DefaultJBS.optical_input(L, d_left)
    alpha_right = DefaultJBS.optical_input(L, d_right)
    system.set_input("retina_left", alpha_left)
    system.set_input("retina_right", alpha_right)
    # Corticales en 0 (reflejo autonómico puro)
    DefaultJBS.set_cortical_baseline(system, excitatory=0.0, emotional=0.0, novelty=0.0)

    # Paso síncrono
    system.step()

    # Muestrear estado actual y actualizar diámetros para el próximo paso
    time_arr[i]      = t
    source_arr[i]    = L
    retinal_l_arr[i] = alpha_left
    retinal_r_arr[i] = alpha_right
    pta_l_arr[i]     = system.cajas['PTA_l'].o
    pta_r_arr[i]     = system.cajas['PTA_r'].o
    ewpg_l_arr[i]    = system.cajas['EWpg_l'].o
    ewpg_r_arr[i]    = system.cajas['EWpg_r'].o
    cg_l_arr[i]      = system.cajas['CG_l'].o
    cg_r_arr[i]      = system.cajas['CG_r'].o
    scg_l_arr[i]     = system.cajas['SCG_l'].o
    scg_r_arr[i]     = system.cajas['SCG_r'].o
    d_left  = system.get_output("pupil_left")
    d_right = system.get_output("pupil_right")
    pupil_l_arr[i] = d_left
    pupil_r_arr[i] = d_right

print("Hecho.")

# ---------------------------------------------------------------------------
# Gráfico
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(13, 10))
gs  = gridspec.GridSpec(3, 1, hspace=0.45)

# --- Panel 1: Luminancia de fuente vs entrada retinal efectiva ---
# Visualiza el efecto del closed-loop óptico: la fuente cambia en escalón
# (lo que emite el ambiente), pero la luz que efectivamente llega a la
# retina queda modulada por el área pupilar y se suaviza dinámicamente.
ax1 = fig.add_subplot(gs[0])
ax1.step(time_arr, source_arr, where='post', color='goldenrod', linewidth=2,
         label='Luminancia de fuente L')
ax1.fill_between(time_arr, source_arr, step='post', alpha=0.15, color='goldenrod')
ax1.plot(time_arr, retinal_l_arr, color='darkred', linewidth=1.5,
         label='Entrada retinal efectiva (ojo izq.)')
ax1.plot(time_arr, retinal_r_arr, color='maroon', linewidth=1.5, linestyle=':',
         label='Entrada retinal efectiva (ojo der.)')
ax1.set_ylabel('Luminancia (u.a.)', fontsize=11)
ax1.set_title('Entrada: fuente vs efectiva en retina (closed-loop óptico)',
              fontsize=12, fontweight='bold')
ax1.set_ylim(-0.05, 1.15)
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)

# Anotar valores de escalón de fuente
for t_on, intensity in SOURCE_LUMINANCE_SCHEDULE:
    ax1.axvline(t_on, color='gray', linestyle='--', alpha=0.4)
    ax1.text(t_on + 0.3, intensity + 0.03, f'{intensity:.2f}',
             fontsize=9, color='saddlebrown')

# --- Panel 2: Señales intermedias (lateralizadas) ---
# Cada núcleo se grafica con dos curvas: lado izquierdo (sólida) y lado
# derecho (punteada). Por simetría del estímulo lumínico (las 4 retinas
# reciben lo mismo), ambas curvas deben superponerse — sirve como
# verificación visual del cableado bilateral.
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.plot(time_arr, pta_l_arr,  label='PTA L',  color='steelblue', linewidth=1.2)
ax2.plot(time_arr, pta_r_arr,  label='PTA R',  color='steelblue', linewidth=1.2, linestyle=':')
ax2.plot(time_arr, ewpg_l_arr, label='EWpg L', color='royalblue', linewidth=1.2)
ax2.plot(time_arr, ewpg_r_arr, label='EWpg R', color='royalblue', linewidth=1.2, linestyle=':')
ax2.plot(time_arr, cg_l_arr,   label='CG L (esfínter)',    color='crimson',  linewidth=1.5)
ax2.plot(time_arr, cg_r_arr,   label='CG R (esfínter)',    color='crimson',  linewidth=1.5, linestyle=':')
ax2.plot(time_arr, scg_l_arr,  label='SCG L (dilatador)',  color='seagreen', linewidth=1.5)
ax2.plot(time_arr, scg_r_arr,  label='SCG R (dilatador)',  color='seagreen', linewidth=1.5, linestyle=':')
ax2.set_ylabel('Salida  o  (u.a.)', fontsize=11)
ax2.set_title('Activaciones nucleares intermedias (lateralizadas)', fontsize=12, fontweight='bold')
ax2.set_ylim(-0.05, 1.15)
ax2.legend(loc='upper right', fontsize=8, ncol=4)
ax2.grid(True, alpha=0.3)
for t_on, _ in SOURCE_LUMINANCE_SCHEDULE:
    ax2.axvline(t_on, color='gray', linestyle='--', alpha=0.4)

# --- Panel 3: Diámetro pupilar (uno por ojo) ---
ax3 = fig.add_subplot(gs[2], sharex=ax1)
ax3.plot(time_arr, pupil_l_arr, color='indigo',     linewidth=2,   label='Ojo izquierdo')
ax3.plot(time_arr, pupil_r_arr, color='darkorange', linewidth=2,   linestyle=':', label='Ojo derecho')
ax3.fill_between(time_arr, pupil_l_arr, D_MIN, alpha=0.1, color='indigo')
ax3.axhline(3.0, color='gray', linestyle=':', alpha=0.6, label='≈ luz normal (3 mm)')
ax3.set_ylabel('Diámetro pupilar (mm)', fontsize=11)
ax3.set_xlabel('Tiempo (s)', fontsize=11)
ax3.set_title('Respuesta del diámetro pupilar (por ojo)', fontsize=12, fontweight='bold')
# ax3.invert_yaxis()   # convención: constricción hacia abajo
ax3.legend(loc='lower right', fontsize=9)
ax3.grid(True, alpha=0.3)
for t_on, _ in SOURCE_LUMINANCE_SCHEDULE:
    ax3.axvline(t_on, color='gray', linestyle='--', alpha=0.4)

# Límites compartidos del eje x
ax3.set_xlim(0, T_TOTAL)

plt.suptitle(
    'Johansson & Balkenius (2018) — Prueba de reflejo de luz (lateralizado)\n'
    'Parasimpático: Retinas→PTA→EWpg→CG   |   Simpático: PVN→IML→SCG   |   '
    'Loop CB↔EWpg vía EWpg_pred',
    fontsize=11, y=1.01
)

out_path = 'pupil_light_reflex_test.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Figura guardada → {out_path}")
plt.show()
