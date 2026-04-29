"""
Prueba: Hippus por perdida de actualizacion cerebellar (US loss).

Hipotesis fisiologica:
    Durante fatiga / somnolencia, la proyeccion EWpg -> cerebelo (que en
    el modelo cumple el rol funcional de fibra trepadora del olivo inferior:
    señal de error / teaching) decae. CB queda sin actualizacion (regla
    delta congelada) y opera con sus pesos previos.

    Mientras la corteza coincida con el patron entrenado, CB sigue
    cancelando bien la salida de EWpg (cancelacion predictiva). Pero
    cuando la actividad cortical diverge del estado entrenado, CB pasa
    de cancelador predictivo a fuente de perturbacion en el lazo EWpg.
    Esto rompe la cancelacion y desestabiliza el sistema.

Fases de la simulacion:
    Fase 1 - Aprendizaje (t = 0 a T_FREEZE s):
        CB.plastic = True. La regla delta ajusta los pesos de CB
        (Ecuacion 3 del paper) para que su salida prediga la actividad
        de EWpg. Se espera que CB.o converja hacia EWpg.o, y por lo
        tanto EWpg_pred = CB.o - EWpg.o se mantenga cerca de cero.

    Fase 2 - Perdida de US (t = T_FREEZE a T_TOTAL s):
        CB.plastic = False. Los pesos quedan congelados al valor
        entrenado. CB sigue activo (corteza ruidosa drive) pero ya no
        se sincroniza con EWpg. El error EWpg_pred deja de cancelar
        y se inyecta como perturbacion en EWpg via el retorno
        EWpg_pred -> EWpg.

Estimulos (ambas condiciones):
    Luminancia constante 50% con ruido gaussiano sigma = 0.1
    Corticales con ruido gaussiano sigma = 0.25 alrededor de 0.5

Comparacion (dos simulaciones con MISMA secuencia de ruido):
    Control: CB.plastic siempre True (sin perdida de US)
    Test:    CB.plastic se apaga en t = T_FREEZE s (fatiga / perdida de US)

Topologia: configuracion por defecto del sistema J&B 2018, con tau
estandar (20 ms en todas las conexiones, 2.1 s en cortex_novelty -> AMY).
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
# Configuracion
# ---------------------------------------------------------------------------

DT = 0.002          # 2 ms
T_TOTAL = 30.0      # 30 segundos de simulacion
T_FREEZE = 10.0     # En t = T_FREEZE s se desactiva CB.plastic en la condicion 'test'

N_STEPS = int(round(T_TOTAL / DT))
N_FREEZE = int(round(T_FREEZE / DT))

# Limites del diametro pupilar (usados en graficado)
D_MIN, D_MAX = 2.0, 8.0

# Estimulo
L_CONSTANT       = 0.5    # luminancia base 50%
LUMINANCE_SIGMA  = 0.1
CORTEX_MEAN      = 0.5
CORTEX_SIGMA     = 0.25

# Semilla para reproducibilidad y para que ambas condiciones vean la
# MISMA secuencia de ruido (la unica diferencia entre control y test
# es el toggle de CB.plastic en t = T_FREEZE)
RNG_SEED = 42

CONDITIONS = [
    ('control', False, 'Control (CB plastico siempre activo)',     'steelblue'),
    ('test',    True,  f'Test (US perdido en t={T_FREEZE:.0f}s)',  'crimson'),
]

# ---------------------------------------------------------------------------
# Ejecutar simulaciones
# ---------------------------------------------------------------------------

results = {}

# Guardamos el estado inicial del RNG para resetearlo en cada condicion
np.random.seed(RNG_SEED)
initial_rng_state = np.random.get_state()

for cond_name, do_freeze, cond_label, cond_color in CONDITIONS:
    # Resetear RNG para que ambas condiciones vean la misma secuencia
    np.random.set_state(initial_rng_state)

    # Sistema fresco
    system = JohanssonBalkeniusSystem("default")
    DefaultJBS.setup_standard_inputs(system)
    DefaultJBS.setup_standard_outputs(system)

    # Arrays para guardar resultados
    time_arr       = np.empty(N_STEPS)
    source_arr     = np.empty(N_STEPS)
    cortex_arr     = np.empty(N_STEPS)
    pupil_l_arr    = np.empty(N_STEPS)
    cb_arr         = np.empty(N_STEPS)   # CB.o
    ewpg_arr       = np.empty(N_STEPS)   # promedio EWpg_l/r .o
    cb_weight_arr  = np.empty(N_STEPS)   # peso aprendido cortex_excitatory -> CB
    ewpg_pred_arr  = np.empty(N_STEPS)   # promedio EWpg_pred_l/r .o (señal de error)

    # Diametros iniciales
    d_left  = DefaultJBS.pupil_diameter(0.0, 0.0)
    d_right = DefaultJBS.pupil_diameter(0.0, 0.0)

    print(f"Simulando condicion: {cond_label}")

    for i in range(N_STEPS):
        t = system.t

        # Aplicar el freeze de plasticidad en la condicion 'test'
        if do_freeze and i == N_FREEZE:
            system.cajas['CB'].plastic = False
            print(f"  t={t:.2f}s -> CB.plastic = False (US perdido)")

        # Estimulos ruidosos
        L    = float(np.clip(np.random.normal(L_CONSTANT, LUMINANCE_SIGMA), 0.0, 1.0))
        cx_e = float(np.random.normal(CORTEX_MEAN, CORTEX_SIGMA))
        cx_m = float(np.random.normal(CORTEX_MEAN, CORTEX_SIGMA))
        cx_n = float(np.random.normal(CORTEX_MEAN, CORTEX_SIGMA))

        # Closed-loop optico
        alpha_left  = DefaultJBS.optical_input(L, d_left)
        alpha_right = DefaultJBS.optical_input(L, d_right)
        system.set_input("retina_left",       alpha_left)
        system.set_input("retina_right",      alpha_right)
        system.set_input("cortex_excitatory", cx_e)
        system.set_input("cortex_emotional",  cx_m)
        system.set_input("cortex_novelty",    cx_n)

        # Paso de simulacion
        system.step()

        # Muestrear estado
        time_arr[i]      = t
        source_arr[i]    = L
        cortex_arr[i]    = cx_e
        pupil_l_arr[i]   = system.get_output("pupil_left")
        cb_arr[i]        = system.cajas['CB'].o
        ewpg_arr[i]      = 0.5 * (system.cajas['EWpg_l'].o + system.cajas['EWpg_r'].o)
        cb_weight_arr[i] = system.cajas['CB']._weights.get(0, 0.0)
        ewpg_pred_arr[i] = 0.5 * (system.cajas['EWpg_pred_l'].o
                                  + system.cajas['EWpg_pred_r'].o)

        # Actualizar diametros para el proximo paso
        d_left  = pupil_l_arr[i]
        d_right = system.get_output("pupil_right")

    results[cond_name] = {
        'label':      cond_label,
        'color':      cond_color,
        'time':       time_arr,
        'source':     source_arr,
        'cortex':     cortex_arr,
        'pupil_left': pupil_l_arr,
        'cb_output':  cb_arr,
        'ewpg_output':ewpg_arr,
        'cb_weight':  cb_weight_arr,
        'ewpg_pred':  ewpg_pred_arr,
    }

print("Hecho.")

# ---------------------------------------------------------------------------
# Grafico
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(14, 13))
gs  = gridspec.GridSpec(5, 1, hspace=0.55)

# --- Panel 1: estimulos ---
ax0 = fig.add_subplot(gs[0])
t_axis = results['control']['time']
ax0.plot(t_axis, results['control']['source'], color='goldenrod',
         linewidth=1.2, alpha=0.8,
         label=f'Luminancia (sigma={LUMINANCE_SIGMA})')
ax0.plot(t_axis, results['control']['cortex'], color='navy',
         linewidth=1.0, alpha=0.6,
         label=f'cortex_excitatory (sigma={CORTEX_SIGMA})')
ax0.axvline(T_FREEZE, color='red', linestyle='--', alpha=0.7,
            label=f'Perdida US (t={T_FREEZE:.0f}s)')
ax0.set_ylabel('Estimulo (u.a.)', fontsize=10)
ax0.set_title('Estimulos: luminancia + drive cortical ruidosos '
              '(misma secuencia en ambas condiciones)',
              fontsize=11, fontweight='bold')
ax0.legend(loc='upper right', fontsize=9, ncol=3)
ax0.grid(True, alpha=0.3)
ax0.set_xlim(0, T_TOTAL)

# --- Panel 2: diametro pupilar, ambas condiciones overlaid ---
ax1 = fig.add_subplot(gs[1], sharex=ax0)
for cond_name, _, _, _ in CONDITIONS:
    r = results[cond_name]
    ax1.plot(r['time'], r['pupil_left'], color=r['color'],
             linewidth=1.4, label=r['label'])
ax1.axvline(T_FREEZE, color='red', linestyle='--', alpha=0.6)
ax1.axhline(3.0, color='gray', linestyle=':', alpha=0.5,
            label='~ luz normal (3 mm)')
ax1.set_ylabel('Diametro pupilar (mm)', fontsize=10)
ax1.set_title('Respuesta pupilar (ojo izquierdo): control vs test',
              fontsize=11, fontweight='bold')
ax1.legend(loc='lower right', fontsize=9)
ax1.grid(True, alpha=0.3)

# --- Panel 3: CB.o vs EWpg.o, condicion control ---
ax2 = fig.add_subplot(gs[2], sharex=ax0)
r = results['control']
ax2.plot(r['time'], r['ewpg_output'], color='forestgreen',
         linewidth=1.2, label='EWpg.o (promedio L/R)')
ax2.plot(r['time'], r['cb_output'], color='purple',
         linewidth=1.2, linestyle='--', label='CB.o')
ax2.axvline(T_FREEZE, color='red', linestyle='--', alpha=0.4)
ax2.set_ylabel('Salida nuclear', fontsize=10)
ax2.set_title('CONTROL: CB.o sigue a EWpg.o '
              '(cancelacion predictiva activa todo el tiempo)',
              fontsize=11, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)

# --- Panel 4: CB.o vs EWpg.o, condicion test ---
ax3 = fig.add_subplot(gs[3], sharex=ax0)
r = results['test']
ax3.plot(r['time'], r['ewpg_output'], color='forestgreen',
         linewidth=1.2, label='EWpg.o (promedio L/R)')
ax3.plot(r['time'], r['cb_output'], color='crimson',
         linewidth=1.2, linestyle='--', label='CB.o (congelado tras freeze)')
ax3.axvline(T_FREEZE, color='red', linestyle='--', alpha=0.7,
            label=f'US perdido')
ax3.set_ylabel('Salida nuclear', fontsize=10)
ax3.set_title('TEST: tras la perdida de US, CB.o se decorrelaciona '
              'de EWpg.o',
              fontsize=11, fontweight='bold')
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(True, alpha=0.3)

# --- Panel 5: peso aprendido en CB ---
ax4 = fig.add_subplot(gs[4], sharex=ax0)
for cond_name, _, _, _ in CONDITIONS:
    r = results[cond_name]
    ax4.plot(r['time'], r['cb_weight'], color=r['color'],
             linewidth=1.5, label=r['label'])
ax4.axvline(T_FREEZE, color='red', linestyle='--', alpha=0.6)
ax4.set_ylabel('w[cortex_excit. -> CB]', fontsize=10)
ax4.set_xlabel('Tiempo (s)', fontsize=11)
ax4.set_title('Peso aprendido en CB: '
              'la perdida de US congela la actualizacion de pesos',
              fontsize=11, fontweight='bold')
ax4.legend(loc='lower right', fontsize=9)
ax4.grid(True, alpha=0.3)

ax4.set_xlim(0, T_TOTAL)

plt.suptitle(
    'Johansson & Balkenius (2018) - Hippus por perdida de actualizacion '
    'cerebellar (US loss)\n'
    f'CB.plastic se desactiva en t={T_FREEZE:.0f}s simulando fatiga del '
    'input EWpg -> CB',
    fontsize=12, fontweight='bold', y=0.998
)

out_path = 'pupil_hippus_US_loss.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Figura guardada -> {out_path}")
plt.close()
