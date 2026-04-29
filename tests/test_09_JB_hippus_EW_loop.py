"""
Prueba: Oscilaciones tipo "pupillary hippus" variando tau SOLO en las
conexiones del lazo cerebellar EWpg <-> CB <-> EWpg_pred -> EWpg.

Hipotesis fisiologica:
    El hippus pupilar asociado a fatiga / somnolencia se interpreta como una
    perdida de damping del lazo de control parasimpatico, no como una
    modificacion global de la dinamica del sistema. Las conexiones del ciclo
    CB <-> EWpg son el locus natural donde tal degradacion ocurre (adaptacion
    neuronal central de EWpg, depresion sinaptica de la modulacion LC->EWpg,
    o remodelacion del feedback cerebellar predictor).

A diferencia de test_07 (donde se varia tau de TODAS las conexiones), aqui
SOLO se modifican las 8 conexiones del lazo cerebellar:

    EWpg_l       -> CB              (us)
    EWpg_r       -> CB              (us)
    CB           -> EWpg_pred_l     (excitatory)
    CB           -> EWpg_pred_r     (excitatory)
    EWpg_l       -> EWpg_pred_l     (inhibitory, sustraccion local)
    EWpg_r       -> EWpg_pred_r     (inhibitory, sustraccion local)
    EWpg_pred_l  -> EWpg_l          (excitatory, retorno al EW)
    EWpg_pred_r  -> EWpg_r          (excitatory, retorno al EW)

El resto del sistema (vias parasimpatica primaria, simpatica, hipotalamica,
LC, corteza->AMY) mantiene tau=20 ms del paper. Esto aisla el efecto al
locus del ciclo y permite demostrar que el hippus emerge como propiedad
del lazo, no del sistema en su conjunto.

Topologia bilateral (lateralizada L/R):
    Parasimpatico: Retinas -> PTA -> EWpg -> CG  -> Esfinter (constriccion)
    Simpatico:     PVN     -> IML -> SCG          -> Dilatador (dilatacion)
    Loop predictor cerebellar: CB <-> EWpg via EWpg_pred (resta lateralizada).

Acople optico (extension de J&B 2018):
    El flujo luminico que llega a la retina es proporcional al area de la
    apertura pupilar (~ d^2), cerrando el loop optico-mecanico.

Objetivo: Investigar como el retraso del lazo EW (y solo el lazo) afecta la
dinamica del sistema y puede generar oscilaciones tipo "pupillary hippus"
en condiciones de luminancia constante con ruido.
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

DT = 0.002  # 2 ms
T_TOTAL = 30.0  # 30 segundos de simulacion
N_STEPS = int(round(T_TOTAL / DT))

# Limites del diametro pupilar (usados en graficado)
D_MIN, D_MAX = 2.0, 8.0

# Luminancia constante con ruido gaussiano (sigma = 0.1)
L_CONSTANT = 0.5  # 50% de luminancia

# Conexiones del lazo cerebellar EWpg <-> CB <-> EWpg_pred (8 conexiones).
# Estas son las unicas cuyo tau se modifica. El resto del sistema mantiene
# el tau por defecto (20 ms del paper).
LOOP_CONNECTIONS = {
    ('EWpg_l',      'CB'),              # us         - feedforward al cerebelo
    ('EWpg_r',      'CB'),              # us         - feedforward al cerebelo
    ('CB',          'EWpg_pred_l'),     # excitatory - prediccion cerebellar
    ('CB',          'EWpg_pred_r'),     # excitatory - prediccion cerebellar
    ('EWpg_l',      'EWpg_pred_l'),     # inhibitory - sustraccion local
    ('EWpg_r',      'EWpg_pred_r'),     # inhibitory - sustraccion local
    ('EWpg_pred_l', 'EWpg_l'),          # excitatory - retorno al EW
    ('EWpg_pred_r', 'EWpg_r'),          # excitatory - retorno al EW
}

# Valores de tau a probar SOLO sobre las conexiones del lazo.
# Default tau = 0.020 s (20 ms, el del paper), epsilon = 0.2. 
# Se probaran valores de tau desde 20ms hasta 400ms para mostrar la transicion
# sobreamortiguado -> sostenido (Hopf).
TAU_VALUES = [0.020, 0.100, 0.100, 0.200]  # 20ms, 100ms, 200ms, 400ms
EPSILON_VALUES = [0.020, 0.020, 0.020, 0.020]  # bajo, medio, alto, muy alto

# ---------------------------------------------------------------------------
# Helper: modificar tau solo en el lazo
# ---------------------------------------------------------------------------

def set_loop_tau(system: JohanssonBalkeniusSystem, tau: float) -> int:
    """
    Modifica el tau SOLO de las conexiones que forman el ciclo cerebellar
    EWpg <-> CB <-> EWpg_pred -> EWpg. El resto de las conexiones queda
    con su tau original (20 ms del default).

    Args:
        system: Sistema J&B ya construido con configuracion por defecto.
        tau: Nuevo valor de tau para las conexiones del lazo (en segundos).

    Returns:
        Numero de conexiones modificadas (deberia ser 8 si la topologia es
        la default).
    """
    n_modified = 0
    for dst_name, conns in system._incoming.items():
        for c in conns:
            src_name = c.fuente.name
            if (src_name, dst_name) in LOOP_CONNECTIONS:
                c.tau = tau
                n_modified += 1
    return n_modified

def set_loop_epsilon(system: JohanssonBalkeniusSystem, epsilon: float) -> int:
    """
    Modifica el epsilon SOLO de las conexiones que forman el ciclo cerebellar
    EWpg <-> CB <-> EWpg_pred -> EWpg. El resto de las conexiones queda
    con su epsilon original (20 ms del default).

    Args:
        system: Sistema J&B ya construido con configuracion por defecto.
        epsilon: Nuevo valor de epsilon para las conexiones del lazo (en segundos).

    Returns:
        Numero de conexiones modificadas (deberia ser 8 si la topologia es
        la default).
    """
    n_modified = 0
    for dst_name, conns in system._incoming.items():
        for c in conns:
            src_name = c.fuente.name
            if (src_name, dst_name) in LOOP_CONNECTIONS:
                c.epsilon = epsilon
                n_modified += 1
    return n_modified
# ---------------------------------------------------------------------------
# Ejecutar simulaciones
# ---------------------------------------------------------------------------

results = {}

for tau, epsilon in zip(TAU_VALUES, EPSILON_VALUES):
    # Sistema fresco para cada valor de tau (sin contaminacion de buffers)
    system = JohanssonBalkeniusSystem("default")

    # Modificar tau SOLO de las 8 conexiones del lazo cerebellar
    n_mod = set_loop_tau(system, tau)
    n_mod_epsilon = set_loop_epsilon(system, epsilon)

    if n_mod != len(LOOP_CONNECTIONS):
        print(f"  AVISO: se esperaban {len(LOOP_CONNECTIONS)} conexiones del "
              f"lazo, se modificaron {n_mod}. Verificar topologia default.")

    # Configurar inputs/outputs estandar usando DefaultJBS
    DefaultJBS.setup_standard_inputs(system)
    DefaultJBS.setup_standard_outputs(system)

    # Arrays para guardar resultados
    time_arr      = np.empty(N_STEPS)
    source_arr    = np.empty(N_STEPS)   # luminancia de fuente
    retinal_l_arr = np.empty(N_STEPS)
    retinal_r_arr = np.empty(N_STEPS)
    pupil_l_arr   = np.empty(N_STEPS)
    pupil_r_arr   = np.empty(N_STEPS)

    # Diametros iniciales
    d_left  = DefaultJBS.pupil_diameter(0.0, 0.0)
    d_right = DefaultJBS.pupil_diameter(0.0, 0.0)

    DefaultJBS.set_cortical_baseline(
        system,
        excitatory=0.50,
        emotional=0.0,
        novelty=0.0
    )


    print(f"Simulando con tau del lazo EW = {tau*1000:.0f}ms "
          f"(resto del sistema = 20ms)...")

    for i in range(N_STEPS):
        t = system.t

        # Luminancia constante con ruido gaussiano (sigma = 0.1)
        L = float(np.clip(np.random.normal(L_CONSTANT, 0.1), 0.0, 1.0))

        # Closed-loop optico (entrada efectiva ~ L * (d/D_REF)^2)
        alpha_left  = DefaultJBS.optical_input(L, d_left)
        alpha_right = DefaultJBS.optical_input(L, d_right)
        system.set_input("retina_left",  alpha_left)
        system.set_input("retina_right", alpha_right)
        system.set_input("cortex_excitatory", np.random.normal(0.50, 0.25))
        # system.set_input("cortex_emotional", np.random.normal(0.50, 0.25))
        # system.set_input("cortex_novelty", np.random.normal(0.50, 0.25))

        # Corticales en 0: reflejo autonomico puro, sin emocion ni novelty
        # DefaultJBS.set_cortical_baseline(system)

        # Paso de simulacion
        system.step()

        # Muestrear estado actual
        time_arr[i]      = t
        source_arr[i]    = L
        retinal_l_arr[i] = alpha_left
        retinal_r_arr[i] = alpha_right
        pupil_l_arr[i]   = system.get_output("pupil_left")
        pupil_r_arr[i]   = system.get_output("pupil_right")

        # Actualizar diametros para el proximo paso
        d_left  = pupil_l_arr[i]
        d_right = pupil_r_arr[i]

    results[(tau, epsilon)] = {
        'time':         time_arr,
        'source':       source_arr,
        'retinal_left': retinal_l_arr,
        'retinal_right':retinal_r_arr,
        'pupil_left':   pupil_l_arr,
        'pupil_right':  pupil_r_arr,
    }

print("Hecho.")

# ---------------------------------------------------------------------------
# Grafico
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(len(TAU_VALUES) + 1, 1, hspace=0.4)

# --- Panel 1: estimulo (luminancia con ruido) ---
ax0 = fig.add_subplot(gs[0])
data_first = results[(TAU_VALUES[0], EPSILON_VALUES[0])]
ax0.plot(data_first['time'], data_first['source'], color='goldenrod',
         linewidth=2, label='Luminancia constante con ruido')
ax0.fill_between(data_first['time'], data_first['source'],
                 alpha=0.15, color='goldenrod')
ax0.set_ylabel('Luminancia (u.a.)', fontsize=11)
ax0.set_title('Estimulo: Luminancia constante al 50% con ruido (sigma=0.1)',
              fontsize=12, fontweight='bold')
ax0.set_ylim(-0.05, 1.15)
ax0.legend(loc='upper right', fontsize=10)
ax0.grid(True, alpha=0.3)
ax0.set_xlim(0, T_TOTAL)

# --- Paneles 2..N: respuesta pupilar para cada tau del lazo ---
for idx, tau in enumerate(TAU_VALUES):
    ax = fig.add_subplot(gs[idx + 1], sharex=ax0)
    data = results[(TAU_VALUES[idx], EPSILON_VALUES[idx])]

    ax.plot(data['time'], data['pupil_left'],  color='indigo',
            linewidth=1.5, label='Ojo izquierdo')
    ax.plot(data['time'], data['pupil_right'], color='darkorange',
            linewidth=1.5, linestyle=':', label='Ojo derecho')
    ax.axhline(3.0, color='gray', linestyle=':', alpha=0.6,
               label='~ luz normal (3 mm)')
    ax.set_ylabel('Diametro pupilar (mm)', fontsize=11)
    ax.set_title(
        f'Diametro pupilar: tau_lazo = {tau*1000:.0f} ms '
        f'(resto del sistema = 20 ms)',
        fontsize=12, fontweight='bold'
    )
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, T_TOTAL)

    # Solo el ultimo panel lleva la etiqueta del eje x
    if idx == len(TAU_VALUES) - 1:
        ax.set_xlabel('Tiempo (s)', fontsize=11)

plt.suptitle(
    'Johansson & Balkenius (2018) - Oscilaciones tipo "pupillary hippus"\n'
    'Variando tau SOLO del lazo cerebellar EWpg <-> CB <-> EWpg_pred  |  '
    f'Luminancia constante al {L_CONSTANT*100:.0f}% con ruido (sigma=0.1)  |  '
    'Corticales en 0',
    fontsize=12, fontweight='bold', y=0.995
)

plt.tight_layout()

out_path = 'pupil_hippus_EW_loop.png'
plt.savefig(out_path, dpi=150)
print(f"Figura guardada -> {out_path}")
plt.close()
