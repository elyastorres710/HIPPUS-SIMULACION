"""
Test A04: Respuesta Pupilar a Escalones de Luminancia con ConfigurableJBS

Este test evalúa la respuesta dinámica del sistema JBS optimizado ante 
escalones de luminancia, utilizando el nuevo método simulation_loop con ruido
bilateral y estocástico opcional.

Objetivo:
- Evaluar respuesta dinámica a cambios abruptos de luminancia
- Utilizar 3-4 escalones de luminancia diferentes
- Incorporar ruido realista (bilateral y estocástico)
- Graficar respuesta temporal de ambos ojos
- Comparar con comportamiento esperado

Basado en: test_04_JB_generator.py adaptado a ConfigurableJBS
"""
import os
import sys

import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Agregar directorio padre al path para importar lib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.generadores.JohanssonBalkenius import ConfigurableJBS, DEFAULT_CONFIG

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

# Escalones de luminancia con tiempo absoluto (reemplazo implícito)
LUMINANCE_STEPS = [
    (5.0, 0.0),    # hasta t=5.0s: L=0.0 (oscuridad)
    (15.0, 0.3),    # hasta t=5.0s: L=0.3 (luz media)
    (25.0, 0.7),   # hasta t=15.0s: L=0.7 (luz alta)
    (30.0, 0.1)    # hasta t=25.0s: L=0.1 (luz baja)
]

# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

def create_stabilized_system(L_background=0.0, T_stabilize=60.0):
    """
    Estabiliza el sistema con luminancia de fondo constante (de test_02_JB_pulsed_light.py).
    
    Args:
        system: Sistema JBS a estabilizar
        L_background: Luminancia de fondo
        T_stabilize: Tiempo de estabilización
        
    Returns:
        system: Sistema estabilizado con tiempo
    """
    custom_config = copy.deepcopy(DEFAULT_CONFIG)
    custom_config['default_jitter_epsilon'] = 0.05 # Jitter biológico de retraso en procesamiento de los nucleos
                        # define una realización del sujeto
    custom_config['default_tau_rec'] = 0.5 # Constante de recuperación del pool vesicular [s]:
                        # bajo → sinapsis robusta; alto → fatiga sostenida.
    custom_config['default_U'] = 0.8 # Fracción del recurso consumida por unidad de actividad [-]:
                        # 0 = sin depleción (caso JB ideal); 1 = agotamiento inmediato total.
    custom_config['default_sigma'] = 0.6 # Amplitud del ruido multiplicativo de canal [-]:
                        # escala con el agotamiento (1−u); 0 = determinista (caso JB ideal).
    custom_config['default_tau_jitter'] = 0.2  # Variabilidad temporal del retraso como fracción de τ [-]:
                        # 20% → σ_jitter = 0.2·τ; 0 = timing exacto (caso JB ideal).

    system = ConfigurableJBS(custom_config)
    system.config('enable_bilateral_noise', True)
    system.config('enable_stochastic_noise', True)
    
    system.enable_history = False

    n_steps = int(round(T_stabilize / system.dt))

    
    for i in range(n_steps):
        system.step_simulation(L=L_background)

    # resetea el tiempo y habilita la historia
    system.t = 0
    system.enable_history = True
    
    # Mostrar especificaciones del sistema
    print(f"Sistema creado y estabilizado con especificaciones:")
    print(f"  \033[91mParámetros pupilar:\033[0m") # red
    print(f"    D_min = \033[93m{system.D_min:.2f}\033[0m mm")
    print(f"    D_max = \033[93m{system.D_max:.2f}\033[0m mm")
    print(f"    D_ref = \033[93m{system.D_ref:.2f}\033[0m mm")
    print(f"  \033[94mParámetros luminancia:\033[0m") # blue
    print(f"    L_gain = \033[93m{system.L_gain:.4f}\033[0m")
    print(f"    L_gamma = \033[93m{system.L_gamma:.4f}\033[0m")
    print(f"  \033[95mRuido configurado:\033[0m") # magenta
    print(f"    L_BILATERAL_NOISE_STD = \033[93m{system._config['L_BILATERAL_NOISE_STD']:.3f}\033[0m")
    print(f"    L_STOCHASTIC_NOISE_STD = \033[93m{system._config['L_STOCHASTIC_NOISE_STD']:.3f}\033[0m")
    print(f"  \033[96mParámetros sinápticos:\033[0m") # cyan
    print(f"    tau_rec = \033[93m{custom_config['default_tau_rec']}\033[0m s (recuperación)")
    print(f"    U = \033[93m{custom_config['default_U']}\033[0m (uso de recursos)")
    print(f"    sigma = \033[93m{custom_config['default_sigma']}\033[0m (ruido de canal)")
    print(f"    tau_jitter = \033[93m{custom_config['default_tau_jitter']}\033[0m (variabilidad temporal)")
    print(f"    epsilon_jitter =  \033[93m{custom_config['default_jitter_epsilon']}\033[0m (variabilidad respuesta)")
    print(f"  \033[92mEstabilización:\033[0m \033[93m{T_stabilize}s\033[0m con L=\033[93m{L_background}\033[0m")
    print()
    
    return system


def run_simulation(system:ConfigurableJBS):
    """
    Ejecuta simulación completa con escalones de luminancia, replicando estructura de test_04.
    
    Args:
        system: Sistema JBS configurado
        enable_bilateral_noise: Si habilitar ruido bilateral
        enable_stochastic_noise: Si habilitar ruido estocástico
        
    Returns:
        dict: Resultados completos de la simulación con todas las señales intermedias
    """
    t_total = max([step_time for step_time,_ in LUMINANCE_STEPS])
    dt = system.dt
    n_steps = int(t_total/dt)

    print(f"Ejecutando simulación de {t_total}s ({n_steps} pasos a dt={dt*1000:.1f}ms)...")
    # Pre-asignar arrays para velocidad — todos los núcleos del trayecto
    # parasimpático/simpático se muestrean lateralizados (estilo test_04)
    time_arr      = np.empty(n_steps)
    source_arr    = np.empty(n_steps)   # luminancia de fuente (lo que emite el ambiente)
    retinal_l_arr = np.empty(n_steps)   # entrada efectiva a retina izquierda
    retinal_r_arr = np.empty(n_steps)   # entrada efectiva a retina derecha
    pta_l_arr     = np.empty(n_steps)
    pta_r_arr     = np.empty(n_steps)
    ewpg_l_arr    = np.empty(n_steps)
    ewpg_r_arr    = np.empty(n_steps)
    cg_l_arr      = np.empty(n_steps)
    cg_r_arr      = np.empty(n_steps)
    scg_l_arr     = np.empty(n_steps)
    scg_r_arr     = np.empty(n_steps)
    lc_l_arr      = np.empty(n_steps)
    lc_r_arr      = np.empty(n_steps)
    pupil_l_arr   = np.empty(n_steps)
    pupil_r_arr   = np.empty(n_steps)
    
    # Establecer baseline cortical
    system.set_cortical_baseline()
    
    # Ejecutar simulación
    for i in range(n_steps):
        t = system.t
        
        # Determinar nivel de luminancia actual según escalones (forma simple)
        current_L = next((step_l for step_t,step_l in LUMINANCE_STEPS if t <= step_t),0)

        # Ejecutar paso de simulación con ruido
        system.step_simulation(L=current_L)
        
        # Obtener estado completo
        state = system.view_state("all")
        
        # Muestrear estado actual (estilo test_04)
        time_arr[i]      = t
        source_arr[i]    = current_L
        pta_l_arr[i]     = state['box'].get('PTA_l', 0.0)
        pta_r_arr[i]     = state['box'].get('PTA_r', 0.0)
        ewpg_l_arr[i]    = state['box'].get('EWpg_l', 0.0)
        ewpg_r_arr[i]    = state['box'].get('EWpg_r', 0.0)
        cg_l_arr[i]      = state['box'].get('CG_l', 0.0)
        cg_r_arr[i]      = state['box'].get('CG_r', 0.0)
        scg_l_arr[i]     = state['box'].get('SCG_l', 0.0)
        scg_r_arr[i]     = state['box'].get('SCG_r', 0.0)
        lc_l_arr[i]      = state['box'].get('LC_l', 0.0)
        lc_r_arr[i]      = state['box'].get('LC_r', 0.0)
        pupil_l_arr[i]   = state['pupil_left']
        pupil_r_arr[i]   = state['pupil_right']
    
    print("Hecho.")
    
    return {
        'time': time_arr,
        'source': source_arr,
        'retinal_left': retinal_l_arr,
        'retinal_right': retinal_r_arr,
        'pta_left': pta_l_arr,
        'pta_right': pta_r_arr,
        'ewpg_left': ewpg_l_arr,
        'ewpg_right': ewpg_r_arr,
        'cg_left': cg_l_arr,
        'cg_right': cg_r_arr,
        'scg_left': scg_l_arr,
        'scg_right': scg_r_arr,
        'lc_left': lc_l_arr,
        'lc_right': lc_r_arr,
        'pupil_left': pupil_l_arr,
        'pupil_right': pupil_r_arr
    }

# ---------------------------------------------------------------------------
# Ejecución principal
# ---------------------------------------------------------------------------

def main():
    """Función principal del test."""
    print("=== Test A04: Respuesta Pupilar a Escalones de Luminancia ===\n")
    
    # Crear sistema
    system = create_stabilized_system()
    
    # Ejecutar simulación principal
    results = run_simulation(system)
    
    print("Simulación completada. Generando gráficos...")
    
    # ---------------------------------------------------------------------------
    # Generar gráficos (estilo test_04 completo)
    # ---------------------------------------------------------------------------
    
    fig = plt.figure(figsize=(13, 10))
    gs  = gridspec.GridSpec(3, 1, hspace=0.45)
    
    # --- Panel 1: Luminancia de fuente vs entrada retinal efectiva ---
    # Visualiza el efecto del closed-loop óptico: la fuente cambia en escalón
    # pero la luz que efectivamente llega a la retina queda modulada
    ax1 = fig.add_subplot(gs[0])
    ax1.step(results['time'], results['source'], where='post', color='goldenrod', linewidth=2,
             label='Luminancia de fuente L')
    ax1.fill_between(results['time'], results['source'], step='post', alpha=0.15, color='goldenrod')
    ax1.set_ylabel('Luminancia (u.a.)', fontsize=11)
    ax1.set_title('Entrada: fuente vs efectiva en retina (closed-loop óptico)',
                  fontsize=12, fontweight='bold')
    ax1.set_ylim(-0.05, 1.15)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Anotar valores de escalón de fuente
    for time,L_level in LUMINANCE_STEPS:
        ax1.axvline(time, color='gray', linestyle='--', alpha=0.4)
        ax1.text(time + 0.3, L_level + 0.03, f'{L_level:.2f}',
                 fontsize=9, color='saddlebrown')
    
    # --- Panel 2: Señales intermedias (lateralizadas) ---
    # Cada núcleo se grafica con dos curvas: lado izquierdo (sólida) y lado
    # derecho (punteada). Por simetría del estímulo lumínico, ambas curvas
    # deben superponerse — sirve como verificación visual del cableado bilateral.
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(results['time'], results['pta_left'],   label='PTA L',  color='steelblue', linewidth=1.2)
    ax2.plot(results['time'], results['pta_right'],  label='PTA R',  color='steelblue', linewidth=1.2, linestyle=':')
    ax2.plot(results['time'], results['ewpg_left'],  label='EWpg L', color='royalblue', linewidth=1.2)
    ax2.plot(results['time'], results['ewpg_right'], label='EWpg R', color='royalblue', linewidth=1.2, linestyle=':')
    ax2.plot(results['time'], results['cg_left'],    label='CG L (esfínter)',    color='crimson',  linewidth=1.5)
    ax2.plot(results['time'], results['cg_right'],   label='CG R (esfínter)',    color='crimson',  linewidth=1.5, linestyle=':')
    ax2.plot(results['time'], results['scg_left'],   label='SCG L (dilatador)',  color='seagreen', linewidth=1.5)
    ax2.plot(results['time'], results['scg_right'],  label='SCG R (dilatador)',  color='seagreen', linewidth=1.5, linestyle=':')
    ax2.plot(results['time'], results['lc_left'],    label='LC (izq.)', color='darkviolet', linewidth=1.5)
    ax2.plot(results['time'], results['lc_right'],   label='LC (der.)', color='darkviolet', linewidth=1.5, linestyle=':')    
    ax2.set_ylabel('Salida o (u.a.)', fontsize=11)
    ax2.set_title('Activaciones nucleares intermedias (lateralizadas)', fontsize=12, fontweight='bold')
    ax2.set_ylim(-2.0, 2.0)
    ax2.legend(loc='upper right', fontsize=8, ncol=4)
    ax2.grid(True, alpha=0.3)
    
    # Marcar cambios de escalón
    for time, L_level in LUMINANCE_STEPS:
        ax2.axvline(time, color='gray', linestyle='--', alpha=0.4)
    
    # --- Panel 3: Respuesta del diámetro pupilar (por ojo) ---
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(results['time'], results['pupil_left'], color='darkgreen', linewidth=2, 
             label='Ojo izquierdo')
    ax3.plot(results['time'], results['pupil_right'], color='darkorange', linewidth=2, 
             linestyle='--', label='Ojo derecho')
    ax3.set_xlabel('Tiempo (s)', fontsize=11)
    ax3.set_ylabel('Diámetro pupilar (mm)', fontsize=11)
    ax3.set_title('Respuesta del diámetro pupilar (por ojo)', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Marcar cambios de escalón
    for time, L_level in LUMINANCE_STEPS:
        ax2.axvline(time, color='gray', linestyle='--', alpha=0.4)
    
    # Título general (estilo test_04)
    plt.suptitle(
        'Johansson & Balkenius (2018) — Prueba de reflejo de luz (lateralizado)\n'
        'Parasimpático: Retinas→PTA→EWpg→CG   |   Simpático: PVN→IML→SCG   |   '
        'Loop CB↔EWpg vía EWpg_pred',
        fontsize=11, y=1.01
    )
    
    plt.tight_layout()
    
    # Guardar gráfico
    plot_file = "data/test/A04_luminance_steps_response.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Gráfico guardado en: {plot_file}")
    
    plt.show()
    
    print("\n=== Test A04 Completado ===")

if __name__ == "__main__":
    main()
