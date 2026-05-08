#!/usr/bin/env python3
"""
Test A05: Ruido Intencional Adicional

Basado en test_A04_luminance_steps.py pero incorporando una nueva entrada LC
para agregar ruido intencional en cada paso de simulación.

El ruido intencional simula factores externos que pueden afectar la percepción
luminosa o el estado emocional del sujeto, independientemente de la luminancia
ambiental.
"""

# Agregar directorio padre al path para importar lib
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from lib.generadores.JohanssonBalkenius import ConfigurableJBS, CUSTOM_CONFIG

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

DT = 0.002  # 2 ms
T_TOTAL = 30.0  # 30 segundos de simulación
N_STEPS = int(round(T_TOTAL / DT))

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
    custom_config = copy.deepcopy(CUSTOM_CONFIG)
    custom_config['default_jitter_epsilon'] = 0.05 # Jitter biológico de retraso en procesamiento de los nucleos
    custom_config['default_tau_rec'] = 0.5 # Constante de recuperación del pool vesicular [s]:
    custom_config['default_U'] = 0.8 # Fracción del recurso consumida por unidad de actividad [-]:
    custom_config['default_sigma'] = 0.6 # Amplitud del ruido multiplicativo de canal [-]:
    custom_config['default_tau_jitter'] = 0.2  # Variabilidad temporal del retraso como fracción de τ [-]:

    # Simulación de sujetos distintos
    custom_config['default_epsilon'] = custom_config['default_epsilon'] * (1 + np.random.normal(0,0.1))
    custom_config['default_tau']     = custom_config['default_tau'] * (1 + np.random.normal(0,0.1))

    custom_config['boxes'].append({'name': 'disturbance', 'alpha': 0.0,  'beta': 0, 'gamma': 0})
    
    # # Añadir conexiones individualmente
    # custom_config['connections'].append({'from': 'disturbance', 'to': 'EWpg_l', 'tipo': 'shunting'})
    # custom_config['connections'].append({'from': 'disturbance', 'to': 'EWpg_r', 'tipo': 'shunting'})
    # custom_config['connections'].append({'from': 'disturbance', 'to': 'EWpg_l', 'tipo': 'excitatory'})
    # custom_config['connections'].append({'from': 'disturbance', 'to': 'EWpg_r', 'tipo': 'excitatory'})
    # custom_config['connections'].append({'from': 'disturbance', 'to': 'EWpg_l', 'tipo': 'inhibitory'})
    # custom_config['connections'].append({'from': 'disturbance', 'to': 'EWpg_r', 'tipo': 'inhibitory'})
    custom_config['connections'].append({'from': 'disturbance', 'to': 'IML_l', 'tipo': 'inhibitory'})
    custom_config['connections'].append({'from': 'disturbance', 'to': 'IML_r', 'tipo': 'inhibitory'})

    system = ConfigurableJBS(custom_config)
    system.config('enable_bilateral_noise', True)
    system.config('enable_stochastic_noise', True)

    # deshabilita la historia para evitar sobrecarga de memoria
    system.enable_history = False

    n_steps = int(round(T_stabilize / system.dt))

    # system.define_input("disturbance", ['disturbance']) # Aplicación de ruido como componente externo
    system.define_input("disturbance", ['LC_l','LC_r']) # Modificación del nucleo en su componente alpha

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
    print(f"    epsilon_jitter = \033[93m{custom_config['default_jitter_epsilon']}\033[0m (variabilidad respuesta)")
    print(f"  \033[92mEstabilización:\033[0m \033[93m{T_stabilize}s\033[0m con L=\033[93m{L_background}\033[0m")
    print()
    
    return system

def run_simulation(system: ConfigurableJBS, LC_noise=0.0):
    """
    Ejecuta simulación completa con escalones de luminancia y ruido intencional adicional.
    
    Args:
        system: Sistema JBS configurado
        LC_noise: Nivel de ruido intencional [0-1]
        
    Returns:
        dict: Resultados completos de la simulación con todas las señales intermedias
    """
    t_total = max([step_time for step_time,_ in LUMINANCE_STEPS])
    dt = system.dt
    n_steps = int(t_total/dt)

    print(f"Ejecutando simulación de {t_total}s ({n_steps} pasos a dt={dt*1000:.1f}ms)...")
    print(f"Ruido bilateral: {system._config['enable_bilateral_noise']}, Ruido estocástico: {system._config['enable_stochastic_noise']}")
    print(f"Ruido intencional LC: {LC_noise:.3f}")
    
    # Pre-asignar arrays para velocidad — todos los núcleos del trayecto
    # parasimpático/simpático se muestrean lateralizados (estilo test_04)
    time_arr      = np.empty(n_steps)
    source_arr    = np.empty(n_steps)   # luminancia de fuente (lo que emite el ambiente)
    retinal_l_arr = np.empty(n_steps)   # entrada efectiva a retina izquierda
    retinal_r_arr = np.empty(n_steps)   # entrada efectiva a retina derecha
    lc_noise_arr  = np.empty(n_steps)   # ruido intencional LC
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
    
    # Ejecutar simulación
    for i in range(n_steps):
        t = system.t
        # Nivel de ruido a LC
        LC_noise_value = np.random.normal(0,LC_noise)

        # Determinar nivel de luminancia actual según escalones (forma simple)
        current_L = next((step_L for step_time, step_L in LUMINANCE_STEPS if t <= step_time), 0)
        
        # Ejecutar paso de simulación con ruido intencional
        system.step_simulation(L = current_L, inputs = {'disturbance':LC_noise_value})
        
        # Obtener estado completo
        state = system.view_state("all")
        
        # Muestrear estado actual (estilo test_04)
        time_arr[i]      = t
        source_arr[i]    = current_L
        lc_noise_arr[i]  = LC_noise_value
        retinal_l_arr[i] = state['luminance_left']['perceived']
        retinal_r_arr[i] = state['luminance_right']['perceived']
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
        'lc_noise': lc_noise_arr,  # Nueva señal de ruido intencional
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

def main():
    """Función principal del test."""
    print("=== Test A05: Ruido Intencional Adicional ===\n")
    
    # Crear sistema
    system = create_stabilized_system()
    
    # Configurar ruido intencional
    LC_noise_level = 2  # Nivel de ruido intencional
    print(f"Ruido intencional LC configurado: {LC_noise_level:.3f}")
    
    # Ejecutar simulación principal
    results = run_simulation(system, LC_noise=LC_noise_level)
    
    print("Simulación completada. Generando gráficos...")
    
    # ---------------------------------------------------------------------------
    # Generar gráficos (estilo test_04 completo)
    # ---------------------------------------------------------------------------
    
    fig = plt.figure(figsize=(13, 10))
    gs  = gridspec.GridSpec(4, 1, hspace=0.45)  # 4 paneles ahora
    
    # --- Panel 1: Luminancia de fuente vs entrada retinal efectiva ---
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
    for time, L_level in LUMINANCE_STEPS:
        ax1.axvline(time, color='gray', linestyle='--', alpha=0.4)
        ax1.text(time + 0.3, L_level + 0.03, f'{L_level:.2f}',
                 fontsize=9, color='saddlebrown')
    
    # --- Panel 2: Ruido intencional LC ---
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(results['time'], results['lc_noise'], color='purple', linewidth=2,
             label=f'Ruido intencional LC (nivel={LC_noise_level:.2f})')
    ax2.fill_between(results['time'], results['lc_noise'], step='post', alpha=0.15, color='purple')
    ax2.set_ylabel('Ruido Intencional', fontsize=11)
    ax2.set_title('Señal de Ruido Intencional Adicional', fontsize=12, fontweight='bold')
    # ax2.set_ylim(-0.1, 1.1)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # --- Panel 3: Señales intermedias (lateralizadas) ---
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(results['time'], results['pta_left'],   label='PTA L',  color='steelblue', linewidth=1.2)
    ax3.plot(results['time'], results['pta_right'],  label='PTA R',  color='steelblue', linewidth=1.2, linestyle=':')
    ax3.plot(results['time'], results['ewpg_left'],  label='EWpg L', color='royalblue', linewidth=1.2)
    ax3.plot(results['time'], results['ewpg_right'], label='EWpg R', color='royalblue', linewidth=1.2, linestyle=':')
    ax3.plot(results['time'], results['cg_left'],    label='CG L (esfínter)',    color='crimson',  linewidth=1.5)
    ax3.plot(results['time'], results['cg_right'],   label='CG R (esfínter)',    color='crimson',  linewidth=1.5, linestyle=':')
    ax3.plot(results['time'], results['scg_left'],   label='SCG L (dilatador)',  color='seagreen', linewidth=1.5)
    ax3.plot(results['time'], results['scg_right'],  label='SCG R (dilatador)',  color='seagreen', linewidth=1.5, linestyle=':')
    ax3.plot(results['time'], results['lc_left'],    label='LC (izq.)', color='darkviolet', linewidth=1.5)
    ax3.plot(results['time'], results['lc_right'],   label='LC (der.)', color='darkviolet', linewidth=1.5, linestyle=':')    
    ax3.set_ylabel('Activación de Núcleos', fontsize=11)
    ax3.set_title('Señales Intermedias (lateralizadas)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # --- Panel 4: Respuesta del diámetro pupilar (por ojo) ---
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(results['time'], results['pupil_left'], color='steelblue', linewidth=2,
             label='Pupila izquierda')
    ax4.plot(results['time'], results['pupil_right'], color='royalblue', linewidth=2, linestyle=':',
             label='Pupila derecha')
    ax4.set_xlabel('Tiempo (s)', fontsize=11)
    ax4.set_ylabel('Diámetro pupilar (mm)', fontsize=11)
    ax4.set_title('Respuesta del Diámetro Pupilar (por ojo)', fontsize=12, fontweight='bold')
    ax4.legend(loc='lower right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Título general (estilo test_04)
    plt.suptitle(
        f'Test A05: Ruido Intencional Adicional (LC={LC_noise_level:.2f})',
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    
    # Guardar gráfico
    plot_file = "data/test/A05_intentional_noise_response.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Gráfico guardado en: {plot_file}")
    print("=== Test A05 Completado ===")
    
    # ---------------------------------------------------------------------------
    # Ejecutar análisis comparativo con A04 (sin ruido intencional)
    # ---------------------------------------------------------------------------
    
    # Ejecutar simulación de referencia sin ruido intencional
    print("\nEjecutando simulación de referencia (sin ruido intencional)...")
    system_ref = create_stabilized_system()
    results_ref = run_simulation(system_ref, LC_noise=0.0)
    
    # Generar gráfico comparativo
    fig_comp = plt.figure(figsize=(13, 8))
    ax_comp = fig_comp.add_subplot(111)
    
    # Comparar diámetros pupilares
    ax_comp.plot(results['time'], results['pupil_left'], color='steelblue', linewidth=2,
                label=f'Con ruido LC ({LC_noise_level:.2f})', alpha=0.8)
    ax_comp.plot(results_ref['time'], results_ref['pupil_left'], color='gray', linewidth=2,
                label='Sin ruido LC (0.0)', alpha=0.8)
    ax_comp.set_xlabel('Tiempo (s)', fontsize=11)
    ax_comp.set_ylabel('Diámetro Pupilar Izquierdo (mm)', fontsize=11)
    ax_comp.set_title(f'Comparación: Pupila Izquierda (con vs sin ruido intencional)',
                  fontsize=12, fontweight='bold')
    ax_comp.legend(loc='upper right', fontsize=9)
    ax_comp.grid(True, alpha=0.3)
    
    plt.tight_layout()
    comp_file = "data/test/A05_intentional_noise_comparison.png"
    plt.savefig(comp_file, dpi=150, bbox_inches='tight')
    print(f"Gráfico comparativo guardado en: {comp_file}")

if __name__ == "__main__":
    main()
