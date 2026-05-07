"""
Test A02: Curva de Luminancia vs Diámetro Pupilar (Sin Entrenamiento)

Este test obtiene la curva característica del sistema JBS sin entrenamiento,
midiendo el diámetro pupilar en estado estacionario para diferentes
niveles de luminancia.

Objetivo:
- Obtener curva luminancia vs diámetro pupilar
- Calcular valor estacionario como promedio del último segundo
- Presentar resultados en escala semilogarítmica para luminancia
- Usar 10-20 puntos de luminancia para buena resolución

Sin entrenamiento: solo se usa la configuración por defecto del sistema.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os

# Agregar directorio padre al path para importar lib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.generadores.JohanssonBalkenius import ConfigurableJBS

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

DT = 0.002  # 2 ms
T_STABILIZE = 15.0  # 15 segundos para alcanzar estado estacionario
T_FINAL = 5.0  # 5 segundos adicionales para medir estado estacionario
T_TOTAL = T_STABILIZE + T_FINAL

N_STEPS = int(round(T_TOTAL / DT))

# Puntos de luminancia (escala logarítmica)
# 15 puntos desde 1e-4 hasta 1e4 cd/m² (escala log)
N_LUMINANCE_POINTS = 15
LUMINANCE_MIN = 1e-4  # 0.0001 cd/m²
LUMINANCE_MAX = 1e4   # 10000 cd/m²

# Generar puntos en escala logarítmica
luminance_points = np.logspace(np.log10(LUMINANCE_MIN), np.log10(LUMINANCE_MAX), N_LUMINANCE_POINTS)

# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------
def measure_steady_state(system, L_level, T_stabilize=T_STABILIZE, T_measure=T_FINAL):
    """
    Mide el diámetro pupilar en estado estacionario para un nivel de luminancia.
    
    Args:
        system: Sistema JBS
        L_level: Nivel de luminancia en candelas (cd/m²)
        T_stabilize: Tiempo de estabilización
        T_measure: Tiempo de medición del estado estacionario
        
    Returns:
        steady_diameter: Diámetro pupilar promedio en estado estacionario
        time_history: Vector de tiempos completos
        diameter_history: Vector de diámetros pupilar completos
    """
    n_steps = int(round((T_stabilize + T_measure) / DT))
    n_measure_steps = int(round(T_measure / DT))
    
    time_history = np.empty(n_steps)
    diameter_history = np.empty(n_steps)
    
    # Establecer baseline cortical
    system.set_cortical_baseline()
    
    # Obtener diámetro inicial
    d_left = system.pupil_diameter(0.0, 0.0)
    d_right = system.pupil_diameter(0.0, 0.0)
    
    for i in range(n_steps):
        t = system.t
        
        # Entrada óptica constante usando candelas
        alpha_left = system.candel_input(L_level, d_left)
        alpha_right = system.candel_input(L_level, d_right)
        system.set_input("retina_left", alpha_left)
        system.set_input("retina_right", alpha_right)
        
        # Paso de simulación
        system.step()
        
        # Obtener salidas
        d_left = system.get_output("pupil_left")
        d_right = system.get_output("pupil_right")
        
        time_history[i] = t
        diameter_history[i] = (d_left + d_right) / 2.0  # Promedio ambos ojos
    
    # Calcular valor estacionario como promedio del último segundo
    steady_diameter = np.mean(diameter_history[-n_measure_steps:])
    
    return steady_diameter, time_history, diameter_history

# ---------------------------------------------------------------------------
# Ejecución principal
# ---------------------------------------------------------------------------

def main():
    """Función principal del test."""
    print("=== Test A02: Curva de Luminancia vs Diámetro Pupilar ===\n")
    
    print(f"Sistema creado sin entrenamiento")
    print(f"Configuración: {N_LUMINANCE_POINTS} puntos de luminancia")
    print(f"Rango: {LUMINANCE_MIN:.1e} - {LUMINANCE_MAX:.1f} cd/m²")
    print(f"Tiempo estabilización: {T_STABILIZE}s, Tiempo medición: {T_FINAL}s\n")
    
    # Medir estado estacionario para cada nivel de luminancia
    steady_diameters = []
    all_time_histories = []
    all_diameter_histories = []
    projected_diameter = []
    
    print("Midiendo estados estacionarios...")
    for i, L_level in enumerate(luminance_points):
        print(f"  [{i+1:2d}/{N_LUMINANCE_POINTS}] L = {L_level:.2e} cd/m²")
        
        # Crear copia fresca del sistema para cada medición con parámetros por defecto
        system = ConfigurableJBS()
        steady_diameter, time_hist, diam_hist = measure_steady_state(system, L_level)
        
        steady_diameters.append(steady_diameter)
        all_time_histories.append(time_hist)
        all_diameter_histories.append(diam_hist)
        projected_diameter.append(ConfigurableJBS.stanley_davies_diameter(L_level,25.4))    
    print("\nMedición completada.")
    
    # ---------------------------------------------------------------------------
    # Generar gráficos
    # ---------------------------------------------------------------------------
    
    print("Generando gráficos...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # --- Panel 1: Curva principal Luminancia vs Diámetro ---
    ax1.semilogx(luminance_points, steady_diameters, 'o-', 
                  color='darkblue', linewidth=2, markersize=6, 
                  markerfacecolor='white', markeredgewidth=1.5)
    ax1.semilogx(luminance_points, projected_diameter, 's-', 
                  color='darkred', linewidth=2, markersize=6, 
                  markerfacecolor='white', markeredgewidth=1.5)
    ax1.set_xlabel('Luminancia (cd/m²)', fontsize=12)
    ax1.set_ylabel('Diámetro Pupilar (mm)', fontsize=12)
    ax1.set_title('Curva Característica: Luminancia vs Diámetro Pupilar', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(['Simulación', 'Modelo Stanley-Davies'])
    ax1.set_xlim(LUMINANCE_MIN, LUMINANCE_MAX)
    
    # Agregar valores en puntos clave
    for i, (L, d) in enumerate(zip(luminance_points, steady_diameters)):
        if i % 3 == 0:  # Mostrar cada 3er punto para no saturar
            ax1.annotate(f'{d:.2f}mm', (L, d), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.7)
    
    # --- Panel 2: Series temporales seleccionadas ---
    # Mostrar 3 series temporales representativas
    indices_to_show = [0, N_LUMINANCE_POINTS//2, -1]  # Primera, media, última
    
    colors = ['lightblue', 'orange', 'red']
    labels = ['Mínima', 'Media', 'Máxima']
    
    for i, idx in enumerate(indices_to_show):
        if idx < len(all_time_histories):
            color = colors[i]
            label = labels[i]
            time_rel = all_time_histories[idx] - T_STABILIZE
            diam = all_diameter_histories[idx]
            
            ax2.plot(time_rel, diam, color=color, linewidth=1.5, 
                    alpha=0.8, label=f'{label} (L={luminance_points[idx]:.2e})')
    
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5, label='Inicio medición')
    ax2.set_xlabel('Tiempo (s)', fontsize=12)
    ax2.set_ylabel('Diámetro Pupilar (mm)', fontsize=12)
    ax2.set_title('Evolución Temporal al Estado Estacionario', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-1, T_FINAL)
    
    # --- Panel 3: Derivada (sensibilidad) ---
    # Calcular sensibilidad como derivada numérica
    log_L = np.log10(luminance_points)
    d_diam_d_logL = np.gradient(steady_diameters, log_L)
    projected_diam_d_logL = np.gradient(projected_diameter, log_L)
    
    ax3.semilogx(luminance_points, d_diam_d_logL, 's-', 
                  color='darkgreen', linewidth=2)
    ax3.semilogx(luminance_points, projected_diam_d_logL, 'o-', 
                  color='red', linewidth=2, alpha=0.5)
    ax3.set_xlabel('Luminancia (cd/m²)', fontsize=12)
    ax3.set_ylabel('d(diam)/d(log L) (mm/dec)', fontsize=12)
    ax3.set_title('Sensibilidad del Sistema', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.set_xlim(LUMINANCE_MIN, LUMINANCE_MAX)
    ax3.legend(['Simulación', 'Modelo Stanley-Davies'])
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # --- Panel 4: Información del experimento ---
    ax4.axis('off')
    
    info_text = f"""Parámetros del Experimento:
    
Sistema: Johansson-Balkenius (sin entrenamiento)
Configuración: Parámetros óptimos
Puntos de luminancia: \033[91m{N_LUMINANCE_POINTS}\033[0m
Rango: \033[91m{LUMINANCE_MIN:.1e} - {LUMINANCE_MAX:.1f}\033[0m cd/m²

Tiempo estabilización: \033[91m{T_STABILIZE}s\033[0m
Tiempo medición: \033[91m{T_FINAL}s\033[0m
Paso temporal: \033[91m{DT*1000:.1f} ms\033[0m

Resultados clave:
Diámetro mínimo: \033[91m{min(steady_diameters):.2f} mm\033[0m
Diámetro máximo: \033[91m{max(steady_diameters):.2f} mm\033[0m
Rango dinámico: \033[91m{max(steady_diameters) - min(steady_diameters):.2f} mm\033[0m

Punto de máxima sensibilidad:
L ≈ {luminance_points[np.argmax(np.abs(d_diam_d_logL))]:.2e} cd/m²
Sensibilidad máxima: {np.max(np.abs(d_diam_d_logL)):.3f} mm/dec
"""
    
    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Guardar gráfico
    plot_file = "data/test/A02_luminance_vs_diameter.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Gráfico guardado en: {plot_file}")

    
    plt.show()
    
    print("\n=== Test A02 Completado ===")

if __name__ == "__main__":
    main()
