"""
Test A03: Optimización de Parámetros Pupilar (D_MIN, D_MAX, D_REF)

Este test optimiza dinámicamente los parámetros del diámetro pupilar del sistema JBS
para mejorar el ajuste con la curva empírica de Stanley & Davies (1995).

Objetivo:
- Comparar curva JBS vs Stanley & Davies
- Optimizar D_MIN, D_MAX, D_REF dinámicamente
- Minimizar error cuadrático medio (RMSE)
- Visualizar evolución del ajuste

Sin entrenamiento: solo se ajustan parámetros geométricos del sistema.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
from scipy.optimize import minimize
import copy

# Agregar directorio padre al path para importar lib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.generadores.JohanssonBalkenius import ConfigurableJBS

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

DT = 0.002  # 2 ms
T_STABILIZE = 5.0  # 5 segundos para alcanzar estado estacionario
T_FINAL = 2.0  # 2 segundos adicionales para medir estado estacionario
T_TOTAL = T_STABILIZE + T_FINAL

# Puntos de luminancia para optimización
N_LUMINANCE_POINTS = 16  # Menos puntos para optimización más rápida
LUMINANCE_MIN = 1e-4  # 0.0001 cd/m²
LUMINANCE_MAX = 1e4   # 10000 cd/m²

# Generar puntos en escala logarítmica
luminance_points = np.logspace(np.log10(LUMINANCE_MIN), np.log10(LUMINANCE_MAX), N_LUMINANCE_POINTS)

# Área del estímulo (grados) - valor típico para experimentos PLR
STIMULUS_AREA_DEG2 = 25.4  # 25.4 grados

# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

def create_system_with_params(D_min, D_max, D_ref, L_gain=1.0, L_gamma=1.0):
    """Crea un sistema JBS con parámetros de diámetro y entrada lumínica personalizados."""
    system = ConfigurableJBS("default", D_min=D_min, D_max=D_max, D_ref=D_ref,
                             L_gain=L_gain, L_gamma=L_gamma)
    return system

def measure_steady_state_fast(system, L_level, T_stabilize=T_STABILIZE, T_measure=T_FINAL):
    """
    Versión rápida de medición de estado estacionario.
    
    Args:
        system: Sistema JBS
        L_level: Nivel de luminancia en candelas (cd/m²)
        T_stabilize: Tiempo de estabilización
        T_measure: Tiempo de medición del estado estacionario
        
    Returns:
        steady_diameter: Diámetro pupilar promedio en estado estacionario
    """
    n_steps = int(round((T_stabilize + T_measure) / DT))
    n_measure_steps = int(round(T_measure / DT))
    
    # Establecer baseline cortical
    system.set_cortical_baseline()
    
    # Obtener diámetro inicial
    d_left = system.get_output("pupil_left")
    d_right = system.get_output("pupil_right")
    
    diameter_history = []
    
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
        

        # Guardar solo últimos segundos para eficiencia
        if i >= n_steps - n_measure_steps:
            diameter_history.append((d_left + d_right) / 2.0)
    
    # Calcular valor estacionario como promedio del último segundo
    steady_diameter = np.mean(diameter_history)
    
    return steady_diameter

def calculate_rmse(jbs_diameters, stanley_diameters):
    """Calcula el error cuadrático medio entre dos curvas."""
    return np.sqrt(np.mean((jbs_diameters - stanley_diameters) ** 2))

def objective_function(params):
    """
    Función objetivo para optimización con targets específicos para mínima y máxima luminancia.

    Args:
        params: Array [D_min, D_max, D_ref, L_gain, L_gamma]

    Returns:
        combined_error: Error combinado (RMSE + targets de extremos + forma de curva)
    """
    D_min, D_max, D_ref, L_gain, L_gamma = params

    try:
        # Medir curva JBS con estos parámetros
        jbs_diameters = []
        for L_level in luminance_points:
            system = ConfigurableJBS(system_config="default",
                                     D_min=D_min, D_max=D_max, D_ref=D_ref,
                                     L_gain=L_gain, L_gamma=L_gamma)
            steady_diameter = measure_steady_state_fast(system, L_level)
            jbs_diameters.append(steady_diameter)

        # Calcular curva Stanley & Davies
        stanley_diameters = [ConfigurableJBS.stanley_davies_diameter(L, STIMULUS_AREA_DEG2)
                             for L in luminance_points]

        # Calcular RMSE general
        rmse = calculate_rmse(np.array(jbs_diameters), np.array(stanley_diameters))

        # TARGETS ESPECÍFICOS PARA MÍNIMA Y MÁXIMA LUMINANCIA
        # -----------------------------------------------------------------
        # Target para mínima luminancia (LUMINANCE_MIN): diámetro máximo
        target_min_diameter = stanley_diameters[0]   # Stanley-Davies en L_min
        actual_min_diameter = jbs_diameters[0]        # JBS en L_min
        min_luminance_error = abs(actual_min_diameter - target_min_diameter)

        # Target para máxima luminancia (LUMINANCE_MAX): diámetro mínimo
        target_max_diameter = stanley_diameters[-1]   # Stanley-Davies en L_max
        actual_max_diameter = jbs_diameters[-1]        # JBS en L_max
        max_luminance_error = abs(actual_max_diameter - target_max_diameter)

        # Error combinado para extremos
        extreme_error = min_luminance_error + max_luminance_error

        # Calcular ajuste de pendiente (derivada)
        if len(jbs_diameters) > 1:
            jbs_derivatives = np.diff(jbs_diameters) / np.diff(luminance_points)
            stanley_derivatives = np.diff(stanley_diameters) / np.diff(luminance_points)
            derivative_error = np.mean((jbs_derivatives - stanley_derivatives) ** 2)
        else:
            derivative_error = 0.0

        # COMBINACIÓN DE ERRORES CON PESOS AJUSTABLES
        # -----------------------------------------------
        weight_extremes   = 2.0   # Peso alto para errores en extremos
        weight_rmse       = 3.0   # Peso estándar para RMSE general
        weight_derivative = 0.5   # Peso para forma de la curva

        combined_error = (weight_rmse       * rmse +
                          weight_extremes   * extreme_error +
                          weight_derivative * derivative_error)

        print(f"\n  Params: D_min=\033[94m{D_min:.2f}\033[0m, D_max=\033[94m{D_max:.2f}\033[0m, D_ref=\033[94m{D_ref:.2f}\033[0m, "
              f"L_gain=\033[94m{L_gain:.4f}\033[0m, L_gamma=\033[94m{L_gamma:.4f}\033[0m")
        print(f"    RMSE=\033[91m{rmse:.4f}\033[0m, Extremos=\033[91m{extreme_error:.4f}\033[0m, Deriv=\033[91m{derivative_error:.4f}\033[0m")
        print(f"    TargetMin=\033[94m{target_min_diameter:.2f}\033[0m→\033[92m{actual_min_diameter:.2f}\033[0m (err=\033[91m{min_luminance_error:.4f}\033[0m)")
        print(f"    TargetMax=\033[94m{target_max_diameter:.2f}\033[0m→\033[92m{actual_max_diameter:.2f}\033[0m (err=\033[91m{max_luminance_error:.4f}\033[0m)")
        print(f"    Combined=\033[91m{combined_error:.4f}\033[0m")

        return combined_error

    except Exception as e:
        print(f"Error en optimización: {e}")
        return 1e6

# ---------------------------------------------------------------------------
# Ejecución principal
# ---------------------------------------------------------------------------

def main():
    """Función principal del test."""
    print("=== Test A03: Optimización de Parámetros Pupilar ===\n")
    
    # Valores iniciales - SEMILLA ÓPTIMA DE OPTIMIZACIÓN ANTERIOR
    # -----------------------------------------------------------------
    # Usando los parámetros óptimos encontrados en la optimización anterior
    # para verificar convergencia al mismo punto.
    #
    # Parámetros óptimos anteriores:
    # D_MIN  = 2.44 mm
    # D_MAX  = 7.16 mm  
    # D_REF  = 5.90 mm
    # L_gain = 1.1358
    # L_gamma= 0.5848
    
    # SEMILLA ÓPTIMA (comentar para usar valores originales)
    initial_params = [2.44, 7.16, 5.90, 1.1358, 0.5848]
    system = ConfigurableJBS(
        system_config = "default",
        D_min   = initial_params[0],
        D_max   = initial_params[1],
        D_ref   = initial_params[2],
        L_gain  = initial_params[3],
        L_gamma = initial_params[4])
    # VALORES ORIGINALES (descomentar para usar valores del sistema)
    # system = ConfigurableJBS()
    # initial_params = [system.D_min, system.D_max, system.D_ref,
    #                   system.L_gain, system.L_gamma]
    
    print(f"Parámetros iniciales (semilla):")
    print(f"  D_MIN  = {initial_params[0]:.2f} mm")
    print(f"  D_MAX  = {initial_params[1]:.2f} mm")
    print(f"  D_REF  = {initial_params[2]:.2f} mm")
    print(f"  L_gain = {initial_params[3]:.4f}")
    print(f"  L_gamma= {initial_params[4]:.4f}")
    print(f"  Puntos de luminancia: {N_LUMINANCE_POINTS}")
    print(f"  Rango: {LUMINANCE_MIN:.1e} - {LUMINANCE_MAX:.1e} cd/m²")
    print(f"  Área del estímulo: {STIMULUS_AREA_DEG2:.1f} grados²\n")
    
    # Calcular curva inicial
    print("Calculando curva inicial...")
    initial_jbs_diameters = []
    initial_stanley_diameters = []
    
    for L_level in luminance_points:
        steady_diameter = measure_steady_state_fast(system, L_level)
        initial_jbs_diameters.append(steady_diameter)
        
        stanley_diam = ConfigurableJBS.stanley_davies_diameter(L_level, STIMULUS_AREA_DEG2)
        initial_stanley_diameters.append(stanley_diam)
    
    initial_rmse = calculate_rmse(np.array(initial_jbs_diameters), np.array(initial_stanley_diameters))
    print(f"RMSE inicial: {initial_rmse:.4f} mm\n")
    
    # Optimización
    print("Iniciando optimización...")
    
    # Límites para los parámetros
    bounds = [
        (-5.0, 5.0),    # D_min (mm)
        (3.0, 28.0),    # D_max (mm)
        (3.0, 16.0),    # D_ref (mm)
        (0.01, 100.0),  # L_gain: ganancia de entrada lumínica
        (0.1, 2.0),     # L_gamma: exponente de ley de potencia (0.1→fuerte compresión, 2→expansión)
    ]

    # Optimización usando método Nelder-Mead (robusto)
    result = minimize(objective_function, initial_params, method='Nelder-Mead',
                      bounds=bounds, options={'maxiter': 200, 'disp': True})

    optimal_params = result.x
    optimal_rmse = result.fun

    print(f"\nOptimización completada:")
    print(f"  D_MIN  óptimo = {optimal_params[0]:.2f} mm")
    print(f"  D_MAX  óptimo = {optimal_params[1]:.2f} mm")
    print(f"  D_REF  óptimo = {optimal_params[2]:.2f} mm")
    print(f"  L_gain óptimo = {optimal_params[3]:.4f}")
    print(f"  L_gamma óptimo= {optimal_params[4]:.4f}")
    print(f"  RMSE final = {optimal_rmse:.4f} mm")
    print(f"  Mejora = {((initial_rmse - optimal_rmse) / initial_rmse * 100):.1f}%\n")
    
    # Calcular curvas finales
    print("Calculando curvas optimizadas...")
    final_jbs_diameters = []
    final_stanley_diameters = []

    for L_level in luminance_points:
        system = ConfigurableJBS(system_config="default",
                                 D_min=optimal_params[0], D_max=optimal_params[1],
                                 D_ref=optimal_params[2],
                                 L_gain=optimal_params[3], L_gamma=optimal_params[4])
        steady_diameter = measure_steady_state_fast(system, L_level)
        final_jbs_diameters.append(steady_diameter)

        stanley_diam = ConfigurableJBS.stanley_davies_diameter(L_level, STIMULUS_AREA_DEG2)
        final_stanley_diameters.append(stanley_diam)
    
    # ---------------------------------------------------------------------------
    # Generar gráficos
    # ---------------------------------------------------------------------------
    
    print("Generando gráficos...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # --- Panel 1: Comparación de curvas ---
    ax1.semilogx(luminance_points, initial_stanley_diameters, 'k-', 
                  linewidth=3, label='Stanley & Davies (1995)', alpha=0.8)
    ax1.semilogx(luminance_points, initial_jbs_diameters, 'b--', 
                  linewidth=2, label=f'JBS Inicial (RMSE={initial_rmse:.3f})', alpha=0.7)
    ax1.semilogx(luminance_points, final_jbs_diameters, 'r-', 
                  linewidth=2.5, label=f'JBS Optimizado (RMSE={optimal_rmse:.3f})')
    
    ax1.set_xlabel('Luminancia (cd/m²)', fontsize=12)
    ax1.set_ylabel('Diámetro Pupilar (mm)', fontsize=12)
    ax1.set_title('Optimización de Parámetros Pupilar', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xlim(LUMINANCE_MIN, LUMINANCE_MAX)
    
    # --- Panel 2: Error residual ---
    residual_initial = np.array(initial_jbs_diameters) - np.array(initial_stanley_diameters)
    residual_final = np.array(final_jbs_diameters) - np.array(final_stanley_diameters)
    
    ax2.semilogx(luminance_points, residual_initial, 'b--', 
                  linewidth=2, label=f'Error Inicial (RMSE={initial_rmse:.3f})', alpha=0.7)
    ax2.semilogx(luminance_points, residual_final, 'r-', 
                  linewidth=2, label=f'Error Final (RMSE={optimal_rmse:.3f})')
    ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Luminancia (cd/m²)', fontsize=12)
    ax2.set_ylabel('Error (JBS - Stanley) [mm]', fontsize=12)
    ax2.set_title('Error Residual', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim(LUMINANCE_MIN, LUMINANCE_MAX)
    
    # --- Panel 3: Comparación de parámetros ---
    param_names = ['D_MIN', 'D_MAX', 'D_REF', 'L_gain', 'L_gamma']
    initial_values = initial_params
    optimal_values = optimal_params
    
    x = np.arange(len(param_names))
    width = 0.35
    
    ax3.bar(x - width/2, initial_values, width, label='Inicial', 
             color='blue', alpha=0.7)
    ax3.bar(x + width/2, optimal_values, width, label='Óptimo', 
             color='red', alpha=0.7)
    
    ax3.set_xlabel('Parámetro', fontsize=12)
    ax3.set_ylabel('Valor', fontsize=12)
    ax3.set_title('Comparación de Parámetros', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(param_names)
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores en las barras
    for i, (init, opt) in enumerate(zip(initial_values, optimal_values)):
        # Formato especial para L_gain y L_gamma
        if param_names[i] in ['L_gain', 'L_gamma']:
            init_text = f'{init:.3f}'
            opt_text = f'{opt:.3f}'
        else:
            init_text = f'{init:.2f}'
            opt_text = f'{opt:.2f}'
        
        ax3.text(i - width/2, init + 0.1, init_text, 
                 ha='center', va='bottom', fontsize=9)
        ax3.text(i + width/2, opt + 0.1, opt_text, 
                 ha='center', va='bottom', fontsize=9)
    
    # --- Panel 4: Información del experimento ---
    ax4.axis('off')
    
    info_text = f"""Resultados de Optimización:

Parámetros Iniciales:
  D_MIN   = {initial_params[0]:.2f} mm
  D_MAX   = {initial_params[1]:.2f} mm
  D_REF   = {initial_params[2]:.2f} mm
  L_gain  = {initial_params[3]:.4f}
  L_gamma = {initial_params[4]:.4f}
  RMSE    = {initial_rmse:.4f} mm

Parámetros Óptimos:
  D_MIN   = {optimal_params[0]:.2f} mm
  D_MAX   = {optimal_params[1]:.2f} mm
  D_REF   = {optimal_params[2]:.2f} mm
  L_gain  = {optimal_params[3]:.4f}
  L_gamma = {optimal_params[4]:.4f}
  RMSE    = {optimal_rmse:.4f} mm

Mejora: {((initial_rmse - optimal_rmse) / initial_rmse * 100):.1f}%

Configuración:
  Puntos de luminancia: {N_LUMINANCE_POINTS}
  Rango: {LUMINANCE_MIN:.1e} - {LUMINANCE_MAX:.1e} cd/m²
  Área estímulo: {STIMULUS_AREA_DEG2:.1f} grados²
  Método: Nelder-Mead
  Iteraciones máx: 20
"""
    
    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Guardar gráfico
    plot_file = "../data/test/A03_optimization_results.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Gráfico guardado en: {plot_file}")
    
    # Guardar resultados
    results_file = "../data/test/A03_optimization_results.txt"
    with open(results_file, 'w') as f:
        f.write("# Test A03: Optimización de Parámetros Pupilar\n")
        f.write("# Sistema JBS vs Stanley & Davies (1995)\n")
        f.write(f"# Área estímulo: {STIMULUS_AREA_DEG2:.1f} grados²\n")
        f.write(f"# Método: Nelder-Mead, max_iter=20\n\n")
        
        f.write("# Parámetros iniciales:\n")
        f.write(f"D_MIN   {initial_params[0]:.6f}\n")
        f.write(f"D_MAX   {initial_params[1]:.6f}\n")
        f.write(f"D_REF   {initial_params[2]:.6f}\n")
        f.write(f"L_gain  {initial_params[3]:.6f}\n")
        f.write(f"L_gamma {initial_params[4]:.6f}\n")
        f.write(f"RMSE    {initial_rmse:.6f}\n\n")

        f.write("# Parámetros óptimos:\n")
        f.write(f"D_MIN   {optimal_params[0]:.6f}\n")
        f.write(f"D_MAX   {optimal_params[1]:.6f}\n")
        f.write(f"D_REF   {optimal_params[2]:.6f}\n")
        f.write(f"L_gain  {optimal_params[3]:.6f}\n")
        f.write(f"L_gamma {optimal_params[4]:.6f}\n")
        f.write(f"RMSE    {optimal_rmse:.6f}\n")
        f.write(f"MEJORA  {((initial_rmse - optimal_rmse) / initial_rmse * 100):.6f}\n\n")
        
        f.write("# Datos de curvas:\n")
        f.write("# Luminancia(cd/m2)\tStanley(mm)\tJBS_Inicial(mm)\tJBS_Optimizado(mm)\tError_Inicial(mm)\tError_Final(mm)\n")
        for i, L in enumerate(luminance_points):
            f.write(f"{L:.6e}\t{initial_stanley_diameters[i]:.6f}\t")
            f.write(f"{initial_jbs_diameters[i]:.6f}\t{final_jbs_diameters[i]:.6f}\t")
            f.write(f"{residual_initial[i]:.6f}\t{residual_final[i]:.6f}\n")
    
    print(f"Resultados guardados en: {results_file}")
    
    plt.show()
    
    print("\n=== Test A03 Completado ===")

if __name__ == "__main__":
    main()
