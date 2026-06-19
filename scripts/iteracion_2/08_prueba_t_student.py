"""
Script 08 - Prueba t de Student para PUI
==========================================
Realiza una prueba t de Student para comparar las medias del biomarcador
PUI entre los grupos Control y Migraña Vestibular en la iteración 2.

Objetivo:
    Determinar si existen diferencias estadísticamente significativas en
    el biomarcador PUI entre sujetos con Migraña Vestibular y sujetos
    Control.

Entradas:
    - data/processed/analisis_resultados_it2.csv  (biomarcadores por sujeto)

Salidas (data/test/):
    - prueba_t_pui_it2.txt  : reporte completo de la prueba t de Student
    - prueba_t_pui_it2.png  : gráfico de distribución de PUI por grupo

Ejecución:
    python scripts/iteracion_2/08_prueba_t_student.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # backend no interactivo para guardar figuras sin pantalla
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple

# ---------------------------------------------------------------------------
# Constantes de configuración
# ---------------------------------------------------------------------------
PATH_DATOS: str = 'data/processed/analisis_final.csv'
PATH_SALIDA: str = 'data/test/'

BIOMARCADOR: str = 'PUI'
ALPHA: float = 0.05  # Nivel de significancia

ETIQUETA_POSITIVA: str = 'Migraña Vestibular'
ETIQUETA_NEGATIVA: str = 'Control'

DPI_FIGURA: int = 150


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
def cargar_datos(path_datos: str) -> pd.DataFrame:
    """
    Carga el dataset procesado de biomarcadores.

    Args:
        path_datos: Ruta al CSV con biomarcadores extraídos por sujeto.

    Returns:
        DataFrame con todos los biomarcadores y diagnóstico.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    if not os.path.exists(path_datos):
        raise FileNotFoundError(f"Dataset no encontrado: {path_datos}")

    df = pd.read_csv(path_datos)
    df.columns = df.columns.str.strip()
    return df


# ---------------------------------------------------------------------------
# Prueba t de Student
# ---------------------------------------------------------------------------
def realizar_prueba_t(
    df: pd.DataFrame,
    biomarcador: str,
    grupo_positivo: str,
    grupo_negativo: str
) -> Tuple[dict, dict, dict]:
    """
    Realiza la prueba t de Student para comparar las medias de un biomarcador
    entre dos grupos.

    Args:
        df: DataFrame con biomarcadores y diagnóstico.
        biomarcador: Nombre del biomarcador a analizar.
        grupo_positivo: Nombre del grupo positivo (Migraña Vestibular).
        grupo_negativo: Nombre del grupo negativo (Control).

    Returns:
        Tuple con:
            - estadisticas_grupo_positivo: dict con media, std, n del grupo positivo.
            - estadisticas_grupo_negativo: dict con media, std, n del grupo negativo.
            - resultados_prueba_t: dict con estadístico t, p-valor, conclusión.
    """
    # Separar datos por grupo
    datos_positivo = df[df['Diagnostico'] == grupo_positivo][biomarcador]
    datos_negativo = df[df['Diagnostico'] == grupo_negativo][biomarcador]

    # Calcular estadísticas descriptivas
    estadisticas_positivo = {
        'media': datos_positivo.mean(),
        'std': datos_positivo.std(),
        'n': len(datos_positivo),
        'min': datos_positivo.min(),
        'max': datos_positivo.max(),
        'mediana': datos_positivo.median()
    }

    estadisticas_negativo = {
        'media': datos_negativo.mean(),
        'std': datos_negativo.std(),
        'n': len(datos_negativo),
        'min': datos_negativo.min(),
        'max': datos_negativo.max(),
        'mediana': datos_negativo.median()
    }

    # Realizar prueba t de Student (dos colas, asumiendo varianzas diferentes)
    t_statistic, p_value = stats.ttest_ind(datos_positivo, datos_negativo, equal_var=False)

    # Determinar significancia
    es_significativo = p_value < ALPHA
    conclusion = (
        f"La diferencia es ESTADÍSTICAMENTE SIGNIFICATIVA (p < {ALPHA})"
        if es_significativo
        else f"La diferencia NO es estadísticamente significativa (p >= {ALPHA})"
    )

    resultados_prueba_t = {
        'estadistico_t': t_statistic,
        'p_valor': p_value,
        'alpha': ALPHA,
        'es_significativo': es_significativo,
        'conclusion': conclusion
    }

    return estadisticas_positivo, estadisticas_negativo, resultados_prueba_t


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------
def guardar_grafico_distribucion(
    df: pd.DataFrame,
    biomarcador: str,
    grupo_positivo: str,
    grupo_negativo: str,
    estadisticas_positivo: dict,
    estadisticas_negativo: dict,
    resultados_prueba_t: dict,
    path_salida: str
) -> None:
    """
    Genera y guarda un gráfico de distribución del biomarcador por grupo.

    Args:
        df: DataFrame con biomarcadores y diagnóstico.
        biomarcador: Nombre del biomarcador a graficar.
        grupo_positivo: Nombre del grupo positivo.
        grupo_negativo: Nombre del grupo negativo.
        estadisticas_positivo: Estadísticas del grupo positivo.
        estadisticas_negativo: Estadísticas del grupo negativo.
        resultados_prueba_t: Resultados de la prueba t.
        path_salida: Directorio donde se guarda el archivo PNG.
    """
    datos_positivo = df[df['Diagnostico'] == grupo_positivo][biomarcador]
    datos_negativo = df[df['Diagnostico'] == grupo_negativo][biomarcador]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histograma
    ax1.hist(datos_negativo, alpha=0.7, label=grupo_negativo, bins=30, color='blue', edgecolor='black')
    ax1.hist(datos_positivo, alpha=0.7, label=grupo_positivo, bins=30, color='red', edgecolor='black')
    ax1.axvline(estadisticas_negativo['media'], color='blue', linestyle='--', linewidth=2, 
                label=f'Media {grupo_negativo}: {estadisticas_negativo["media"]:.4f}')
    ax1.axvline(estadisticas_positivo['media'], color='red', linestyle='--', linewidth=2,
                label=f'Media {grupo_positivo}: {estadisticas_positivo["media"]:.4f}')
    ax1.set_xlabel(biomarcador, fontsize=12)
    ax1.set_ylabel('Frecuencia', fontsize=12)
    ax1.set_title(f'Distribución de {biomarcador} por Grupo', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Boxplot
    box_data = [datos_negativo, datos_positivo]
    bp = ax2.boxplot(box_data, labels=[grupo_negativo, grupo_positivo], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax2.set_ylabel(biomarcador, fontsize=12)
    ax2.set_title(f'Boxplot de {biomarcador} por Grupo', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')

    # Título general con resultados de prueba t
    significancia_texto = "SIGNIFICATIVA" if resultados_prueba_t['es_significativo'] else "NO SIGNIFICATIVA"
    fig.suptitle(
        f'Prueba t de Student: {biomarcador}\n'
        f't = {resultados_prueba_t["estadistico_t"]:.4f}, p = {resultados_prueba_t["p_valor"]:.6f} → {significancia_texto}',
        fontsize=16, fontweight='bold', y=0.98
    )

    plt.tight_layout()
    fig.savefig(
        os.path.join(path_salida, 'prueba_t_pui_it2.png'),
        dpi=DPI_FIGURA, bbox_inches='tight'
    )
    plt.close(fig)
    print("  [OK] prueba_t_pui_it2.png")


# ---------------------------------------------------------------------------
# Reporte de texto
# ---------------------------------------------------------------------------
def guardar_reporte_texto(
    biomarcador: str,
    grupo_positivo: str,
    grupo_negativo: str,
    estadisticas_positivo: dict,
    estadisticas_negativo: dict,
    resultados_prueba_t: dict,
    path_salida: str
) -> None:
    """
    Guarda un reporte detallado de la prueba t de Student en texto plano.

    Args:
        biomarcador: Nombre del biomarcador analizado.
        grupo_positivo: Nombre del grupo positivo.
        grupo_negativo: Nombre del grupo negativo.
        estadisticas_positivo: Estadísticas del grupo positivo.
        estadisticas_negativo: Estadísticas del grupo negativo.
        resultados_prueba_t: Resultados de la prueba t.
        path_salida: Directorio donde se guarda el archivo TXT.
    """
    lineas = [
        "=" * 70,
        f"PRUEBA T DE STUDENT - {biomarcador} | Iteración 2",
        "=" * 70,
        "",
        f"Grupos comparados:",
        f"  - Grupo Positivo: {grupo_positivo}",
        f"  - Grupo Negativo: {grupo_negativo}",
        f"  - Nivel de significancia (α): {ALPHA}",
        "",
        "-" * 70,
        f"ESTADÍSTICAS DESCRIPTIVAS - {biomarcador}",
        "-" * 70,
        "",
        f"{grupo_positivo} (n = {estadisticas_positivo['n']}):",
        f"  Media:          {estadisticas_positivo['media']:.6f}",
        f"  Desviación SD: {estadisticas_positivo['std']:.6f}",
        f"  Mediana:        {estadisticas_positivo['mediana']:.6f}",
        f"  Mínimo:         {estadisticas_positivo['min']:.6f}",
        f"  Máximo:         {estadisticas_positivo['max']:.6f}",
        "",
        f"{grupo_negativo} (n = {estadisticas_negativo['n']}):",
        f"  Media:          {estadisticas_negativo['media']:.6f}",
        f"  Desviación SD: {estadisticas_negativo['std']:.6f}",
        f"  Mediana:        {estadisticas_negativo['mediana']:.6f}",
        f"  Mínimo:         {estadisticas_negativo['min']:.6f}",
        f"  Máximo:         {estadisticas_negativo['max']:.6f}",
        "",
        "-" * 70,
        "RESULTADOS DE LA PRUEBA T DE STUDENT",
        "-" * 70,
        "",
        f"Estadístico t:  {resultados_prueba_t['estadistico_t']:.6f}",
        f"Valor p:        {resultados_prueba_t['p_valor']:.6f}",
        f"Alpha:          {resultados_prueba_t['alpha']:.4f}",
        "",
        f"Diferencia de medias: {estadisticas_positivo['media'] - estadisticas_negativo['media']:.6f}",
        f"Porcentaje de cambio:  {100 * (estadisticas_positivo['media'] - estadisticas_negativo['media']) / estadisticas_negativo['media']:.2f}%",
        "",
        "-" * 70,
        "CONCLUSIÓN",
        "-" * 70,
        "",
        resultados_prueba_t['conclusion'],
        "",
        "Interpretación:",
        "  - p < α: Rechazamos la hipótesis nula (las medias son diferentes)",
        "  - p ≥ α: No rechazamos la hipótesis nula (no hay evidencia de diferencia)",
        "",
        "=" * 70,
    ]

    ruta_txt = os.path.join(path_salida, 'prueba_t_pui_it2.txt')
    with open(ruta_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lineas) + '\n')

    print("  [OK] prueba_t_pui_it2.txt")
    print('\n'.join(lineas))


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Ejecuta el pipeline completo de prueba t de Student para PUI.

    Flujo:
        1. Carga los datos de biomarcadores.
        2. Separa los datos por grupo diagnóstico.
        3. Realiza la prueba t de Student para PUI.
        4. Genera gráfico de distribución.
        5. Guarda reporte detallado en texto.
    """
    os.makedirs(PATH_SALIDA, exist_ok=True)

    print("\n=== Script 08 — Prueba t de Student para PUI | Iteración 2 ===\n")

    # 1. Datos
    df = cargar_datos(PATH_DATOS)
    print(f"Dataset cargado: {len(df)} sujetos")
    print(f"Distribución por diagnóstico:")
    print(df['Diagnostico'].value_counts())
    print()

    # 2. Prueba t
    print(f"Realizando prueba t de Student para {BIOMARCADOR}...")
    estadisticas_positivo, estadisticas_negativo, resultados_prueba_t = realizar_prueba_t(
        df, BIOMARCADOR, ETIQUETA_POSITIVA, ETIQUETA_NEGATIVA
    )

    print(f"\nResultados preliminares:")
    print(f"  t = {resultados_prueba_t['estadistico_t']:.6f}")
    print(f"  p = {resultados_prueba_t['p_valor']:.6f}")
    print(f"  {resultados_prueba_t['conclusion']}")
    print()

    # 3. Gráfico
    print("Generando gráfico de distribución...")
    guardar_grafico_distribucion(
        df, BIOMARCADOR, ETIQUETA_POSITIVA, ETIQUETA_NEGATIVA,
        estadisticas_positivo, estadisticas_negativo, resultados_prueba_t, PATH_SALIDA
    )

    # 4. Reporte
    print("\nGenerando reporte detallado...")
    guardar_reporte_texto(
        BIOMARCADOR, ETIQUETA_POSITIVA, ETIQUETA_NEGATIVA,
        estadisticas_positivo, estadisticas_negativo, resultados_prueba_t, PATH_SALIDA
    )

    print(f"\nTodos los archivos guardados en: {PATH_SALIDA}")


if __name__ == '__main__':
    main()
