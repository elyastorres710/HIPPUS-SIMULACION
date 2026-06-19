"""
Script 07 - Análisis de Explicabilidad mediante SHAP (XAI)
===========================================================
Aplica SHapley Additive exPlanations (SHAP) al modelo Random Forest
seleccionado automáticamente por el ranking de la iteración 2, con el
objetivo de explicar qué biomarcadores del hippus pupilar impulsan la
clasificación de Migraña Vestibular vs. Control.

Entradas:
    - data/processed/analisis_resultados_it2.csv  (biomarcadores por sujeto)
    - scripts/iteracion_2/metricas_completas_it2.csv  (ranking de combinaciones)

Salidas (data/test/):
    - xai_it2_importancia_global.png   : importancia media |SHAP| por biomarcador
    - xai_it2_summary_beeswarm.png     : distribución de impacto por sujeto/característica
    - xai_it2_caso_migrana.png         : explicación individual — caso Migraña Vestibular
    - xai_it2_caso_control.png         : explicación individual — caso Control
    - xai_it2_dependencia.png          : dependencia SHAP vs. valor real por biomarcador
    - xai_it2_reporte.txt              : resumen cuantitativo de importancias SHAP

Ejecución:
    python scripts/iteracion_2/07_xai_shap.py
"""

import ast
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # backend no interactivo para guardar figuras sin pantalla
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from typing import Tuple

# ---------------------------------------------------------------------------
# Constantes de configuración (alineadas con scripts 04 y 06)
# ---------------------------------------------------------------------------
PATH_DATOS: str = 'data/processed/analisis_resultados_it2.csv'
PATH_RANKING: str = 'scripts/iteracion_2/metricas_finales.csv'
PATH_SALIDA: str = 'data/test/'

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
N_ESTIMADORES: int = 100

ETIQUETA_POSITIVA: str = 'Migraña Vestibular'   # clase de interés clínico (y=1)
ETIQUETA_NEGATIVA: str = 'Control'               # clase de referencia (y=0)

DPI_FIGURA: int = 150


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
def cargar_datos_y_biomarcadores(
    path_datos: str,
    path_ranking: str
) -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    Carga el dataset procesado y selecciona los biomarcadores óptimos.

    Lee la primera fila del ranking (mayor F1-Score) para obtener la
    combinación de variables seleccionada automáticamente por el script 04.

    Args:
        path_datos: Ruta al CSV con biomarcadores extraídos por sujeto.
        path_ranking: Ruta al CSV de ranking de combinaciones (metricas_completas_it2.csv).

    Returns:
        Tuple con:
            - X (DataFrame): matriz de biomarcadores seleccionados.
            - y (Series): etiqueta binaria (0=Control, 1=Migraña Vestibular).
            - biomarcadores (list): nombres de columnas seleccionadas.

    Raises:
        FileNotFoundError: Si alguno de los archivos no existe.
        KeyError: Si las columnas esperadas no están presentes en los CSV.
    """
    if not os.path.exists(path_datos):
        raise FileNotFoundError(f"Dataset no encontrado: {path_datos}")
    if not os.path.exists(path_ranking):
        raise FileNotFoundError(f"Ranking no encontrado: {path_ranking}")

    df_ranking = pd.read_csv(path_ranking)
    df_ranking.columns = df_ranking.columns.str.strip()
    # Las variables están separadas por " + ", hacer split y limpiar espacios
    variables_str = df_ranking.iloc[0]['Variables']
    biomarcadores: list = [v.strip() for v in variables_str.split(' + ')]

    df_datos = pd.read_csv(path_datos)
    df_datos.columns = df_datos.columns.str.strip()

    X: pd.DataFrame = df_datos[biomarcadores]
    y: pd.Series = df_datos['Diagnostico'].map(
        {ETIQUETA_NEGATIVA: 0, ETIQUETA_POSITIVA: 1}
    )

    return X, y, biomarcadores


# ---------------------------------------------------------------------------
# Entrenamiento del modelo
# ---------------------------------------------------------------------------
def entrenar_modelo_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> RandomForestClassifier:
    """
    Entrena el Random Forest con la configuración estándar de la iteración 2.

    Args:
        X_train: Biomarcadores del conjunto de entrenamiento.
        y_train: Etiquetas binarias del conjunto de entrenamiento.

    Returns:
        Modelo RandomForestClassifier entrenado.
    """
    modelo = RandomForestClassifier(
        n_estimators=N_ESTIMADORES,
        random_state=RANDOM_STATE
    )
    modelo.fit(X_train, y_train)
    return modelo


# ---------------------------------------------------------------------------
# Cálculo de valores SHAP
# ---------------------------------------------------------------------------
def calcular_shap(
    modelo: RandomForestClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[np.ndarray, float]:
    """
    Calcula los valores SHAP para la clase positiva (Migraña Vestibular).

    Utiliza TreeExplainer (exacto y eficiente para Random Forest).
    El explainer se inicializa con X_train como datos de fondo para
    obtener el valor esperado correcto. Los valores SHAP se calculan
    sobre X_test para reflejar el desempeño real del modelo.

    Compatibilidad:
        - SHAP < 0.40 : shap_values() retorna lista [clase_0, clase_1]
        - SHAP >= 0.40: retorna ndarray 3D (n, features, clases)

    Nota: TreeExplainer para modelos basados en árboles calcula valores SHAP
    exactos usando la estructura del árbol directamente (feature_perturbation=
    'tree_path_dependent'), por lo que no requiere datos de fondo externos.

    Args:
        modelo: RandomForestClassifier entrenado.
        X_train: No utilizado; se mantiene en la firma para coherencia de interfaz.
        X_test: Conjunto de prueba a explicar.

    Returns:
        Tuple con:
            - shap_vals (ndarray): valores SHAP por muestra/feature, clase positiva.
            - valor_esperado (float): predicción base del modelo para la clase positiva.
    """
    explainer = shap.TreeExplainer(modelo)
    shap_values_raw = explainer.shap_values(X_test)
    expected_value_raw = explainer.expected_value

    if isinstance(shap_values_raw, list):
        # SHAP < 0.40: lista [clase_0, clase_1]
        shap_vals: np.ndarray = shap_values_raw[1]
        valor_esperado: float = (
            expected_value_raw[1]
            if isinstance(expected_value_raw, (list, np.ndarray))
            else float(expected_value_raw)
        )
    else:
        # SHAP >= 0.40: array 3D (n_muestras, n_features, n_clases)
        shap_vals = shap_values_raw[:, :, 1]
        valor_esperado = (
            float(expected_value_raw[1])
            if hasattr(expected_value_raw, '__len__')
            else float(expected_value_raw)
        )

    return shap_vals, valor_esperado


# ---------------------------------------------------------------------------
# Visualizaciones
# ---------------------------------------------------------------------------
def guardar_importancia_global(
    shap_vals: np.ndarray,
    X_test: pd.DataFrame,
    path_salida: str
) -> None:
    """
    Genera y guarda el gráfico de importancia global de biomarcadores.

    Muestra el valor |SHAP| medio de cada biomarcador en el conjunto de
    prueba, ordenado de mayor a menor impacto en la clasificación.

    Args:
        shap_vals: Valores SHAP para la clase Migraña Vestibular.
        X_test: Datos del conjunto de prueba (para nombres de columnas).
        path_salida: Directorio donde se guarda el archivo PNG.
    """
    shap.summary_plot(shap_vals, X_test, plot_type='bar', show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    plt.title('Importancia Global de Biomarcadores (|SHAP| medio)\nMigraña Vestibular vs. Control', pad=12)
    plt.tight_layout()
    fig.savefig(
        os.path.join(path_salida, 'xai_it2_importancia_global.png'),
        dpi=DPI_FIGURA, bbox_inches='tight'
    )
    plt.close(fig)
    print("  [OK] xai_it2_importancia_global.png")


def guardar_summary_beeswarm(
    shap_vals: np.ndarray,
    X_test: pd.DataFrame,
    path_salida: str
) -> None:
    """
    Genera y guarda el gráfico beeswarm de distribución SHAP.

    Muestra para cada biomarcador cómo el valor real de la variable
    (color) y su magnitud SHAP (posición horizontal) contribuyen a
    empujar la predicción hacia Migraña Vestibular (SHAP > 0) o
    hacia Control (SHAP < 0).

    Args:
        shap_vals: Valores SHAP para la clase Migraña Vestibular.
        X_test: Datos del conjunto de prueba.
        path_salida: Directorio donde se guarda el archivo PNG.
    """
    shap.summary_plot(shap_vals, X_test, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 6)
    plt.title(
        'Distribución de Valores SHAP por Biomarcador\n'
        '(SHAP > 0 → Migraña Vestibular   |   SHAP < 0 → Control)',
        pad=12
    )
    plt.tight_layout()
    fig.savefig(
        os.path.join(path_salida, 'xai_it2_summary_beeswarm.png'),
        dpi=DPI_FIGURA, bbox_inches='tight'
    )
    plt.close(fig)
    print("  [OK] xai_it2_summary_beeswarm.png")


def guardar_caso_individual(
    shap_vals: np.ndarray,
    valor_esperado: float,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    etiqueta_clase: int,
    nombre_archivo: str,
    titulo: str,
    path_salida: str
) -> None:
    """
    Genera y guarda el gráfico waterfall para un caso individual.

    Visualiza cómo cada biomarcador suma o resta probabilidad de
    Migraña Vestibular respecto al valor base del modelo para un
    sujeto concreto del conjunto de prueba.

    Args:
        shap_vals: Valores SHAP de todos los sujetos de prueba.
        valor_esperado: Predicción base del modelo (valor esperado global).
        X_test: Datos del conjunto de prueba.
        y_test: Etiquetas reales del conjunto de prueba.
        etiqueta_clase: Clase del sujeto a mostrar (0=Control, 1=MV).
        nombre_archivo: Nombre del PNG sin extensión.
        titulo: Título del gráfico.
        path_salida: Directorio donde se guarda el archivo PNG.
    """
    indices_clase = np.where(y_test.values == etiqueta_clase)[0]
    if len(indices_clase) == 0:
        print(f"  [AVISO] No se encontraron casos de clase {etiqueta_clase} en prueba.")
        return

    idx = indices_clase[0]
    explicacion = shap.Explanation(
        values=shap_vals[idx],
        base_values=valor_esperado,
        data=X_test.iloc[idx].values,
        feature_names=list(X_test.columns)
    )

    shap.plots.waterfall(explicacion, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 4)
    plt.title(titulo, pad=10)
    plt.tight_layout()
    fig.savefig(
        os.path.join(path_salida, f'{nombre_archivo}.png'),
        dpi=DPI_FIGURA, bbox_inches='tight'
    )
    plt.close(fig)
    print(f"  [OK] {nombre_archivo}.png")


def guardar_graficos_dependencia(
    shap_vals: np.ndarray,
    X_test: pd.DataFrame,
    biomarcadores: list,
    path_salida: str
) -> None:
    """
    Genera y guarda los gráficos de dependencia SHAP por biomarcador.

    Muestra la relación entre el valor real de cada biomarcador y su
    valor SHAP (impacto en la predicción), permitiendo detectar
    umbrales y no linealidades relevantes clínicamente.

    Args:
        shap_vals: Valores SHAP para la clase Migraña Vestibular.
        X_test: Datos del conjunto de prueba.
        biomarcadores: Lista de nombres de biomarcadores.
        path_salida: Directorio donde se guarda el archivo PNG.
    """
    n = len(biomarcadores)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for i, (biomarcador, ax) in enumerate(zip(biomarcadores, axes)):
        shap.dependence_plot(
            ind=i,
            shap_values=shap_vals,
            features=X_test,
            ax=ax,
            show=False
        )
        ax.set_title(f'Dependencia SHAP\n{biomarcador}', fontsize=10)
        ax.set_ylabel('Valor SHAP (impacto en MV)')

    fig.suptitle(
        'Dependencia entre Biomarcadores y su Impacto SHAP\n'
        '(↑ SHAP → mayor probabilidad de Migraña Vestibular)',
        fontsize=11, y=1.02
    )
    plt.tight_layout()
    fig.savefig(
        os.path.join(path_salida, 'xai_it2_dependencia.png'),
        dpi=DPI_FIGURA, bbox_inches='tight'
    )
    plt.close(fig)
    print("  [OK] xai_it2_dependencia.png")


# ---------------------------------------------------------------------------
# Reporte de texto
# ---------------------------------------------------------------------------
def guardar_reporte_texto(
    shap_vals: np.ndarray,
    biomarcadores: list,
    valor_esperado: float,
    path_salida: str
) -> None:
    """
    Guarda un resumen cuantitativo de las importancias SHAP en texto plano.

    Incluye el valor esperado base, el |SHAP| medio por biomarcador
    (ordenado de mayor a menor) y su porcentaje de contribución relativa.

    Args:
        shap_vals: Valores SHAP para la clase Migraña Vestibular.
        biomarcadores: Lista de nombres de biomarcadores.
        valor_esperado: Valor esperado base del modelo (log-odds clase 1).
        path_salida: Directorio donde se guarda el archivo TXT.
    """
    importancias_medias = np.abs(shap_vals).mean(axis=0)
    total = importancias_medias.sum()
    ranking = sorted(
        zip(biomarcadores, importancias_medias),
        key=lambda x: x[1], reverse=True
    )

    lineas = [
        "=== REPORTE XAI — SHAP sobre Random Forest (Iteración 2) ===",
        f"Biomarcadores evaluados : {biomarcadores}",
        f"Valor esperado base     : {valor_esperado:.4f}  "
        f"(predicción media del modelo en log-odds para {ETIQUETA_POSITIVA})",
        "",
        "Importancia media |SHAP| por biomarcador (ordenada de mayor a menor):",
        f"  {'Biomarcador':<20} {'|SHAP| medio':>14} {'Contribución relativa':>22}",
        "  " + "-" * 58,
    ]
    for nombre, imp in ranking:
        porcentaje = 100.0 * imp / total if total > 0 else 0.0
        lineas.append(f"  {nombre:<20} {imp:>14.4f} {porcentaje:>21.1f}%")

    lineas += [
        "",
        "Interpretación:",
        f"  SHAP > 0 → el biomarcador empuja la predicción hacia {ETIQUETA_POSITIVA}",
        f"  SHAP < 0 → el biomarcador empuja la predicción hacia {ETIQUETA_NEGATIVA}",
        "  |SHAP| medio → importancia global independiente de la dirección",
    ]

    ruta_txt = os.path.join(path_salida, 'xai_it2_reporte.txt')
    with open(ruta_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lineas) + '\n')

    print("  [OK] xai_it2_reporte.txt")
    print('\n'.join(lineas))


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Ejecuta el pipeline completo de explicabilidad SHAP para la iteración 2.

    Flujo:
        1. Carga datos y selecciona biomarcadores del ranking automático.
        2. Divide en entrenamiento/prueba con la misma configuración que el modelo base.
        3. Entrena el Random Forest.
        4. Calcula los valores SHAP con TreeExplainer.
        5. Genera y guarda cinco gráficos + un reporte de texto.
    """
    os.makedirs(PATH_SALIDA, exist_ok=True)

    print("\n=== Script 07 — XAI SHAP | Iteración 2 ===\n")

    # 1. Datos
    X, y, biomarcadores = cargar_datos_y_biomarcadores(PATH_DATOS, PATH_RANKING)
    print(f"Biomarcadores seleccionados (ranking automático): {biomarcadores}")
    print(f"Total de sujetos: {len(X)}  |  "
          f"Migraña Vestibular: {y.sum()}  |  Control: {(y == 0).sum()}")

    # 2. Partición
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Entrenamiento: {len(X_train)} sujetos | Prueba: {len(X_test)} sujetos\n")

    # 3. Modelo
    modelo_rf = entrenar_modelo_rf(X_train, y_train)
    print(f"Modelo Random Forest entrenado ({N_ESTIMADORES} árboles)")

    # 4. SHAP
    print("Calculando valores SHAP (TreeExplainer)...")
    shap_vals, valor_esperado = calcular_shap(modelo_rf, X_train, X_test)
    print(f"Dimensiones SHAP: {shap_vals.shape}  "
          f"(sujetos × biomarcadores, clase {ETIQUETA_POSITIVA})")
    print(f"Valor esperado base: {valor_esperado:.4f}\n")

    # 5. Gráficos
    print("Generando y guardando gráficos XAI...")
    guardar_importancia_global(shap_vals, X_test, PATH_SALIDA)
    guardar_summary_beeswarm(shap_vals, X_test, PATH_SALIDA)
    guardar_caso_individual(
        shap_vals, valor_esperado, X_test, y_test,
        etiqueta_clase=1,
        nombre_archivo='xai_it2_caso_migrana',
        titulo=f'Explicación Individual — Caso {ETIQUETA_POSITIVA}\n'
               f'(Biomarcadores: {", ".join(biomarcadores)})',
        path_salida=PATH_SALIDA
    )
    guardar_caso_individual(
        shap_vals, valor_esperado, X_test, y_test,
        etiqueta_clase=0,
        nombre_archivo='xai_it2_caso_control',
        titulo=f'Explicación Individual — Caso {ETIQUETA_NEGATIVA}\n'
               f'(Biomarcadores: {", ".join(biomarcadores)})',
        path_salida=PATH_SALIDA
    )
    guardar_graficos_dependencia(shap_vals, X_test, biomarcadores, PATH_SALIDA)

    # 6. Reporte texto
    print("\nGenerando reporte cuantitativo...")
    guardar_reporte_texto(shap_vals, biomarcadores, valor_esperado, PATH_SALIDA)

    print(f"\nTodos los archivos guardados en: {PATH_SALIDA}")


if __name__ == '__main__':
    main()
