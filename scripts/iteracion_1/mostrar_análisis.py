"""
Script para mostrar el análisis de datos.

carga el archivo de processed/analisis_resultados.csv
y muestra los resultados de forma gráfica como puntos distribuidos por diagnosticos

1 color = 1 medida
1 forma = 1 diagnóstico
1 punto = 1 paciente

Eje X: Sujetos (índice de paciente)
Eje Y: Magnitud de la medida
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def generar_distribucion_medida(df, medida, diagnosticos, color_dict, ruta_resultados):
    """
    Genera y guarda la distribución de probabilidad (PDF) para una medida específica.
    
    Args:
        df: DataFrame con los datos
        medida: Nombre de la medida a graficar
        diagnosticos: Lista de diagnósticos únicos
        color_dict: Diccionario de colores por diagnóstico
        ruta_resultados: Ruta donde guardar la imagen
    """
    plt.figure(figsize=(10, 6))
    
    # separa la medida por diagnóstico
    for diagnostico in diagnosticos:
        subset = df[df["Diagnostico"] == diagnostico][medida]
        
        # Crea el histograma con KDE (Kernel Density Estimation)
        sns.histplot(data=subset, kde=True, stat="density", 
                    label=diagnostico, alpha=0.5, bins=20,
                    color=color_dict[diagnostico])
    
    plt.xlabel(medida, fontsize=12)
    plt.ylabel("Densidad de Probabilidad", fontsize=12)
    plt.title(f"Distribución de {medida} por Diagnóstico", fontsize=14)
    plt.legend(title="Diagnóstico")
    plt.grid(True, alpha=0.3)
    
    # Guardar cada distribución
    plt.savefig(ruta_resultados + f"distribucion_{medida.lower()}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  - Guardada: distribucion_{medida.lower()}.png")

def generar_scatter_medida(df, medida, diagnosticos, color_dict, marcador_dict, ruta_resultados):
    """
    Genera y guarda el gráfico scatter de una medida específica.
    
    Args:
        df: DataFrame con los datos
        medida: Nombre de la medida a graficar
        diagnosticos: Lista de diagnósticos únicos
        color_dict: Diccionario de colores por diagnóstico
        marcador_dict: Diccionario de marcadores por diagnóstico
        ruta_resultados: Ruta donde guardar la imagen
    """
    plt.figure(figsize=(12, 6))
    
    # Para cada diagnóstico, graficar sus puntos
    for diagnostico in diagnosticos:
        subset = df[df["Diagnostico"] == diagnostico]
        
        # X: índice del paciente, Y: valor de la medida
        x_vals = subset.index
        y_vals = subset[medida]
        
        plt.scatter(x_vals, y_vals, 
                  label=diagnostico,
                  color=color_dict[diagnostico],
                  marker=marcador_dict[diagnostico],
                  s=60,
                  alpha=0.7,
                  edgecolors='black',
                  linewidth=0.5)
    
    plt.xlabel("Índice del Paciente (Sujeto)", fontsize=12)
    plt.ylabel(medida, fontsize=12)
    plt.title(f"Distribución de {medida} por Paciente", fontsize=14)
    plt.legend(title="Diagnóstico", loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    plt.tight_layout()
    
    # Guardar figura individual
    plt.savefig(ruta_resultados + f"scatter_{medida.lower()}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  - Guardada: scatter_{medida.lower()}.png")

def main():
    ruta_entrada = "data/processed/analisis_resultados.csv"
    ruta_resultados = "data/results/"
    # Cargar datos
    df = pd.read_csv(ruta_entrada)
    
    # Mostrar resultados (solo encabezado)
    print(df.head())
    
    # Identificar las medidas (columnas excepto Diagnostico)
    medidas = [col for col in df.columns if col != "Diagnostico"]
    print("Medidas:", medidas)
    print("Diagnósticos:", df["Diagnostico"].unique())
    
    # Configuración inicial
    os.makedirs(ruta_resultados, exist_ok=True)
    diagnosticos = df["Diagnostico"].unique()
    colores = sns.color_palette("husl", len(diagnosticos))
    color_dict = dict(zip(diagnosticos, colores))
    marcadores = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    marcador_dict = {diag: marcadores[i % len(marcadores)] for i, diag in enumerate(diagnosticos)}
    
    # Generar scatter plots individuales por medida
    print("\nGenerando scatter plots por medida...")
    for medida in medidas:
        generar_scatter_medida(df, medida, diagnosticos, color_dict, marcador_dict, ruta_resultados)
    
    # Generar distribuciones por medida (PDF)
    print("\nGenerando distribuciones por medida...")
    for medida in medidas:
        generar_distribucion_medida(df, medida, diagnosticos, color_dict, ruta_resultados)
    
    print(f"\nTodas las visualizaciones guardadas en: {ruta_resultados}")
        
if __name__ == "__main__":
    main()