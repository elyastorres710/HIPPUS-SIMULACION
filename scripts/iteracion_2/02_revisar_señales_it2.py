import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Rutas de entrada y salida basadas en la estructura del proyecto
    input_file = "data/raw/dataset_final.csv"
    output_dir = "scripts/iteracion_2"
    output_plot = os.path.join(output_dir, "verificacion_final.png")

    # Verificar que el archivo generado en el script 01 exista
    if not os.path.exists(input_file):
        print(f"Error: No se encontró el archivo '{input_file}'.")
        print("Asegúrate de ejecutar primero el script 01 de generación de datos.")
        return

    print("Cargando el conjunto de datos para la validación visual...")
    df = pd.read_csv(input_file)

    # Filtrar el DataFrame por las etiquetas de diagnóstico conocidas
    df_control = df[df["Diagnostico"] == "Control"]
    df_mv = df[df["Diagnostico"] == "Migraña Vestibular"]

    # Validar que ambas clases contengan datos
    if df_control.empty or df_mv.empty:
        print("Error: El archivo no contiene ambas clases ('Control' y 'Migraña Vestibular').")
        return

    # Selección específica de una fila para cada grupo
    idx_control = 165
    idx_num_control = df_control.index.get_loc(idx_control)

    idx_mv = 987
    idx_num_mv = df_mv.index.get_loc(idx_mv)    
    
    # Extraer las series de tiempo numéricas omitiendo la columna del diagnóstico
    valores_control = df_control.loc[idx_control].iloc[:-1].astype(float).values
    valores_mv = df_mv.loc[idx_mv].iloc[:-1].astype(float).values

    # Configuración del eje de tiempo (30 segundos distribuidos uniformemente)
    duracion_total = 30.0
    tiempo = np.linspace(0, duracion_total, len(valores_control))

    print(f"Sujeto de Control seleccionado: Fila {idx_control} (N° {idx_num_control} del grupo sano)")
    print(f"Sujeto con Migraña Vestibular seleccionado: Fila {idx_mv} (N° {idx_num_mv} del grupo patológico)")

    # Construcción de la gráfica comparativa
    plt.figure(figsize=(12, 6))
    
    plt.plot(tiempo, valores_mv, color='#e74c3c', linewidth=1.5,linestyle='--', label=f'Migraña Vestibular (ID: {idx_mv})', alpha=1) 
    plt.plot(tiempo, valores_control, color='#2ecc71', linewidth=1.5, label=f'Sano / Control (ID: {idx_control})', alpha=1)
    
    plt.xlabel('Tiempo (s)', fontsize=11)
    plt.ylabel('Diámetro Pupilar Izquierdo (mm)', fontsize=11)
    plt.title(f'Comparación Dinámica: Sujeto Sano (ID: {idx_control}) vs Migraña Vestibular (ID: {idx_mv}) — Iteración 2', fontsize=12, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(2.5, 6.5)
    
    plt.tight_layout()
    
    # Asegurar la existencia del directorio de destino y guardar la imagen
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nGráfico generado exitosamente y guardado en: '{output_plot}'")

if __name__ == "__main__":
    main()