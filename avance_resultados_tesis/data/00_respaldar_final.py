import shutil
import os
from datetime import datetime

def ejecutar_respaldo_total():
    """
    Realiza una copia de seguridad integral de los datos, resultados y reportes validados.
    """
    # Identificación de la carpeta de destino
    fecha_str = datetime.now().strftime("%Y%m%d")
    nombre_respaldo = f"respaldo_final_validado_{fecha_str}"
    ruta_destino = os.path.join("respaldos", nombre_respaldo)
    
    # Directorios a respaldar
    carpetas_fuente = {
        "data": "data/",
        "reportes": "docs/resultados_iteracion_1/",
        "scripts_uso": "scripts/iteracion_1/",
        "librerias": "lib/"
    }

    print(f"Iniciando proceso de respaldo en: {ruta_destino}")

    try:
        # Creación de la estructura de respaldo
        if not os.path.exists(ruta_destino):
            os.makedirs(ruta_destino)

        for etiqueta, ruta in carpetas_fuente.items():
            if os.path.exists(ruta):
                destino_especifico = os.path.join(ruta_destino, etiqueta)
                # Copia recursiva de los directorios
                shutil.copytree(ruta, destino_especifico, dirs_exist_ok=True)
                print(f" -> {etiqueta.capitalize()} respaldado correctamente.")
            else:
                print(f" -> Aviso: No se localizó la carpeta {ruta}, omitiendo...")

        print("\nPROCESO COMPLETADO EXITOSAMENTE")
        print(f"Toda la evidencia clínica ha sido protegida en: {ruta_destino}")

    except Exception as e:
        print(f"Error durante el respaldo: {e}")

if __name__ == "__main__":
    ejecutar_respaldo_total()
