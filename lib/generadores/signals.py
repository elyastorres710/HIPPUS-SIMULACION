import numpy as np

def generar_señal_pupilar(tiempo: np.ndarray, es_patologico: bool = False) -> np.ndarray:
    """
    Simula el comportamiento del hippus pupilar.
    """
    # Establecer diametro base (variabilidad entre 3.5 y 4.5 mm)
    diametro_base = np.random.normal(4.0, 0.5)
    
    if es_patologico:
        # Criterios para Migraña Vestibular
        amplitud = np.random.uniform(0.6, 1.0)
        frecuencia = np.random.uniform(0.18, 0.22)
        ruido_std = 0.05
    else:
        # Criterios para Sujeto Control
        amplitud = np.random.uniform(0.1, 0.3)
        frecuencia = np.random.uniform(0.2, 0.5)
        ruido_std = 0.01

    # Crear la señal: Base + Onda + Ruido
    onda = amplitud * np.sin(2 * np.pi * frecuencia * tiempo)
    ruido = np.random.normal(0, ruido_std, len(tiempo))
    
    return diametro_base + onda + ruido