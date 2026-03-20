import numpy as np

# TODO: establecer una base de diámetro pupilar dinámica basada en datos poblacionales

def simular_pupila(es_mv=False, t=None):
    """ Función bogus para generar señales de pupila """
    base = 4.0 # ej: base = np.random.normal(4.0, 0.5) para variabilidad (con iluminación base)
    
    if es_mv:
        # Criterios MV: Amplitud exagerada y frecuencia marcada
        amp_media = np.random.uniform(0.6, 1.0) 
        freq = 0.2 # TODO: revisar valores aleatorios
        std_ruido = 0.05
    else:
        # Criterios Control: Estabilidad
        amp_media = np.random.uniform(0.1, 0.3)
        freq = np.random.uniform(0.2, 0.5) 
        std_ruido = 0.01

    # Jitter: Variabilidad biológica en la amplitud (5%)
    jitter_amp = amp_media + np.random.normal(0, amp_media * 0.05, len(t))
    
    # Modelo: Senoidal (Hippus) + Ruido (Falla de inhibición central)
    onda = jitter_amp * np.sin(2 * np.pi * freq * t)
    ruido = np.random.normal(0, std_ruido, len(t))
    
    return base + onda + ruido
