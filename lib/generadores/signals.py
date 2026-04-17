import numpy as np

def generar_ruido_rosa(n_muestras):
    ruido_blanco = np.random.normal(0, 1, n_muestras)
    return np.cumsum(ruido_blanco) / np.sqrt(np.arange(1, n_muestras + 1))

def aplicar_parpadeos(señal, probabilidad=0.7):
    if np.random.rand() < probabilidad:
        n_parpadeos = np.random.randint(1, 4)
        for _ in range(n_parpadeos):
            inicio = np.random.randint(0, len(señal) - 30)
            duracion = np.random.randint(5, 20) 
            señal[inicio : inicio + duracion] = 0
    return señal

def generar_señal_pupilar(es_patologico, t, fs=60.0):
    diametro_base = np.random.uniform(2.5, 6.0)
    envolvente = 0.5 * (1 + np.sin(2 * np.pi * 0.05 * t))
    
    if es_patologico:
        # 25% de casos con PNy débil (Falsos Negativos)
        if np.random.rand() < 0.25:
            amps = [np.random.uniform(0.04, 0.07) for _ in range(3)]
        else:
            amps = [np.random.uniform(0.12, 0.15) for _ in range(3)]
        freqs = [np.random.uniform(0.2, 2.0) for _ in range(3)]
    else:
        # 7% de casos con somnolencia/PH (Falsos Positivos)
        if np.random.rand() < 0.07:
            amps = [np.random.uniform(0.08, 0.18) for _ in range(3)]
            freqs = [np.random.uniform(0.1, 0.6) for _ in range(3)]
        else:
            amps = [np.random.uniform(0.01, 0.04) for _ in range(3)]
            freqs = [np.random.uniform(0.1, 2.0) for _ in range(3)]
    
    onda_hippus = sum(a * np.sin(2 * np.pi * f * t + np.random.uniform(0, 2*np.pi)) 
                      for a, f in zip(amps, freqs))
    
    std_ruido = np.random.uniform(0.08, 0.15)
    r_blanco = np.random.normal(0, std_ruido, len(t))
    deriva = np.cumsum(np.random.normal(0, 0.002, len(t)))
    r_rosa = generar_ruido_rosa(len(t)) * 0.03
    
    señal_biologica = diametro_base + (onda_hippus * envolvente)
    interferencia = r_blanco + deriva + r_rosa
    
    pupila_final = señal_biologica + interferencia
    
    return aplicar_parpadeos(pupila_final)
