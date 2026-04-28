import numpy as np
from scipy.signal import butter, filtfilt
from scipy.ndimage import uniform_filter1d

def filtro_estabilidad_basal(datos, fs=60.0):
    """Elimina derivas de baja frecuencia para asegurar el rango mesópico."""
    frecuencia_nyquist = 0.5 * fs
    b, a = butter(2, 0.05 / frecuencia_nyquist, btype='high')
    return filtfilt(b, a, datos)

def generar_ruido_fisiologico(n):
    """Simula la actividad estocástica del sistema nervioso autónomo (1/f)."""
    blanco = np.random.normal(0, 1, n)
    frecuencias = np.fft.rfftfreq(n)
    frecuencias[0] = frecuencias[1]
    espectro = np.fft.rfft(blanco) / np.sqrt(frecuencias)
    return np.fft.irfft(espectro, n=n)

def generar_señal_pupilar(es_patologico, t, fs=60.0):
    """Genera registros pupilares para validación clínica de Hippus."""
    n, dt = len(t), 1/fs
    
    # 1. PARAMETROS BASE
    diam_base = np.random.uniform(4.0, 5.0)

    if es_patologico:
        # --- Grupo Patológico: Criterios Gufoni ---
        amp_g = np.random.uniform(0.5, 1.1) 
        freq_g = np.random.uniform(0.04, 2.0)
        pual_factor = 5.2 
        atenuacion = np.random.uniform(0.015, 0.65) 
        
        # Coherencia rítmica para el análisis computacional
        jitter_fase = 0.002 
    else:
        # --- Grupo Control: Fisiología Normal ---
        amp_g = np.random.uniform(0.05, 0.15) 
        freq_g = np.random.uniform(0.1, 0.5) 
        atenuacion = np.random.normal(loc=0.15, scale=0.08) 
        atenuacion = np.clip(atenuacion, 0.04, 0.40) 
        pual_factor = 0.88 
        jitter_fase = 0.02

    # 2. COMPOSICION DE LA SEÑAL (Mantenemos tu lógica exacta paso a paso)
    ruido_autonomo = filtro_estabilidad_basal(generar_ruido_fisiologico(n), fs)
    ruido_autonomo = (ruido_autonomo / np.std(ruido_autonomo)) * pual_factor
    
    # Fase biológica y componente oscilatorio
    fase_acumulada = 2 * np.pi * np.cumsum(freq_g + (np.random.normal(0, jitter_fase, n))) * dt
    onda_hippus = (amp_g / 2) * np.sin(fase_acumulada)
    
    # 3. LIMITACION FISIOLOGICA Y RUIDO INSTRUMENTAL
    # El tanh asegura que la pupila no desborde los límites anatómicos del iris
    fluctuacion = 1.4 * np.tanh(((onda_hippus + ruido_autonomo) * atenuacion) / 1.4)

    ruido_equipo = np.random.normal(0, 0.002, n)
    registro_final = diam_base + fluctuacion + ruido_equipo
    
    # Suavizado clínico para emular la captura del videonistagmógrafo
    return np.round(uniform_filter1d(registro_final, size=5), 2)
