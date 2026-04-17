import numpy as np

# LIMPIEZA DE ARTEFACTOS 
def limpiar_señal(señal: np.ndarray) -> np.ndarray:
    """
    Detecta ceros (parpadeos) y los rellena mediante interpolación lineal.
    """
    señal_limpia = señal.copy()
    indices_ceros = señal_limpia == 0
    
    if np.any(indices_ceros):
        puntos_validos = ~indices_ceros
        x_validos = np.where(puntos_validos)[0]
        y_validos = señal_limpia[puntos_validos]
        x_ceros = np.where(indices_ceros)[0]
        
        # Relleno de huecos para evitar saltos en la FFT
        señal_limpia[indices_ceros] = np.interp(x_ceros, x_validos, y_validos)
    
    return señal_limpia

# MÉTRICAS EN EL DOMINIO DEL TIEMPO 

def calcular_media_pupilar(señal: np.ndarray) -> float:
    # Cálculo sobre datos no nulos para evitar sesgo por parpadeo
    datos_reales = señal[señal > 0]
    return float(np.mean(datos_reales)) if len(datos_reales) > 0 else 0.0

def calcular_desviacion_estandar(señal: np.ndarray) -> float:
    # Evaluación de la variabilidad biológica sin artefactos
    señal_limpia = limpiar_señal(señal)
    return float(np.std(señal_limpia))

def calcular_rms(señal: np.ndarray) -> float:
    # Valor eficaz de la señal pre-procesada
    señal_limpia = limpiar_señal(señal)
    return float(np.sqrt(np.mean(np.square(señal_limpia))))

def calcular_pui(señal: np.ndarray) -> float:
    # Índice de inquietud basado en diferencias absolutas
    señal_limpia = limpiar_señal(señal)
    return float(np.sum(np.abs(np.diff(señal_limpia))))

def calcular_dfi(señal: np.ndarray) -> float:
    # Amplitud pico a pico de la fluctuación real
    señal_limpia = limpiar_señal(señal)
    return float(np.ptp(señal_limpia))

def calcular_velocidad_promedio(señal: np.ndarray, fs: float) -> float:
    # Dinámica de cambio pupilar en mm/s
    señal_limpia = limpiar_señal(señal)
    cambios = np.abs(np.diff(señal_limpia) * fs)
    return float(np.mean(cambios))

# MÉTRICAS EN EL DOMINIO DE LA FRECUENCIA (FFT) 

def calcular_pual(señal: np.ndarray, fs: float) -> float:
    # Cálculo del PUAL absoluto (0.2 - 2.0 Hz) según Casani 2026
    señal_limpia = limpiar_señal(señal)
    señal_centrada = señal_limpia - np.mean(señal_limpia)
    n = len(señal_limpia)
    
    fft_vals = np.abs(np.fft.rfft(señal_centrada)) / n
    freqs = np.fft.rfftfreq(n, 1/fs)
    
    mask_casani = (freqs >= 0.2) & (freqs <= 2.0)
    pual_absoluto = np.sum(fft_vals[mask_casani])
    
    return round(float(pual_absoluto), 4)

def calcular_pual_ratio(señal: np.ndarray, fs: float) -> float:
    # Ratio de energía patológica normalizada
    señal_limpia = limpiar_señal(señal)
    señal_centrada = señal_limpia - np.mean(señal_limpia)
    n = len(señal_limpia)
    
    fft_vals = np.abs(np.fft.rfft(señal_centrada)) / n
    freqs = np.fft.rfftfreq(n, 1/fs)
    
    mask_casani = (freqs >= 0.2) & (freqs <= 2.0)
    mask_total = (freqs >= 0) & (freqs <= 2.0)
    
    p_patologica = np.sum(fft_vals[mask_casani])
    p_total = np.sum(fft_vals[mask_total])
    
    ratio = p_patologica / p_total if p_total > 0 else 0
    return round(float(ratio), 4)

def calcular_frecuencia_dominante(señal: np.ndarray, fs: float) -> float:
    # Identificación del pico espectral principal
    señal_limpia = limpiar_señal(señal)
    señal_centrada = señal_limpia - np.mean(señal_limpia)
    
    fft_vals = np.abs(np.fft.rfft(señal_centrada))
    frecuencias = np.fft.rfftfreq(len(señal_limpia), d=1/fs)
    
    return float(frecuencias[np.argmax(fft_vals)])
    