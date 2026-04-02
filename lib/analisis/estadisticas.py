import numpy as np

def calcular_media_pupilar(señal: np.ndarray) -> float:
    """
    Calcula el promedio aritmetico del diametro pupilar.
    """
    return float(np.mean(señal))

def calcular_desviacion_estandar(señal: np.ndarray) -> float:
    """
    Calcula la desviacion estandar para cuantificar la variabilidad del hippus.
    """
    return float(np.std(señal))

def calcular_rms(señal: np.ndarray) -> float:
    """
    Calcula el valor eficaz (Root Mean Square) de la señal pupilar.
    """
    return float(np.sqrt(np.mean(np.square(señal))))

def calcular_pui(señal: np.ndarray) -> float:
    """
    Calcula el Indice de Inquietud Pupilar (PUI) como la suma de cambios absolutos.
    """
    return float(np.sum(np.abs(np.diff(señal))))
    #revisar segun bibliografia

def calcular_dfi(señal: np.ndarray) -> float:
    """
    Calcula el Indice de Fluctuacion Diferencial (Dfi) midiendo la amplitud pico a pico.
    """
    return float(np.ptp(señal))

def calcular_velocidad_promedio(señal: np.ndarray, fs: float) -> float:
    """
    Calcula la velocidad media de cambio pupilar en mm/s.
    """
    velocidad = np.abs(np.diff(señal) * fs)
    return float(np.mean(velocidad))
    #revisar segun bibliografia

def calcular_frecuencia_dominante(señal: np.ndarray, fs: float) -> float:
    """
    Identifica la frecuencia con mayor potencia mediante FFT (Transformada Rapida de Fourier).
    """
    señal_centrada = señal - np.mean(señal)
    fft_vals = np.abs(np.fft.rfft(señal_centrada))
    frecuencias = np.fft.rfftfreq(len(señal), d=1/fs)
    
    return float(frecuencias[np.argmax(fft_vals)])