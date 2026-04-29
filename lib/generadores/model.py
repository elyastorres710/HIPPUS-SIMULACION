"""
Modelo generador de signal pupilar avanzado.
Clase para encapsular la lógica de generación de señales pupilares con características patológicas y no patológicas.
"""

import numpy as np
from typing import Optional, Tuple


class PupilSignalGenerator:
    """Generador de signales pupilares con características avanzadas."""
    
    def __init__(self, fs: float = 60.0, seed: Optional[int] = None, 
                 nivel_afectacion: float = 0.0, # 0.0 es salud perfecta
                 amplitud_max: float = 8.0, 
                 amplitud_min: float = 2.0,
                 diametro_base_range: Tuple[float, float] = (3.0, 7.0),
                 ruido_std_range: Tuple[float, float] = (0.02, 0.08), # Valores más realistas
                 deriva_std: float = 0.001,
                 ruido_rosa_scale: float = 0.02,
                 filtro_cutoff: float = 2.5, # Frecuencia de corte del filtro pasa bajos (Hz)
                 parpadeo_frecuencia_base: float = 0.25, # Frecuencia base de parpadeo (Hz, ~15 parpadeos/min)
                 parpadeo_variabilidad: float = 0.3, # Variabilidad aleatoria de la frecuencia (0-1)
                 video_resolucion: Tuple[int, int] = (1920, 1080), # Resolución del video (ancho_px, alto_px)
                 porcentaje_pupila_resolucion: float = 0.15): # % de la resolución que ocupa la pupila (0-1)
        """
        Inicializa el generador de signal pupilar.
        
        Args:
            fs: Frecuencia de muestreo en Hz
            seed: Semilla para reproducibilidad
            nivel_afectacion: Nivel de afectación por irregularidades (0-1)
            amplitud_max: Amplitud máxima de la signal (normalizador global)
            amplitud_min: Amplitud mínima de la signal (normalizador global)
            diametro_base_range: Rango del diámetro base (min, max)
            ruido_std_range: Rango de desviación estándar del ruido blanco (min, max)
            deriva_std: Desviación estándar de la deriva
            ruido_rosa_scale: Escala del ruido rosa
            filtro_cutoff: Frecuencia de corte del filtro pasa bajos (Hz)
            parpadeo_frecuencia_base: Frecuencia base de parpadeo (Hz)
            parpadeo_variabilidad: Variabilidad aleatoria de la frecuencia (0-1)
            video_resolucion: Resolución del video (ancho_px, alto_px)
            porcentaje_pupila_resolucion: % de la resolución que ocupa la pupila (0-1)
        """
        self.fs = fs
        self.nivel_afectacion = np.clip(nivel_afectacion, 0.0, 1.0)
        self.amplitud_max = amplitud_max
        self.amplitud_min = amplitud_min
        self.diametro_base_range = diametro_base_range
        self.ruido_std_range = ruido_std_range
        self.deriva_std = deriva_std
        self.ruido_rosa_scale = ruido_rosa_scale
        self.filtro_cutoff = filtro_cutoff
        self.parpadeo_frecuencia_base = parpadeo_frecuencia_base
        self.parpadeo_variabilidad = parpadeo_variabilidad
        self.video_resolucion = video_resolucion
        self.porcentaje_pupila_resolucion = porcentaje_pupila_resolucion
        
        if seed is not None:
            np.random.seed(seed)
    
    def _generar_ruido_rosa(self, signal: np.ndarray) -> np.ndarray:
        """Genera ruido rosa (1/f noise) sobre una signal."""
        n_muestras = len(signal)
        ruido_blanco = np.random.normal(0, 1, n_muestras)
        return np.cumsum(ruido_blanco) / np.sqrt(np.arange(1, n_muestras + 1))
    
    def _aplicar_filtro_pasa_bajos(self, signal: np.ndarray) -> np.ndarray:
        """
        Aplica un filtro pasa bajos Butterworth de segundo orden.
        
        Args:
            signal: Array de la signal a filtrar
            
        Returns:
            Señal filtrada
        """
        from scipy.signal import butter, filtfilt
        
        nyquist = 0.5 * self.fs
        cutoff_normalized = self.filtro_cutoff / nyquist
        
        # Filtro Butterworth de segundo orden
        b, a = butter(2, cutoff_normalized, btype='low')
        
        # Filtrado bidireccional (filtfilt) para evitar desplazamiento de fase
        signal_filtrada = filtfilt(b, a, signal)
        
        return signal_filtrada
    
    def _aplicar_parpadeos(self, signal: np.ndarray) -> np.ndarray:
        """
        Aplica parpadeos artificiales a la signal usando sistema frecuencial pseudo aleatorio.
        
        Args:
            signal: Array de la signal pupilar
            
        Returns:
            Señal con parpadeos aplicados
        """
        signal = signal.copy()
        n_muestras = len(signal)
        
        # Calcular intervalo base entre parpadeos (en muestras)
        intervalo_base_muestras = int((1.0 / self.parpadeo_frecuencia_base) * self.fs)
        
        # Generar parpadeos con variabilidad aleatoria
        indice_actual = 0
        while indice_actual < n_muestras:
            # Variabilidad aleatoria en el intervalo
            variacion = np.random.uniform(-self.parpadeo_variabilidad, self.parpadeo_variabilidad)
            intervalo_muestras = int(intervalo_base_muestras * (1 + variacion))
            
            indice_actual += intervalo_muestras
            
            if indice_actual >= n_muestras:
                break
            
            # Duración del parpadeo en muestras (100-300ms)
            duracion_parpadeo_seg = np.random.uniform(0.1, 0.3)
            duracion_parpadeo_muestras = int(duracion_parpadeo_seg * self.fs)
            indice_fin = indice_actual + duracion_parpadeo_muestras
            
            # Aplicar parpadeo si está dentro de los límites
            if indice_fin <= n_muestras:
                signal[indice_actual:indice_fin] = 0
        
        return signal
    
    def _generar_onda_base(self, t: np.ndarray) -> np.ndarray:
        """
        Genera la onda base con diámetro base, envolvente y componentes sinusoidales.
        
        Args:
            t: Vector de tiempo
            
        Returns:
            Señal biológica completa (diámetro base + onda * envolvente)
        """
        # Diámetro base
        diametro_base = np.random.uniform(*self.diametro_base_range)
        
        # Envolvente
        envolvente = 0.5 * (1 + np.sin(2 * np.pi * 0.05 * t))
        
        # Nivel de afectación determina amplitudes y frecuencias
        nivel = self.nivel_afectacion
        
        # Amplitudes base según nivel de afectación
        if nivel < 0.3:
            # Baja afectación (similar a control)
            base_amps = [np.random.uniform(0.01, 0.04) for _ in range(3)]
            freqs = [np.random.uniform(0.1, 2.0) for _ in range(3)]
        elif nivel < 0.7:
            # Afectación media (similar a somnolencia/PH)
            base_amps = [np.random.uniform(0.08, 0.18) for _ in range(3)]
            freqs = [np.random.uniform(0.1, 0.6) for _ in range(3)]
        else:
            # Alta afectación (similar a patológico)
            base_amps = [np.random.uniform(0.12, 0.15) for _ in range(3)]
            freqs = [np.random.uniform(0.2, 2.0) for _ in range(3)]
        
        # Escalar amplitudes por nivel_afectacion y normalizar
        amps = [a * nivel * (self.amplitud_max - self.amplitud_min) for a in base_amps]
        
        onda_hippus = sum(a * np.sin(2 * np.pi * f * t + np.random.uniform(0, 2*np.pi)) 
                          for a, f in zip(amps, freqs))
        
        # Señal biológica completa
        signal_biologica = diametro_base + (onda_hippus * envolvente)
        return signal_biologica
    
    def _aplicar_interferencia_biologica(self, signal: np.ndarray) -> np.ndarray:
        """
        Aplica componentes de interferencia biológica (deriva, ruido rosa) a la signal.
        
        Args:
            signal: Array de la signal sobre la cual aplicar interferencia
            
        Returns:
            Signal con interferencia biológica aplicada
        """
        n_muestras = len(signal)
        deriva = np.cumsum(np.random.normal(0, self.deriva_std, n_muestras))
        r_rosa = self._generar_ruido_rosa(signal) * self.ruido_rosa_scale
        
        interferencia = deriva + r_rosa
        return signal + interferencia
    
    def _aplicar_ruido_medicion(self, signal: np.ndarray) -> np.ndarray:
        """
        Aplica ruido de medición (ruido blanco) a la signal basado en decimación visual.
        
        El ruido de medición se calcula dinámicamente según:
        - Resolución del video
        - Porcentaje de resolución que ocupa la pupila
        - Asumiendo detección de bordes perfecta (error de 1 píxel)
        
        Args:
            signal: Array de la signal sobre la cual aplicar ruido de medición
            
        Returns:
            Signal con ruido de medición aplicado
        """
        n_muestras = len(signal)
        
        # Calcular diámetro de pupila en píxeles (dimensión menor de la resolución)
        min_dimension = min(self.video_resolucion)
        diametro_px = min_dimension * self.porcentaje_pupila_resolucion
        
        # Calcular diámetro promedio en mm (del rango configurado)
        diametro_mm_promedio = np.mean(self.diametro_base_range)
        
        # Ratio mm/px (cuántos milímetros representa un píxel)
        ratio_mm_px = diametro_mm_promedio / diametro_px
        
        # Ruido de medición = 1 píxel de error * ratio mm/px
        ruido_medicion_mm = ratio_mm_px
        
        r_blanco = np.random.normal(0, ruido_medicion_mm, n_muestras)
        return signal + r_blanco
    
    def generar(self, duracion: float = 60.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera una signal pupilar completa.
        
        Args:
            duracion: Duración de la signal en segundos
            
        Returns:
            Tupla (tiempo, signal) con la signal generada
        """
        n_muestras = int(duracion * self.fs)
        t = np.linspace(0, duracion, n_muestras)
        
        # Generar señal biológica base
        signal_biologica = self._generar_onda_base(t)
        
        # Aplicar interferencia biológica
        pupila_final = self._aplicar_interferencia_biologica(signal_biologica)
        
        # Aplicar filtro pasa bajos
        pupila_final = self._aplicar_filtro_pasa_bajos(pupila_final)
        
        # Aplicar parpadeos (después del filtro)
        pupila_final = self._aplicar_parpadeos(pupila_final)
        
        # Aplicar ruido de medición (etapa final)
        pupila_final = self._aplicar_ruido_medicion(pupila_final)
        
        return t, pupila_final
    
    def view(self, t: np.ndarray, signal: np.ndarray, titulo: str = "Señal Pupilar", 
             guardar: bool = False, ruta_salida: Optional[str] = None):
        """
        Visualiza la signal pupilar generada.
        
        Args:
            t: Vector de tiempo
            signal: Señal pupilar
            titulo: Título del gráfico
            guardar: Si es True, guarda la figura en lugar de mostrarla
            ruta_salida: Ruta donde guardar la figura (solo si guardar=True)
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t, signal, linewidth=0.8, color='blue')
        ax.set_xlabel('Tiempo (s)')
        ax.set_ylabel('Diámetro Pupilar (mm)')
        ax.set_title(titulo)
        ax.grid(True, alpha=0.3)
        
        if guardar:
            if ruta_salida is None:
                raise ValueError("Debe especificar ruta_salida cuando guardar=True")
            plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
