"""
Johansson & Balkenius (2018) - Modelo computacional de dilatación pupilar
Implementación del modelo conexionista a nivel de sistema de control pupilar.

Clases:
    Conexion                  - Conexión sináptica con buffer de retraso
    JohanssonBalkeniusBox     - Núcleo individual / región cerebral
    JohanssonBalkeniusSystem  - Sistema interconectado completo
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple, Union, Callable

# Tipo auxiliar: parámetros β y γ pueden ser un float fijo o el centinela 'auto'
# para que el sistema los resuelva como 1/N (excitatorias) o 1/M (inhibitorias)
# según la topología, siguiendo el comportamiento por defecto del paper de
# Johansson & Balkenius (2018).
ParamFloat = Union[float, str]


# ---------------------------------------------------------------------------
# Conexion
# ---------------------------------------------------------------------------

class Conexion:
    """
    Conexión sináptica entre dos cajas.

    Mantiene un buffer de historial para implementar el retraso de transmisión τ.
    Interpola linealmente entre muestras del buffer para recuperar la
    salida retrasada en cualquier tiempo solicitado.

    Parámetros
    ----------
    fuente : JohanssonBalkeniusBox
        Caja fuente (pre-sináptica).
    tipo : str
        Tipo de conexión: 'excitatory', 'inhibitory', 'shunting', o 'us'.

        - 'excitatory':  contribuye al término β·Σ(E_i·w_i) de la ecuación (1).
        - 'inhibitory':  contribuye al término γ·ΣI_j de la ecuación (1).
        - 'shunting':    contribuye al factor 1/(1+S) de la ecuación (1).
        - 'us':          alimenta el canal de US (estímulo incondicionado) de
                         la regla delta (ecuación 3) en cajas plásticas.
                         Si una caja recibe múltiples conexiones 'us', su valor
                         efectivo se calcula como el PROMEDIO de ellas (US es
                         conceptualmente una señal singular). En cajas no
                         plásticas, las conexiones 'us' son no-op.
    tau : float
        Retraso de transmisión en segundos (ej. 0.020 para 20 ms).
    resting_output : float
        Valor usado para pre-llenar el buffer antes de iniciar la simulación.
    buffer_margin : float
        Tiempo extra más allá de τ mantenido en el buffer (segundos). Por defecto 0.1 s.
    """

    VALID_TYPES = ('excitatory', 'inhibitory', 'shunting', 'us')  # tipos válidos de conexión

    def __init__(
        self,
        fuente: 'JohanssonBalkeniusBox',
        tipo: str,
        tau: float = 0.020,
        resting_output: float = 0.0,
        buffer_margin: float = 0.1,
    ):
        if tipo not in self.VALID_TYPES:
            raise ValueError(f"tipo debe ser uno de {self.VALID_TYPES}, se obtuvo '{tipo}'")

        self.fuente = fuente
        self.tipo = tipo
        self.tau = tau
        self.buffer_margin = buffer_margin

        # Buffer: deque de tuplas (t, o), ordenadas por tiempo
        self._buffer: deque = deque()

        # Pre-llenar buffer desde -tau - margin hasta 0
        self._prefill(resting_output)

    # ------------------------------------------------------------------
    # Gestión del buffer
    # ------------------------------------------------------------------

    def _prefill(self, resting_output: float) -> None:
        """Llena el buffer con el valor de reposo para cubrir el retraso inicial."""
        t_start = -(self.tau + self.buffer_margin)
        # Dos puntos son suficientes: la interpolación lineal retornará resting_output
        self._buffer.append((t_start, resting_output))
        self._buffer.append((0.0, resting_output))

    def push(self, t: float, o: float) -> None:
        """
        Registra la salida de la fuente en el tiempo t.
        Elimina muestras antiguas más allá de tau + margin.
        """
        self._buffer.append((t, o))
        cutoff = t - self.tau - self.buffer_margin
        while len(self._buffer) > 2 and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

    def get_delayed_output(self, t: float) -> float:
        """
        Retorna la salida interpolada de la fuente en el tiempo (t - τ).

        Parámetros
        ----------
        t : float
            Tiempo actual de simulación en segundos.

        Retorna
        -------
        float
            Salida interpolada de la fuente en el tiempo t - τ.
        """
        t_target = t - self.tau

        buf = list(self._buffer)

        # Casos extremos: objetivo antes o después del rango del buffer
        if t_target <= buf[0][0]:
            return buf[0][1]
        if t_target >= buf[-1][0]:
            return buf[-1][1]

        # Interpolación lineal entre las dos muestras circundantes
        for i in range(len(buf) - 1):
            t0, o0 = buf[i]
            t1, o1 = buf[i + 1]
            if t0 <= t_target <= t1:
                if t1 == t0:
                    return o0
                alpha = (t_target - t0) / (t1 - t0)
                return o0 + alpha * (o1 - o0)

        return buf[-1][1]  # valor por defecto


# ---------------------------------------------------------------------------
# JohanssonBalkeniusBox
# ---------------------------------------------------------------------------

class JohanssonBalkeniusBox:
    """
    Núcleo individual en el modelo de control pupilar de Johansson & Balkenius.

    Implementa las Ecuaciones (1), (2), y opcionalmente (3) del artículo.

    Ecuación (1) - dinámica de estado:
        ε · dx/dt = α + β · (1/(1+S)) · Σ(Eᵢ·wᵢ) - γ · ΣIⱼ - x

    Ecuación (2) - salida:
        o = φ · arctan(x),  donde φ = 1/arctan(1)

    Ecuación (3) - actualización de pesos (solo cajas plásticas, cuando US > o):
        ε · dw_i/dt = λ · E_i · (US - o)

    Parámetros
    ----------
    name : str
        Identificador de este núcleo.
    alpha : float
        Nivel de reposo (por defecto 0.0).
    beta : float
        Factor de escala excitatorio (por defecto 1.0; se establece por-N dentro del sistema).
    gamma : float
        Factor de escala inhibitorio (por defecto 1.0).
    epsilon : float
        Constante de tiempo de la dinámica de estado (por defecto 1.0).
    plastic : bool
        Si esta caja usa la actualización de pesos por regla delta (por defecto False).
    lambda_rate : float
        Tasa de aprendizaje λ para cajas plásticas (por defecto 0.05).

    Nota sobre US (estímulo incondicionado):
        Para cajas plásticas, el US ya NO es un atributo escalar de la caja
        sino una señal viva alimentada por conexiones de tipo 'us' desde otras
        cajas (ver Conexion.VALID_TYPES). El valor efectivo de US en cada paso
        se calcula como el promedio de las entradas 'us' presentes, o 0.0 si
        la caja no tiene conexiones 'us' entrantes (en cuyo caso no aprende).
    """

    _PHI = 1.0 / np.arctan(1.0)   # factor de escala φ en la Ecuación (2)

    def __init__(
        self,
        name: str,
        alpha: float = 0.0,
        beta: ParamFloat = 1.0,
        gamma: ParamFloat = 1.0,
        epsilon: float = 1.0,
        plastic: bool = False,
        lambda_rate: float = 0.05,
    ):
        self.name = name
        self.alpha = alpha
        # beta y gamma pueden llegar como 'auto' aquí; el sistema los resolverá
        # a 1/N o 1/M después de registrar todas las conexiones (ver
        # JohanssonBalkeniusSystem._resolve_auto_params). Se conserva el valor
        # tal cual hasta entonces para no perder la intención del usuario.
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.plastic = plastic
        self.lambda_rate = lambda_rate
        # US ya no es atributo: viene de conexiones 'us' en cada paso.

        # Estado y salida
        self.x: float = 0.0
        self.o: float = self._output(0.0)

        # Pesos para entradas excitatorias (se establecen cuando se registran conexiones)
        # clave: índice de conexión, valor: peso
        self._weights: Dict[int, float] = {}

        # Buffer de salida (gestionado externamente vía push_output)
        # pero cada caja posee sus propios objetos Conexion como fuente
        # El sistema mantiene referencias a esas conexiones.

    # ------------------------------------------------------------------
    # Función de salida  — Ecuación (2)
    # ------------------------------------------------------------------

    def _output(self, x: float) -> float:
        return self._PHI * np.arctan(x)

    # ------------------------------------------------------------------
    # Derivada de estado  — Ecuación (1)
    # ------------------------------------------------------------------

    def _dxdt(
        self,
        x: float,
        E: List[Tuple[float, float]],   # lista de (valor, peso)
        I: List[float],
        S: float,
    ) -> float:
        """
        Evalúa dx/dt para un estado x dado y entradas fijas.

        Parámetros
        ----------
        x : float
            Valor de estado actual (o de prueba).
        E : lista de (valor, peso)
            Entradas excitatorias con sus pesos.
        I : lista de float
        S : float
            Valor de inhibición shunting.

        Retorna
        -------
        float
            dx/dt
        """
        exc = sum(v * w for v, w in E)
        inh = sum(I)
        shunt_factor = 1.0 / (1.0 + S) if S >= 0 else 1.0

        # variacion = constante + excitacion - inhibicion - estado_actual
        dxdt = self.alpha + self.beta * shunt_factor * exc - self.gamma * inh - x
        return dxdt / self.epsilon

    # ------------------------------------------------------------------
    # Integración RK4
    # ------------------------------------------------------------------

    def _rk4(
        self,
        x: float,
        E: List[Tuple[float, float]],
        I: List[float],
        S: float,
        dt: float,
    ) -> float:
        """
        Avanza el estado x un paso dt usando RK4.
        E, I, S se mantienen fijos durante el paso (motivado biológicamente).
        """
        f = lambda xv: self._dxdt(xv, E, I, S)
        k1 = f(x)
        k2 = f(x + dt / 2 * k1)
        k3 = f(x + dt / 2 * k2)
        k4 = f(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # ------------------------------------------------------------------
    # Inicialización de pesos
    # ------------------------------------------------------------------

    def init_weights(self, n_excitatory: int) -> None:
        """
        Inicializa los pesos excitatorios.

        Política según Johansson & Balkenius (2018, Sección 3):
            "The excitatory weights wi are set to 1 for all non-plastic
             connections, that is, everywhere except for AMY and CB."

        - Cajas NO plásticas: w_i = 1.0 (peso fijo, sin aprendizaje).
        - Cajas plásticas (AMY, CB):  w_i = 0.0 (sin entrenamiento previo;
          la regla delta de la ecuación 3 los hará crecer cuando US > o).

        Esto evita que cajas plásticas emitan respuestas espurias antes de
        cualquier condicionamiento, lo que sería incongruente con el modelo
        del paper (p.ej., AMY no debe reaccionar al CS antes de aprender la
        asociación CS→US).

        Llamado por el sistema después de que todas las conexiones se registran.
        """
        # initial_w = 0.0 if self.plastic else 1.0
        initial_w = 1.0  # Todos los pesos inician en 1.0
        self._weights = {i: initial_w for i in range(n_excitatory)}

    # ------------------------------------------------------------------
    # Actualización principal  — compute_and_commit
    # ------------------------------------------------------------------

    def compute_and_commit(
        self,
        entradas: Dict,
        dt: float,
    ) -> None:
        """
        Dada una instantánea fija de entradas, avanza la caja un paso.

        Estructura del dict entradas:
            {
                'E': [val0, val1, ...],   # valores de entrada excitatoria
                'I': [val0, val1, ...],   # valores de entrada inhibitoria
                'S': float,               # valor de inhibición shunting
                'U': [val0, val1, ...]    # valores del canal US (regla delta)
            }

        Pasos:
            1. Construir pares (valor, peso) para entradas excitatorias.
            2. Integrar x con RK4 usando la Ecuación (1).
            3. Actualizar pesos con regla delta si es plástico (Ecuación 3),
               usando US efectivo = promedio de entradas 'us'. Si la caja no
               tiene conexiones 'us' entrantes, US efectivo = 0 → no aprende.
            4. Confirmar nuevo x y calcular nuevo o.
        """
        E_vals = entradas.get('E', [])
        I_vals = entradas.get('I', [])
        S_val  = entradas.get('S', 0.0)
        U_vals = entradas.get('U', [])

        # US efectivo: promedio de entradas 'us' presentes; 0.0 si no hay.
        # Esto reemplaza el antiguo atributo self.US como escalar fijo.
        US_eff = sum(U_vals) / len(U_vals) if U_vals else 0.0

        # Construir pares excitatorios ponderados
        E_weighted = [
            (E_vals[i], self._weights.get(i, 1.0))
            for i in range(len(E_vals))
        ]

        # Paso RK4
        x_new = self._rk4(self.x, E_weighted, I_vals, S_val, dt)

        # Actualización de pesos para cajas plásticas — Ecuación (3)
        if self.plastic and US_eff > self.o:
            for i, e_val in enumerate(E_vals):
                dw = self.lambda_rate * e_val * (US_eff - self.o)
                self._weights[i] = self._weights.get(i, 1.0) + dt * dw

        # Confirmar
        self.x = x_new
        self.o = self._output(x_new)

    # ------------------------------------------------------------------
    # Push al buffer de salida  — llamado por el sistema en Fase 3
    # ------------------------------------------------------------------

    def push_output(self, t: float, connections: List['Conexion']) -> None:
        """
        Envía la salida actual o a todas las conexiones salientes.
        Llamado por el sistema después de que todas las cajas han confirmado.
        """
        for c in connections:
            if c.fuente is self:
                c.push(t, self.o)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"JohanssonBalkeniusBox(name='{self.name}', "
            f"x={self.x:.4f}, o={self.o:.4f}, "
            f"plastic={self.plastic})"
        )


# ---------------------------------------------------------------------------
# JohanssonBalkeniusSystem
# ---------------------------------------------------------------------------

# Configuración por defecto del sistema (topología completa de J&B 2018)
DEFAULT_EPSILON = 0.2
DEFAULT_TAU = 0.02
DEFAULT_GAMMA = 'auto'
DEFAULT_DT = 0.002
DEFAULT_CONFIG = {
    'dt': DEFAULT_DT,
    'boxes': [
        # ---- Entrada sensorial (controlada externamente vía alpha) ----
        {'name': 'left_retinae_l',    'alpha': 0.0,  'beta': 0, 'gamma': 0, 'epsilon': DEFAULT_EPSILON},
        {'name': 'left_retinae_r',    'alpha': 0.0,  'beta': 0, 'gamma': 0, 'epsilon': DEFAULT_EPSILON},
        {'name': 'right_retinae_l',   'alpha': 0.0,  'beta': 0, 'gamma': 0, 'epsilon': DEFAULT_EPSILON},
        {'name': 'right_retinae_r',   'alpha': 0.0,  'beta': 0, 'gamma': 0, 'epsilon': DEFAULT_EPSILON},
        {'name': 'cortex_excitatory', 'alpha': 0.0,  'beta': 0, 'gamma': 0, 'epsilon': DEFAULT_EPSILON},
        {'name': 'cortex_emotional',  'alpha': 0.0,  'beta': 0, 'gamma': 0, 'epsilon': DEFAULT_EPSILON},
        {'name': 'cortex_novelty',    'alpha': 0.0,  'beta': 0, 'gamma': 0, 'epsilon': DEFAULT_EPSILON},

        # ---- Vía parasimpática (Tabla A1: α=0, β=1/N, γ=1/M) ----
        {'name': 'PTA_l',  'alpha': 0.0,  'beta': 'auto', 'gamma': DEFAULT_GAMMA, 'epsilon': DEFAULT_EPSILON},
        {'name': 'PTA_r',  'alpha': 0.0,  'beta': 'auto', 'gamma': DEFAULT_GAMMA, 'epsilon': DEFAULT_EPSILON},
        {'name': 'EWpg_l', 'alpha': 0.0,  'beta': 'auto', 'gamma': DEFAULT_GAMMA, 'epsilon': DEFAULT_EPSILON},
        {'name': 'EWpg_r', 'alpha': 0.0,  'beta': 'auto', 'gamma': DEFAULT_GAMMA, 'epsilon': DEFAULT_EPSILON},
        {'name': 'CG_l',   'alpha': 0.0,  'beta': 'auto', 'gamma': DEFAULT_GAMMA, 'epsilon': DEFAULT_EPSILON},
        {'name': 'CG_r',   'alpha': 0.0,  'beta': 'auto', 'gamma': DEFAULT_GAMMA, 'epsilon': DEFAULT_EPSILON},

        # ---- Predictor cerebellar lateralizado (nodo restador D3) ----
        {'name': 'EWpg_pred_l', 'alpha': 0.0, 'beta': 1.0, 'gamma': 1.0, 'epsilon': DEFAULT_EPSILON},
        {'name': 'EWpg_pred_r', 'alpha': 0.0, 'beta': 1.0, 'gamma': 1.0, 'epsilon': DEFAULT_EPSILON},

        # ---- Vía simpática (Tabla A1) ----
        {'name': 'IML_l',  'alpha': 0.0,  'beta': 'auto', 'gamma': DEFAULT_GAMMA, 'epsilon': DEFAULT_EPSILON},
        {'name': 'IML_r',  'alpha': 0.0,  'beta': 'auto', 'gamma': DEFAULT_GAMMA, 'epsilon': DEFAULT_EPSILON},
        {'name': 'SCG_l',  'alpha': 0.0,  'beta': 'auto', 'gamma': DEFAULT_GAMMA, 'epsilon': DEFAULT_EPSILON},
        {'name': 'SCG_r',  'alpha': 0.0,  'beta': 'auto', 'gamma': DEFAULT_GAMMA, 'epsilon': DEFAULT_EPSILON},

        # ---- Hipotálamo (Tabla A1: α=0 salvo PVN; β=1/N, γ=1/M) ----
        {'name': 'VLPO',   'alpha': 0.0,  'beta': 'auto', 'gamma': DEFAULT_GAMMA, 'epsilon': DEFAULT_EPSILON},
        {'name': 'SCN',    'alpha': 0.0,  'beta': 'auto', 'gamma': DEFAULT_GAMMA, 'epsilon': DEFAULT_EPSILON},
        {'name': 'LH',     'alpha': 0.0,  'beta': 'auto', 'gamma': DEFAULT_GAMMA, 'epsilon': DEFAULT_EPSILON},
        {'name': 'DMH',    'alpha': 0.0,  'beta': 'auto', 'gamma': DEFAULT_GAMMA, 'epsilon': DEFAULT_EPSILON},
        {'name': 'PVN',    'alpha': 1.0,  'beta': 'auto', 'gamma': 1.0, 'epsilon': 5*DEFAULT_DT},

        # ---- PONS (Tabla A1: LC con α=0, β=0.05, γ=1/M) ----
        {'name': 'LC_l',   'alpha': 0.0,  'beta': 0.05,   'gamma': 1.0, 'epsilon': DEFAULT_EPSILON},
        {'name': 'LC_r',   'alpha': 0.0,  'beta': 0.05,   'gamma': 1.0, 'epsilon': DEFAULT_EPSILON},

        # ---- AMÍGDALA (Tabla A1: α=0, β=1.0, γ=1/M; plástica con λ=0.05) ----
        {'name': 'AMY',    'alpha': 0.0,  'beta': 1.0,    'gamma': 1.0, 'epsilon': DEFAULT_EPSILON,
         'plastic': True,  'lambda_rate': 0.05},

        # ---- CEREBELO (Tabla A1: α=0, β=1.0, γ=1/M; plástica con λ=0.1) ----
        {'name': 'CB',     'alpha': 0.0,  'beta': 1.0,    'gamma': 1.0, 'epsilon': DEFAULT_EPSILON,
         'plastic': True,  'lambda_rate': 0.1},

    ],
    'connections': [
        # Cadena parasimpática
        {'from': 'left_retinae_l',  'to': 'PTA_l',  'tipo': 'excitatory', 'tau': DEFAULT_TAU},
        {'from': 'left_retinae_r',  'to': 'PTA_r',  'tipo': 'excitatory', 'tau': DEFAULT_TAU},
        {'from': 'right_retinae_l', 'to': 'PTA_l',  'tipo': 'excitatory', 'tau': DEFAULT_TAU},
        {'from': 'right_retinae_r', 'to': 'PTA_r',  'tipo': 'excitatory', 'tau': DEFAULT_TAU},

        {'from': 'PTA_l',   'to': 'EWpg_l', 'tipo': 'excitatory', 'tau': DEFAULT_TAU},
        {'from': 'PTA_l',   'to': 'EWpg_r', 'tipo': 'excitatory', 'tau': DEFAULT_TAU},
        {'from': 'PTA_r',   'to': 'EWpg_l', 'tipo': 'excitatory', 'tau': DEFAULT_TAU},
        {'from': 'PTA_r',   'to': 'EWpg_r', 'tipo': 'excitatory', 'tau': DEFAULT_TAU},

        {'from': 'EWpg_l',  'to': 'CG_l',   'tipo': 'excitatory', 'tau': DEFAULT_TAU},
        {'from': 'EWpg_r',  'to': 'CG_r',   'tipo': 'excitatory', 'tau': DEFAULT_TAU},

        {'from': 'EWpg_l',  'to': 'CB',     'tipo': 'us', 'tau': DEFAULT_TAU},
        {'from': 'EWpg_r',  'to': 'CB',     'tipo': 'us', 'tau': DEFAULT_TAU},

        # Cadena cortical
        {'from': 'cortex_excitatory', 'to': 'CB',     'tipo': 'excitatory', 'tau': DEFAULT_TAU},
        {'from': 'cortex_excitatory', 'to': 'AMY',    'tipo': 'excitatory', 'tau': DEFAULT_TAU},
        {'from': 'cortex_novelty',    'to': 'AMY',    'tipo': 'us',         'tau': 10*DEFAULT_TAU},
        {'from': 'cortex_emotional',  'to': 'LC_l',   'tipo': 'excitatory', 'tau': DEFAULT_TAU},
        {'from': 'cortex_emotional',  'to': 'LC_r',   'tipo': 'excitatory', 'tau': DEFAULT_TAU},

        # Loop CB - EWpg vía predictor lateralizado
        {'from': 'CB',          'to': 'EWpg_pred_l', 'tipo': 'excitatory', 'tau': DEFAULT_TAU},
        {'from': 'CB',          'to': 'EWpg_pred_r', 'tipo': 'excitatory', 'tau': DEFAULT_TAU},
        {'from': 'EWpg_l',      'to': 'EWpg_pred_l', 'tipo': 'inhibitory', 'tau': DEFAULT_TAU},
        {'from': 'EWpg_r',      'to': 'EWpg_pred_r', 'tipo': 'inhibitory', 'tau': DEFAULT_TAU},
        {'from': 'EWpg_pred_l', 'to': 'EWpg_l',      'tipo': 'excitatory', 'tau': DEFAULT_TAU},
        {'from': 'EWpg_pred_r', 'to': 'EWpg_r',      'tipo': 'excitatory', 'tau': DEFAULT_TAU},

        # Cadena amigdalar
        {'from': 'AMY',   'to': 'LC_l',    'tipo': 'excitatory', 'tau': DEFAULT_TAU},
        {'from': 'AMY',   'to': 'LC_r',    'tipo': 'excitatory', 'tau': DEFAULT_TAU},

        # Cadena PONS
        {'from': 'LC_l',  'to': 'IML_l',   'tipo': 'excitatory', 'tau': DEFAULT_TAU},
        {'from': 'LC_r',  'to': 'IML_r',   'tipo': 'excitatory', 'tau': DEFAULT_TAU},
        {'from': 'LC_l',  'to': 'EWpg_l',  'tipo': 'shunting',   'tau': DEFAULT_TAU},
        {'from': 'LC_r',  'to': 'EWpg_r',  'tipo': 'shunting',   'tau': DEFAULT_TAU},

        # Cadena Hypothalamus
        {'from': 'left_retinae_l',  'to': 'VLPO',  'tipo': 'excitatory', 'tau': DEFAULT_TAU},
        {'from': 'left_retinae_r',  'to': 'VLPO',  'tipo': 'excitatory', 'tau': DEFAULT_TAU},
        {'from': 'right_retinae_l', 'to': 'VLPO',  'tipo': 'excitatory', 'tau': DEFAULT_TAU},
        {'from': 'right_retinae_r', 'to': 'VLPO',  'tipo': 'excitatory', 'tau': DEFAULT_TAU},

        {'from': 'left_retinae_l',  'to': 'SCN',   'tipo': 'excitatory', 'tau': DEFAULT_TAU},
        {'from': 'left_retinae_r',  'to': 'SCN',   'tipo': 'excitatory', 'tau': DEFAULT_TAU},
        {'from': 'right_retinae_l', 'to': 'SCN',   'tipo': 'excitatory', 'tau': DEFAULT_TAU},
        {'from': 'right_retinae_r', 'to': 'SCN',   'tipo': 'excitatory', 'tau': DEFAULT_TAU},

        {'from': 'SCN',   'to': 'DMH',   'tipo': 'excitatory', 'tau': DEFAULT_TAU},
        {'from': 'SCN',   'to': 'PVN',   'tipo': 'inhibitory', 'tau': DEFAULT_TAU},
        {'from': 'DMH',   'to': 'LH',    'tipo': 'excitatory', 'tau': DEFAULT_TAU},

        {'from': 'VLPO',  'to': 'LC_l',  'tipo': 'inhibitory', 'tau': DEFAULT_TAU},
        {'from': 'VLPO',  'to': 'LC_r',  'tipo': 'inhibitory', 'tau': DEFAULT_TAU},
        {'from': 'DMH',   'to': 'LC_l',  'tipo': 'excitatory', 'tau': DEFAULT_TAU},
        {'from': 'DMH',   'to': 'LC_r',  'tipo': 'excitatory', 'tau': DEFAULT_TAU},
        {'from': 'LH',    'to': 'LC_l',  'tipo': 'excitatory', 'tau': DEFAULT_TAU},
        {'from': 'LH',    'to': 'LC_r',  'tipo': 'excitatory', 'tau': DEFAULT_TAU},

        {'from': 'PVN',   'to': 'IML_l', 'tipo': 'excitatory', 'tau': DEFAULT_TAU},
        {'from': 'PVN',   'to': 'IML_r', 'tipo': 'excitatory', 'tau': DEFAULT_TAU},

        # Cadena simpática
        {'from': 'IML_l',   'to': 'SCG_l',   'tipo': 'excitatory', 'tau': DEFAULT_TAU},
        {'from': 'IML_r',   'to': 'SCG_r',   'tipo': 'excitatory', 'tau': DEFAULT_TAU},
    ]
}

class JohanssonBalkeniusSystem:
    """
    Sistema completo interconectado de control pupilar de Johansson & Balkenius.

    Implementa el esquema de actualización síncrona:
        Fase 1 — recolectar entradas para TODAS las cajas desde buffers de retraso
        Fase 2 — compute_and_commit para TODAS las cajas (sin tocar buffers)
        Fase 3 — enviar salidas a todos los buffers de conexión salientes
        Fase 4 — registrar historial, avanzar tiempo

    Parámetros
    ----------
    system_config : dict | str
        Diccionario de configuración con la siguiente estructura, o el string "default"
        para usar la configuración predefinida completa de J&B 2018 (topología bilateral
        con todas las vías: parasimpática, simpática, cortical, amigdalar, cerebelar,
        hipotalámica y PONS).

        Estructura del diccionario:
        {
            'dt': float,          # paso de integración en segundos (ej. 0.020)
            'boxes': [
                {
                    'name': str,
                    'alpha': float,
                    'beta': float | 'auto',
                    'gamma': float | 'auto',
                    'epsilon': float,
                    'plastic': bool,
                    'lambda_rate': float,
                    # Nota: el campo 'US' fue removido. Ahora el US para cajas
                    # plásticas viene de conexiones tipo 'us' (ver más abajo).
                },
                ...
            ],
            'connections': [
                {
                    'from': str,        # nombre de caja fuente
                    'to': str,          # nombre de caja destino
                    'tipo': str,        # 'excitatory'|'inhibitory'|'shunting'|'us'
                    'tau': float,       # retraso en segundos
                },
                ...
            ]
        }
    """

    def __init__(self, system_config: Union[Dict, str]):
        # Si system_config es "default", usar la configuración predefinida
        if isinstance(system_config, str) and system_config == "default":
            system_config = DEFAULT_CONFIG
        
        self.dt: float = system_config.get('dt', 0.020)
        self.t: float = 0.0

        # Construir cajas
        # Nota: el parámetro 'US' del config ya no se acepta — el US viene
        # ahora de conexiones 'us' (ver tipo de conexión en Conexion).
        self.cajas: Dict[str, JohanssonBalkeniusBox] = {}
        for bcfg in system_config.get('boxes', []):
            box = JohanssonBalkeniusBox(
                name         = bcfg['name'],
                alpha        = bcfg.get('alpha', 0.0),
                beta         = bcfg.get('beta', 'auto'),
                gamma        = bcfg.get('gamma', 'auto'),
                epsilon      = bcfg.get('epsilon', 1.0),
                plastic      = bcfg.get('plastic', False),
                lambda_rate  = bcfg.get('lambda_rate', 0.05),
            )
            self.cajas[bcfg['name']] = box

        # Construir conexiones
        self.conexiones: List[Conexion] = []
        # Mapa: nombre de caja destino -> lista de Conexion
        self._incoming: Dict[str, List[Conexion]] = {n: [] for n in self.cajas}
        # Mapa: nombre de caja fuente -> lista de Conexion
        self._outgoing: Dict[str, List[Conexion]] = {n: [] for n in self.cajas}

        for ccfg in system_config.get('connections', []):
            src_name = ccfg['from']
            dst_name = ccfg['to']
            fuente   = self.cajas[src_name]
            c = Conexion(
                fuente         = fuente,
                tipo           = ccfg.get('tipo', 'excitatory'),
                tau            = ccfg.get('tau', self.dt),
                resting_output = fuente.o,
            )
            self.conexiones.append(c)
            self._incoming[dst_name].append(c)
            self._outgoing[src_name].append(c)

        # Resolver parámetros 'auto' a partir de la topología recién construida.
        # Esto debe ocurrir ANTES de init_weights, aunque hoy ese orden no es
        # estrictamente necesario porque init_weights no consulta beta/gamma.
        self._resolve_auto_params()

        # Inicializar pesos para cada caja basado en el número de entradas excitatorias
        for name, box in self.cajas.items():
            n_exc = sum(1 for c in self._incoming[name] if c.tipo == 'excitatory')
            box.init_weights(n_exc)

        # Historial: {nombre_caja: [(t, x, o), ...]}
        self.historia: Dict[str, List[Tuple[float, float, float]]] = {
            n: [] for n in self.cajas
        }

        # Mapeos de input/output del sistema
        # input_name -> [box_names] : cajas cuyo alpha se modifica por este input
        self._input_mapping: Dict[str, List[str]] = {}
        # output_name -> ([box_names], function) : combina salidas de cajas
        self._output_mapping: Dict[str, Tuple[List[str], Callable]] = {}

        # Registrar estado inicial
        self._record_history()

    # ------------------------------------------------------------------
    # Resolución de parámetros 'auto' (β = 1/N, γ = 1/M)
    # ------------------------------------------------------------------
    def _resolve_auto_params(self) -> None:
        """
        Convierte beta='auto' / gamma='auto' a valores numéricos según la
        topología de conexiones, replicando la convención por defecto del
        paper J&B 2018: β = 1/N (excitatorias entrantes), γ = 1/M (inhibitorias
        entrantes). Si N (o M) es 0, se fuerza el valor 1.0 silenciosamente
        para evitar división por cero — esto aplica típicamente a cajas
        sensoriales puramente controladas por alpha.
        """
        for name, box in self.cajas.items():
            n_exc = sum(1 for c in self._incoming[name] if c.tipo == 'excitatory')
            n_inh = sum(1 for c in self._incoming[name] if c.tipo == 'inhibitory')

            if isinstance(box.beta, str) and box.beta == 'auto':
                box.beta = 1.0 / max(1, n_exc)
            if isinstance(box.gamma, str) and box.gamma == 'auto':
                box.gamma = 1.0 / max(1, n_inh)

    # ------------------------------------------------------------------
    # Paso único de simulación
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Avanza la simulación un dt."""

        # FASE 1: recolectar entradas para TODAS las cajas desde buffers
        entradas_snapshot: Dict[str, Dict] = {}
        for name, box in self.cajas.items():
            E, I, S, U = [], [], 0.0, []
            for c in self._incoming[name]:
                val = c.get_delayed_output(self.t)
                if c.tipo == 'excitatory':
                    E.append(val)
                elif c.tipo == 'inhibitory':
                    I.append(val)
                elif c.tipo == 'shunting':
                    S += val   # entradas shunting se acumulan
                elif c.tipo == 'us':
                    U.append(val)   # entradas US se promedian (ver compute_and_commit)
            entradas_snapshot[name] = {'E': E, 'I': I, 'S': S, 'U': U}

        # FASE 2: compute_and_commit para TODAS las cajas
        # No se toca ningún buffer aquí — sincronía estricta
        for name, box in self.cajas.items():
            box.compute_and_commit(entradas_snapshot[name], self.dt)

        # FASE 3: enviar salidas a buffers de conexión salientes
        for name, box in self.cajas.items():
            box.push_output(self.t, self._outgoing[name])

        # FASE 4: registrar historial y avanzar tiempo
        self._record_history()
        self.t += self.dt

    # ------------------------------------------------------------------
    # Ejecutar por t_total segundos
    # ------------------------------------------------------------------

    def generate(self, t_total: float) -> Dict[str, List[Tuple[float, float, float]]]:
        """
        Ejecuta la simulación por t_total segundos.

        Parámetros
        ----------
        t_total : float
            Duración total de simulación en segundos.

        Retorna
        -------
        dict
            Diccionario de historial: {nombre_caja: [(t, x, o), ...]}.
        """
        n_steps = int(round(t_total / self.dt))
        for _ in range(n_steps):
            self.step()
        return self.historia

    # ------------------------------------------------------------------
    # Registro de historial
    # ------------------------------------------------------------------

    def _record_history(self) -> None:
        for name, box in self.cajas.items():
            self.historia[name].append((self.t, box.x, box.o))

    # ------------------------------------------------------------------
    # Auxiliares de acceso en tiempo de ejecución
    # ------------------------------------------------------------------

    def get_box(self, name: str) -> JohanssonBalkeniusBox:
        """Retorna una caja por nombre para modificación de parámetros en tiempo de ejecución."""
        return self.cajas[name]

    # Definición de inputs del sistema
    # ------------------------------------------------------------------

    def define_input(self, name: str, box_names: List[str]) -> None:
        """
        Define un input del sistema que mapea a cajas específicas.

        El input modifica el parámetro alpha de las cajas especificadas.

        Args:
            name: Nombre del input (ej. 'retina_left').
            box_names: Lista de nombres de cajas cuyo alpha se modificará.
        """
        for box_name in box_names:
            if box_name not in self.cajas:
                raise ValueError(f"Caja '{box_name}' no existe en el sistema")
        self._input_mapping[name] = box_names

    def set_input(self, name: str, value: float) -> None:
        """
        Establece el valor de un input, actualizando alpha de las cajas asociadas.

        Args:
            name: Nombre del input definido previamente.
            value: Valor a asignar al parámetro alpha de las cajas.
        """
        if name not in self._input_mapping:
            raise ValueError(f"Input '{name}' no definido. Usar define_input primero.")
        for box_name in self._input_mapping[name]:
            self.cajas[box_name].alpha = value

    # Definición de outputs del sistema
    # ------------------------------------------------------------------

    def define_output(self, name: str, box_names: List[str], function: Callable) -> None:
        """
        Define un output del sistema que combina salidas de cajas.

        Args:
            name: Nombre del output (ej. 'pupil_left').
            box_names: Lista de nombres de cajas cuyas salidas se combinarán.
            function: Función que toma las salidas de las cajas y retorna un valor.
                      El orden de argumentos debe coincidir con box_names.
        """
        for box_name in box_names:
            if box_name not in self.cajas:
                raise ValueError(f"Caja '{box_name}' no existe en el sistema")
        self._output_mapping[name] = (box_names, function)

    def get_output(self, name: str) -> float:
        """
        Obtiene el valor de un output combinando salidas de las cajas.

        Args:
            name: Nombre del output definido previamente.

        Returns:
            Valor calculado por la función del output.
        """
        if name not in self._output_mapping:
            raise ValueError(f"Output '{name}' no definido. Usar define_output primero.")
        box_names, func = self._output_mapping[name]
        values = [self.cajas[bn].o for bn in box_names]
        return func(*values)

    # set_US fue removido: el US ahora se entrega vía conexiones 'us'.
    # Para inyectar US sintético en un test, declarar una caja fuente con
    # alpha=valor_deseado y conectarla con tipo='us' a la caja plástica.

    def set_all_epsilon(self, epsilon: float) -> None:
        """
        Establece el parámetro epsilon para todas las cajas del sistema.
        
        Args:
            epsilon (float): Nuevo valor de epsilon para todas las cajas.
        """
        for caja in self.cajas.values():
            caja.epsilon = epsilon

    def set_all_tau(self, tau: float) -> None:
        """
        Establece el parámetro tau para todas las conexiones del sistema.
        
        Args:
            tau (float): Nuevo valor de tau para todas las conexiones (retraso en segundos).
        """
        for conexion in self.conexiones:
            conexion.tau = tau

    def __repr__(self) -> str:
        return (
            f"JohanssonBalkeniusSystem(dt={self.dt}, "
            f"n_cajas={len(self.cajas)}, "
            f"n_conexiones={len(self.conexiones)})"
        )


# ---------------------------------------------------------------------------
# DefaultJBS - Clase auxiliar para configuraciones estándar del sistema
# ---------------------------------------------------------------------------

class DefaultJBS:
    """
    Clase auxiliar que encapsula funciones comunes para el modelo estándar
    de Johansson & Balkenius (2018).

    Proporciona funciones reutilizables para:
    - Modulación óptica del estímulo retinal (closed-loop)
    - Cálculo del diámetro pupilar a partir de salidas nucleares
    """

    # Límites del diámetro pupilar (mm)
    D_MIN = 2.0
    D_MAX = 8.0
    D_REF = 5.0  # diámetro pupilar normal (mm) usado como referencia

    @staticmethod
    def optical_input(L: float, d: float) -> float:
        """
        Modulación óptica del estímulo retinal por la apertura pupilar.

        Modelo cuadrático normalizado: el flujo lumínico que llega a la
        retina es proporcional al área de la apertura pupilar (A = π·d²/4),
        normalizada por un diámetro de referencia (pupila normal).

            α_retina = L · (d / D_REF)²

        Esto cierra el loop óptico-mecánico que falta en J&B 2018: cuando la
        pupila se constriñe, menos luz alcanza la retina, lo cual reduce el
        drive parasimpático y permite que el sistema converja a un equilibrio
        estable en lugar de saturar.

        Args:
            L (float): Luminancia de la fuente (u.a., 0-1).
            d (float): Diámetro pupilar actual (mm).

        Returns:
            float: Entrada efectiva a la retina (u.a.).
        """
        return L * (d / DefaultJBS.D_REF) ** 2

    @staticmethod
    def pupil_diameter(cg_o: float, scg_o: float) -> float:
        """
        Calcula el diámetro pupilar en función de las salidas de CG y SCG.

        Modelo con rangos dinámicos:
            d_basal es el diámetro cuando no hay activación (5.0 mm)
            rango_constriccion = d_basal - D_MIN  # Cuánto puede cerrar (3.0 mm)
            rango_dilatacion = D_MAX - d_basal    # Cuánto puede abrir (3.0 mm)
            d = d_basal - (rango_constriccion * cg_o) + (rango_dilatacion * scg_o)

        Args:
            cg_o (float): Salida del Ganglio Ciliar (esfínter, parasimpático, 0-1).
            scg_o (float): Salida del Ganglio Cervical Superior (dilatador, simpático, 0-1).

        Returns:
            float: Diámetro pupilar (mm), limitado a [D_MIN, D_MAX].
        """
        d_basal = (DefaultJBS.D_MAX + DefaultJBS.D_MIN) / 2
        rango_constriccion = d_basal - DefaultJBS.D_MIN
        rango_dilatacion = DefaultJBS.D_MAX - d_basal
        d = d_basal - (rango_constriccion * cg_o) + (rango_dilatacion * scg_o)
        return float(np.clip(d, DefaultJBS.D_MIN, DefaultJBS.D_MAX))

    @staticmethod
    def setup_standard_inputs(system: JohanssonBalkeniusSystem) -> None:
        """
        Configura los inputs estándar del sistema para un test básico.

        Define inputs para:
        - Retinas (izquierda y derecha)
        - Corteza (excitatory, emotional, novelty)

        Args:
            system: Instancia de JohanssonBalkeniusSystem.
        """
        system.define_input("retina_left", ['left_retinae_l', 'left_retinae_r'])
        system.define_input("retina_right", ['right_retinae_l', 'right_retinae_r'])
        system.define_input("cortex_excitatory", ['cortex_excitatory'])
        system.define_input("cortex_emotional", ['cortex_emotional'])
        system.define_input("cortex_novelty", ['cortex_novelty'])

    @staticmethod
    def setup_standard_outputs(system: JohanssonBalkeniusSystem) -> None:
        """
        Configura los outputs estándar del sistema para un test básico.

        Define outputs para:
        - Diámetro pupilar izquierdo
        - Diámetro pupilar derecho

        Args:
            system: Instancia de JohanssonBalkeniusSystem.
        """
        system.define_output("pupil_left", ['CG_l', 'SCG_l'], DefaultJBS.pupil_diameter)
        system.define_output("pupil_right", ['CG_r', 'SCG_r'], DefaultJBS.pupil_diameter)

    @staticmethod
    def set_cortical_baseline(system: JohanssonBalkeniusSystem, 
                             excitatory: float = 0.0,
                             emotional: float = 0.0,
                             novelty: float = 0.0) -> None:
        """
        Establece los inputs corticales en valores de línea base.

        Útil para tests de reflejo autonómico puro (sin activación cortical).

        Args:
            system: Instancia de JohanssonBalkeniusSystem.
            excitatory: Valor para cortex_excitatory (default 0.0).
            emotional: Valor para cortex_emotional (default 0.0).
            novelty: Valor para cortex_novelty (default 0.0).
        """
        system.set_input("cortex_excitatory", excitatory)
        system.set_input("cortex_emotional", emotional)
        system.set_input("cortex_novelty", novelty)