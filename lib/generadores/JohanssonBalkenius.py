"""
Johansson & Balkenius (2018) - Modelo computacional de dilatación pupilar
Implementación del modelo conexionista a nivel de sistema de control pupilar.

Clases:
    Conexion                  - Conexión sináptica con buffer de retraso
    JohanssonBalkeniusBox     - Núcleo individual / región cerebral
    JohanssonBalkeniusSystem  - Sistema interconectado completo
"""

import numpy as np
import bisect
from typing import Dict, List, Optional, Tuple, Union, Callable

# Tipo auxiliar: parámetros β y γ pueden ser un float fijo o el centinela 'auto'
# para que el sistema los resuelva como 1/N (excitatorias) o 1/M (inhibitorias)
# según la topología, siguiendo el comportamiento por defecto del paper de
# Johansson & Balkenius (2018).
ParamFloat = Union[float, str]


# ---------------------------------------------------------------------------
# Conexion

class Conexion:
    """
    Conexión sináptica entre dos cajas.

    Mantiene un buffer de historial para implementar el retraso de transmisión τ.
    Interpola linealmente entre muestras del buffer para recuperar la
    salida retrasada en cualquier tiempo solicitado.

    Modelo unificado de perturbaciones de comunicación
    --------------------------------------------------
    Todas las perturbaciones que afectan a la transmisión presináptica se
    aplican en un único paso de salida (método `push`):

      1. Depleción del recurso sináptico (Tsodyks-Markram simplificado),
         mediante el estado interno `u ∈ [0, 1]`:

            du/dt = (1 − u) / τ_rec   −   U · u · max(o_fuente, 0)
                    └─ recuperación   └─ depleción por uso

      2. Ruido estocástico multiplicativo escalado por el agotamiento (1−u):

            o_noisy = o · (1 + σ · (1−u) · N(0,1))

      3. Jitter temporal en el timestamp con que se almacena la muestra,
         también escalado por el agotamiento:

            δ_t = τ · τ_jitter · (1−u) · N(0,1)
            t_eff = t + max(δ_t, −τ)        # clamp: retraso efectivo ≥ 0

         (El clamp evita lecturas del futuro al impedir que la combinación
         de jitter negativo y τ pequeño produzca retraso aparente negativo.)

    La señal efectivamente almacenada en el buffer es `u · o_noisy` con
    timestamp `t_eff`. La lectura (`get_delayed_output`) es entonces un puro
    interpolador lineal sobre el buffer y no aplica ninguna perturbación.

    Cada perturbación actúa como un "flag" desactivable: con `U = 0`,
    `σ = 0` y `τ_jitter = 0` (defaults) se recupera exactamente el modelo
    JB ideal — el push se reduce a almacenar `(t, o)` sin modificar.

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
                         efectivo se calcula como el MÁXIMO de ellas. La
                         interpretación: el US es una señal de teaching/dominio,
                         y basta con que una de las fuentes esté activa para
                         reclutar plasticidad — en oposición al promedio, que
                         atenuaría el aprendizaje cuando una vía permanece
                         silente (caso típico del cerebelo lateralizado, donde
                         sólo la fibra trepadora del lado iluminado dispara).
                         En cajas no plásticas, las conexiones 'us' son no-op.
    tau : float
        Retraso de transmisión en segundos (ej. 0.020 para 20 ms).
    resting_output : float
        Valor usado para pre-llenar el buffer antes de iniciar la simulación.
    buffer_margin : float
        Tiempo extra más allá de τ mantenido en el buffer (segundos). Por defecto 0.1 s.
    tau_rec : float
        Constante de tiempo de recuperación del recurso sináptico (segundos).
        Default 1.0 s. Valores grandes modelan fatiga/depleción lenta de
        recuperación; valores pequeños modelan sinapsis robustas.
    U : float
        Fracción del recurso utilizada por unidad de actividad presináptica.
        Default 0.0 (depleción desactivada → JB ideal). Valores típicos
        fisiológicos: 0.05–0.30 según tipo de sinapsis.
    u_initial : float
        Valor inicial de `u` ∈ [0, 1]. Default 1.0 (recurso lleno).
        Valores menores modelan estado pre-fatigado o lesión parcial.
    """

    VALID_TYPES = ('excitatory', 'inhibitory', 'shunting', 'us')  # tipos válidos de conexión

    def __init__(
        self,
        fuente: 'JohanssonBalkeniusBox',
        tipo: str,
        tau: float = 0.020,
        resting_output: float = 0.0,
        buffer_margin: float = 0.1,
        tau_rec: float = 1.0,
        U: float = 0.0,
        u_initial: float = 1.0,
        sigma: float = 0.0,
        tau_jitter: float = 0.0,
    ):
        if tipo not in self.VALID_TYPES:
            raise ValueError(f"tipo debe ser uno de {self.VALID_TYPES}, se obtuvo '{tipo}'")

        self.fuente = fuente
        self.tipo = tipo
        self.tau = tau

        # Jitter en el tiempo de transmisión (proporcional al agotamiento).
        # Aplicado en push() como offset al timestamp con que se almacena la
        # muestra (no en la lectura). El offset se clampa a δ_t ≥ −τ para
        # garantizar retraso efectivo ≥ 0 (sin lecturas del futuro).
        # Valores típicos: 0.05-0.2 (5-20% de variación efectiva en τ).
        self.tau_jitter = tau_jitter

        # Buffer margin con 99 de certeza de acomodar las muestras con jitter
        _Z99 = 2.576
        margin_min = self.tau * self.tau_jitter * _Z99
        self.buffer_margin = max(buffer_margin, margin_min)

        # Recurso sináptico (Tsodyks-Markram simplificado).
        # Aplica a TODOS los tipos de conexión (excitatory, inhibitory,
        # shunting, us) — incluida la fibra trepadora EWpg→CB tipo 'us'.
        # u = 1.0 → recurso lleno (sin agotamiento)
        # u = 0.0 → recurso agotado
        # (1-u) representa el nivel de agotamiento
        self.tau_rec = tau_rec
        self.U = U
        self.u = float(np.clip(u_initial, 0.0, 1.0))

        # Ruido estocástico sináptico (proporcional al agotamiento).
        # Aplicado en push() junto con la depleción.
        self.sigma = sigma

        # Buffer: lista de tuplas (t, o_efectivo) mantenida ordenada por
        # timestamp. Se usa lista (no deque) porque el jitter puede ocasionar
        # inserciones fuera de orden estricto, requiriendo bisect/insert.
        # Nota: se almacena u·o_noisy, NO o_fuente puro.
        self._buffer: List[Tuple[float, float]] = []

        # Pre-llenar buffer con resting_output. Como en t=0 se asume u=u_initial
        # y el sistema está en reposo (o_fuente ≈ resting_output), se prellena
        # con u_initial * resting_output para mantener consistencia.
        self._prefill(self.u * resting_output)

    # ------------------------------------------------------------------
    # Gestión del buffer
    # ------------------------------------------------------------------

    def _prefill(self, resting_output: float) -> None:
        """Llena el buffer con el valor de reposo para cubrir el retraso inicial."""
        t_start = -(self.tau + self.buffer_margin)
        # Dos puntos son suficientes: la interpolación lineal retornará resting_output
        self._buffer.append((t_start, resting_output))
        self._buffer.append((0.0, resting_output))

    def push(self, t: float, o: float, dt: float) -> None:
        """
        Único paso de salida que aplica TODAS las perturbaciones de
        comunicación presinápticas en orden:

            1. Actualiza el recurso sináptico u (depleción/recuperación).
            2. Aplica ruido estocástico multiplicativo: o_noisy.
            3. Multiplica por el recurso recién actualizado: u · o_noisy.
            4. Calcula un timestamp efectivo con jitter, clampado para que
               el retraso efectivo nunca sea negativo: t_eff = t + δ_t,
               con δ_t ≥ −τ.
            5. Inserta (t_eff, u·o_noisy) en el buffer manteniendo el
               orden por timestamp (caso común: append O(1); caso raro
               de jitter negativo grande: insert O(N)).
            6. Limpia muestras anteriores al horizonte τ + margin.

        Si U = σ = τ_jitter = 0 (defaults), los pasos 1, 2 y 4 son no-ops
        y push se reduce a un append directo de (t, o) — comportamiento
        JB ideal exacto.

        Parámetros
        ----------
        t : float
            Tiempo nominal de emisión.
        o : float
            Salida cruda actual de la caja fuente.
        dt : float
            Paso temporal del integrador (s). Necesario para integrar du/dt.
        """
        # 1. Depleción/recuperación del pool — actualiza self.u
        self.update_resource(dt, o)

        # 2-3. Ruido y depleción aplicados a la salida
        o_noisy = self.apply_stochastic_noise(o)

        o_eff = self.u * o_noisy

        # 4. Jitter del timestamp; clamp evita retraso efectivo negativo.
        if self.tau_jitter > 0.0:
            jitter_delta = (
                self.tau * self.tau_jitter * (1.0 - self.u)
                * np.random.normal(0.0, 1.0)
            )
            # τ es el retraso mínimo del canal:
                # el agotamiento solo agrega latencia.
                # Nota: al truncar N(0,1) en 0, jitter_delta
                # sigue una distribución semi-normal con media
                # positiva → el canal agotado transmite en
                # promedio más lento que τ (sesgo sistemático
                # por diseño).
            jitter_delta = max(jitter_delta, -self.tau) # proteccion de no causalidad
        else:
            jitter_delta = 0.0
        t_eff = t + jitter_delta

        # 5. Inserción ordenada (caso común: append; caso raro: insertar).
        if not self._buffer or t_eff >= self._buffer[-1][0]:
            self._buffer.append((t_eff, o_eff))
        else:
            # Búsqueda lineal desde el final — el jitter típico es pequeño,
            # así que la inserción cae cerca del extremo derecho.
            idx = len(self._buffer)
            while idx > 0 and self._buffer[idx - 1][0] > t_eff:
                idx -= 1
            self._buffer.insert(idx, (t_eff, o_eff))

        # 6. Limpieza de muestras obsoletas.
        cutoff = t - self.tau - self.buffer_margin
        while len(self._buffer) > 2 and self._buffer[0][0] < cutoff:
            self._buffer.pop(0)

    def update_resource(self, dt: float, source_activity: float) -> None:
        """
        Actualiza el recurso sináptico `u` según Tsodyks-Markram simplificado:

            du/dt = (1 − u) / τ_rec  −  U · u · max(source_activity, 0)

        Solo la actividad presináptica positiva consume el recurso (las salidas
        negativas del modelo, producto del arctan bipolar, NO deplecionan).

        Cortocircuito: si U == 0, no se hace ningún cómputo (el recurso queda
        fijo en su valor inicial), preservando comportamiento JB ideal.

        Nota: este método se llama internamente desde `push()` como parte del
        paso unificado de salida — no debe invocarse manualmente bajo el
        flujo estándar de `JohanssonBalkeniusSystem.step()`.

        Parámetros
        ----------
        dt : float
            Paso temporal (segundos).
        source_activity : float
            Salida actual de la caja fuente (típicamente self.fuente.o).
        """
        if self.U == 0.0:
            return  # depleción desactivada
        a = max(source_activity, 0.0)

        du = (1.0 - self.u) / self.tau_rec - self.U * self.u * a

        self.u = float(np.clip(self.u + dt * du, 0.0, 1.0))

    def apply_stochastic_noise(self, o: float) -> float:
        """
        Aplica ruido estocástico multiplicativo a la salida, escalado por el
        nivel de agotamiento (1-u). Modela variabilidad en la transmisión
        sináptica que aumenta con la depleción del pool vesicular.

        Si sigma == 0, retorna o sin modificación (comportamiento JB ideal).

        Parámetros
        ----------
        o : float
            Salida original de la caja fuente.

        Retorna
        -------
        float
            Salida con ruido aplicado: o * (1 + sigma * (1-u) * N(0,1)).
        """
        if self.sigma == 0.0:
            return o
        noise_factor = 1.0 + self.sigma * (1.0 - self.u) * np.random.normal(0.0, 1.0)
        return o * noise_factor

    def get_delayed_output(self, t: float) -> float:
        """
        Retorna la salida interpolada de la fuente en el tiempo (t − τ).

        Esta es una operación de PURA LECTURA: no aplica perturbaciones.
        Toda la variabilidad estocástica (depleción, ruido, jitter) se
        introduce en `push()` cuando la muestra entra al buffer.

        El buffer se mantiene ordenado por timestamp, pero el jitter puede
        causar inserciones fuera de orden temporal relativo al índice, por
        lo que se usa bisect para encontrar las muestras circundantes.

        Parámetros
        ----------
        t : float
            Tiempo actual.

        Retorna
        -------
        float
            Salida interpolada de la fuente en el tiempo t − τ.
        """
        t_target = t - self.tau
        buf = self._buffer

        # Casos extremos: objetivo antes o después del rango del buffer
        if t_target <= buf[0][0]:
            return buf[0][1]
        if t_target >= buf[-1][0]:
            return buf[-1][1]

        # Usar bisect para encontrar índice de inserción
        # buf es lista de (timestamp, valor) ordenada por timestamp
        timestamps = [item[0] for item in buf]
        idx = bisect.bisect_left(timestamps, t_target)

        # Si idx es 0, está antes del primer elemento (ya manejado arriba)
        # Si idx es len(buf), está después del último (ya manejado arriba)
        if idx == 0:
            return buf[0][1]
        if idx == len(buf):
            return buf[-1][1]

        # Interpolación lineal entre idx-1 y idx
        t0, o0 = buf[idx - 1]
        t1, o1 = buf[idx]

        if t1 == t0:
            return o0
        alpha = (t_target - t0) / (t1 - t0)
        return o0 + alpha * (o1 - o0)


# ---------------------------------------------------------------------------
# JohanssonBalkeniusBox
# ---------------------------------------------------------------------------

class JohanssonBalkeniusBox:
    """
    Núcleo individual en el modelo de control pupilar de Johansson & Balkenius.

    Implementa las Ecuaciones (1), (2), y opcionalmente (3) del artículo.

    Ecuación (1) - dinámica de estado:
        ε · dx/dt = α + β · (1/(1+S)) · Σ(Eᵢ·wᵢ) - γ · ΣIⱼ - x

    Ecuación (2) - salida (sigmoide bipolar saturante):
        o = φ · arctan(x),  con φ = 1/arctan(1) = 4/π ≈ 1.2732

    El factor φ NO acota el rango total a la unidad; calibra la pendiente
    en x=0 a 1 y hace que la curva pase por (1, 1). Por construcción:

        - Rango "operativo"   (|x| ≤ 1):    o ∈ [-1, 1]
            Zona quasi-lineal donde la pendiente unitaria es válida y donde
            la dinámica del modelo opera la mayor parte del tiempo.
        - Rango asintótico    (|x| → ±∞):   o → ±φ·π/2 = ±2 (cota teórica
            no alcanzable, pero a la que el modelo se aproxima por saturación
            de arctan en estados saturados).

    Las conexiones internas propagan o sin reescala, de modo que los valores
    recibidos por una caja desde otra caja viven en (-2, 2). Para mapear
    a/desde rangos físicos externos (luminancia, activación muscular, diámetro
    pupilar, ...) usar las utilidades de la clase JBNormalization.

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
        se calcula como el MÁXIMO de las entradas 'us' presentes, o 0.0 si la
        caja no tiene conexiones 'us' entrantes (en cuyo caso no aprende).
    """

    # ------------------------------------------------------------------
    # Constantes derivadas de la Ecuación (2): o = φ · arctan(x)
    # ------------------------------------------------------------------
    _PHI                = 1.0 / np.arctan(1.0)   # = 4/π ≈ 1.2732 — pendiente 1 en x=0
    # _O_OPERATIVE_BOUND  = 1.0                    # |o| cuando |x| = 1  (zona quasi-lineal)
    # _O_ASYMPTOTIC_BOUND = _PHI * (np.pi / 2.0)   # ≈ 2.0 — |o| cuando |x| → ∞

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
        # Las salidas o = φ·arctan(x) son bipolares y pueden ser negativas
        # cuando la inhibición domina. Un shunting biológicamente válido
        # requiere S ≥ 0 (no existe "anti-shunting"); por eso se rectifica
        # con max(0, S) en lugar de aceptar valores negativos o ramificar.
        # shunt_factor = 1.0 / (1.0 + max(0.0, S))

        # Las salidas o = φ·arctan(x) son bipolares (negativas cuando domina
        # la inhibición). El shunting requiere magnitud no negativa: se usa
        # |S| en lugar de max(0, S) para que la inhibición dominante atenúe
        # de forma simétrica a la excitación dominante, sin descartar la
        # información del semieje negativo. En S = 0 el factor vale 1 y decae
        # monotónicamente con |S|, preservando shunt_factor ∈ (0, 1].
        shunt_factor = 1.0 / (1.0 + abs(S))

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
        initial_w = 0.0 if self.plastic else 1.0
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
               usando US efectivo = MÁXIMO de entradas 'us'. Si la caja no
               tiene conexiones 'us' entrantes, US efectivo = 0 → no aprende.
            4. Confirmar nuevo x y calcular nuevo o.
        """
        E_vals = entradas.get('E', [])
        I_vals = entradas.get('I', [])
        S_val  = entradas.get('S', 0.0)
        U_vals = entradas.get('U', [])

        # US efectivo: MÁXIMO de entradas 'us' presentes; 0.0 si no hay.
        # Se usa el máximo (no el promedio) para que basta una vía activa
        # reclute plasticidad sin que vías silentes la atenúen — relevante
        # en cerebelo lateralizado donde sólo el lado iluminado descarga.
        US_eff = max(U_vals) if U_vals else 0.0

        # Construir pares excitatorios ponderados
        E_weighted = [
            (E_vals[i], self._weights.get(i, 1.0))
            for i in range(len(E_vals))
        ]

        # Paso RK4
        x_new = self._rk4(self.x, E_weighted, I_vals, S_val, dt)

        # Actualización de pesos para cajas plásticas — Ecuación (3)
        # ε · dw_i/dt = λ · E_i · (US − o)  →  dw_i/dt = (λ/ε) · E_i · (US − o)
        # La división por ε es necesaria para reproducir fielmente la tasa de
        # aprendizaje del paper J&B 2018; omitirla amplifica λ por 1/ε
        # (factor 5× con el ε=0.2 por defecto).
        if self.plastic and US_eff > self.o:
            for i, e_val in enumerate(E_vals):
                dw = self.lambda_rate * e_val * (US_eff - self.o) / self.epsilon
                self._weights[i] = self._weights.get(i, 1.0) + dt * dw

        # Confirmar
        self.x = x_new
        self.o = self._output(x_new)

    # ------------------------------------------------------------------
    # Push al buffer de salida  — llamado por el sistema en Fase 3
    # ------------------------------------------------------------------

    def push_output(self, t: float, connections: List['Conexion'], dt: float) -> None:
        """
        Envía la salida actual `o` a todas las conexiones salientes.
        Llamado por el sistema después de que todas las cajas han confirmado
        su nuevo estado.

        El sistema garantiza (vía `_outgoing[name]`) que `connections` sólo
        contiene conexiones cuya fuente es esta caja; no se requiere filtro
        adicional. Se propaga `dt` para que `Conexion.push` pueda integrar
        las perturbaciones de comunicación (depleción, ruido, jitter).
        """
        for c in connections:
            c.push(t, self.o, dt)

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
    'default_epsilon': DEFAULT_EPSILON,
    'default_tau':DEFAULT_TAU,
    'default_beta':'auto',
    'default_gamma':DEFAULT_GAMMA,
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
        {'name': 'PVN',    'alpha': 1.0,  'beta': 'auto', 'gamma': 1.0, 'epsilon': DEFAULT_EPSILON},
        # {'name': 'PVN',    'alpha': 1.0,  'beta': 'auto', 'gamma': 1.0, 'epsilon': 5*DEFAULT_DT},

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
        self.enable_history = True # flag para habilitar el registro de los estados

        # Defaults globales de las perturbaciones de comunicación.
        # default_U = default_sigma = default_tau_jitter = 0.0 preservan
        # el comportamiento JB ideal por defecto: los tests existentes
        # corren sin cambios. Para activar cualquiera de las perturbaciones,
        # subir el default a nivel de sistema o sobreescribir per-conexión
        # vía los campos opcionales 'tau_rec', 'U', 'u_initial', 'sigma',
        # 'tau_jitter' en cada dict de la lista 'connections' del config.
        self.default_tau_rec:    float = system_config.get('default_tau_rec',    1.0)
        self.default_U:          float = system_config.get('default_U',          0.0)
        self.default_u_initial:  float = system_config.get('default_u_initial',  1.0)
        self.default_sigma:      float = system_config.get('default_sigma',      0.0)
        self.default_tau:        float = system_config.get('default_tau',        0.02)
        self.default_tau_jitter: float = system_config.get('default_tau_jitter', 0.0)
        self.default_beta              = system_config.get('default_beta',    "auto")
        self.default_gamma             = system_config.get('default_gamma',   "auto")

        self.default_epsilon_nom: float = system_config.get('default_epsilon', 1.0)
        self.default_epsilon_std: float = system_config.get('default_jitter_epsilon', 0.0)


        # Construir cajas
        # Nota: el parámetro 'US' del config ya no se acepta — el US viene
        # ahora de conexiones 'us' (ver tipo de conexión en Conexion).
        self.cajas: Dict[str, JohanssonBalkeniusBox] = {}
        for bcfg in system_config.get('boxes', []):
            epsilon_nom = bcfg.get('epsilon', self.default_epsilon_nom)
            epsilon_std = bcfg.get('jitter_epsilon', self.default_epsilon_std)

            if epsilon_std > 0.0:
                # Log-normal: media en escala natural = epsilon_nom
                # sigma_log ≈ CV = epsilon_std / epsilon_nom (válido para CV pequeño)
                mu_log    = np.log(epsilon_nom)
                sigma_log = epsilon_std / epsilon_nom  # coeficiente de variación
                epsilon   = np.random.lognormal(mu_log, sigma_log)
            else:
                epsilon = epsilon_nom

            box = JohanssonBalkeniusBox(
                name         = bcfg['name'],
                alpha        = bcfg.get('alpha', 0.0),
                beta         = bcfg.get('beta', self.default_beta),
                gamma        = bcfg.get('gamma', self.default_gamma),
                epsilon      = epsilon,
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
                tau            = ccfg.get('tau', self.default_tau),
                resting_output = fuente.o,
                # Perturbaciones de comunicación: usar override per-conexión
                # si existe, de lo contrario caer en los defaults globales
                # del sistema.
                tau_rec        = ccfg.get('tau_rec',    self.default_tau_rec),
                U              = ccfg.get('U',          self.default_U),
                u_initial      = ccfg.get('u_initial',  self.default_u_initial),
                sigma          = ccfg.get('sigma',      self.default_sigma),
                tau_jitter     = ccfg.get('tau_jitter', self.default_tau_jitter),
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
    # Gestión de pesos
    # ------------------------------------------------------------------

    def export_weights(self) -> Dict[str, Dict[int, float]]:
        """
        Exporta los pesos aprendidos de todas las cajas plásticas.
        
        Returns:
            Dict con formato: {nombre_caja: {indice_peso: valor_peso, ...}, ...}
            Solo incluye cajas plásticas (CB, AMY) que tienen pesos variables.
        """
        weights_dict = {}
        for name, box in self.cajas.items():
            if box.plastic and hasattr(box, '_weights'):
                weights_dict[name] = box._weights.copy()
        return weights_dict

    def load_weights(self, weights: Dict[str, Dict[int, float]]) -> None:
        """
        Carga pesos aprendidos en las cajas plásticas del sistema.
        
        Args:
            weights: Dict con formato {nombre_caja: {indice_peso: valor_peso, ...}}
                    Debe contener solo cajas plásticas existentes en el sistema.
        """
        for box_name, box_weights in weights.items():
            if box_name in self.cajas:
                box = self.cajas[box_name]
                if box.plastic and hasattr(box, '_weights'):
                    # Solo cargar pesos para cajas plásticas
                    for idx, weight_value in box_weights.items():
                        box._weights[idx] = weight_value
                else:
                    print(f"Advertencia: {box_name} no es una caja plástica, ignorando pesos")
            else:
                print(f"Advertencia: caja {box_name} no existe en el sistema")

    def get_plastic_boxes_info(self) -> Dict[str, Dict]:
        """
        Obtiene información detallada sobre las cajas plásticas del sistema.
        
        Returns:
            Dict con info de cada caja plástica: pesos, parámetros, etc.
        """
        info = {}
        for name, box in self.cajas.items():
            if box.plastic:
                info[name] = {
                    'plastic': box.plastic,
                    'lambda_rate': box.lambda_rate,
                    'epsilon': box.epsilon,
                    'weights': box._weights.copy() if hasattr(box, '_weights') else {},
                    'current_output': box.o,
                    'current_state': box.x
                }
        return info

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
                    U.append(val)   # entradas US: máximo (ver compute_and_commit)
            entradas_snapshot[name] = {'E': E, 'I': I, 'S': S, 'U': U}

        # FASE 2: compute_and_commit para TODAS las cajas
        # No se toca ningún buffer aquí — sincronía estricta
        for name, box in self.cajas.items():
            box.compute_and_commit(entradas_snapshot[name], self.dt)

        # FASE 3: enviar salidas a buffers de conexión salientes.
        # push() encapsula TODAS las perturbaciones de comunicación
        # (depleción Tsodyks-Markram, ruido σ, jitter τ_jitter) en un
        # único paso de salida. Cuando U=σ=τ_jitter=0, push se reduce
        # a un append directo (comportamiento JB ideal).
        for name, box in self.cajas.items():
            box.push_output(self.t, self._outgoing[name], self.dt)

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
        if self.enable_history:
            for name, box in self.cajas.items():
                self.historia[name].append((self.t, box.x, box.o))

    # ------------------------------------------------------------------
    # Auxiliares de acceso en tiempo de ejecución
    # ------------------------------------------------------------------

    def get_box(self, name: str) -> JohanssonBalkeniusBox:
        """Retorna una caja por nombre para modificación de parámetros en tiempo de ejecución."""
        return self.cajas[name]

    def get_connection(self, src_name: str, dst_name: str) -> Optional[Conexion]:
        """
        Retorna la Conexion que va de `src_name` a `dst_name`, o None si
        no existe. Útil para inspección y modificación selectiva en tests
        (ej. cambiar tau_rec/U solo en una rama específica).

        Si hay múltiples conexiones con la misma pareja (src, dst) — caso
        raro pero posible si el config las repite — retorna la primera.
        """
        for c in self._incoming.get(dst_name, []):
            if c.fuente.name == src_name:
                return c
        return None

    # Definición de inputs del sistema
    # ------------------------------------------------------------------

    def define_input(self, name: str, box_names: List[str], function: Optional[Callable] = None) -> None:
        """
        Define un input del sistema que mapea a cajas específicas.

        El input modifica el parámetro alpha de las cajas especificadas.
        Opcionalmente puede aplicar una función de conversión al valor de entrada.

        Args:
            name: Nombre del input (ej. 'retina_left').
            box_names: Lista de nombres de cajas cuyo alpha se modificará.
            function: Función opcional que procesa el valor de entrada antes de asignarlo.
                      Si es None, se usa el valor directamente (comportamiento original).
        """
        for box_name in box_names:
            if box_name not in self.cajas:
                raise ValueError(f"Caja '{box_name}' no existe en el sistema")
        self._input_mapping[name] = (box_names, function)

    def set_input(self, name: str, value: float) -> None:
        """
        Establece el valor de un input, actualizando alpha de las cajas asociadas.

        Args:
            name: Nombre del input definido previamente.
            value: Valor a asignar al parámetro alpha de las cajas.
        """
        if name not in self._input_mapping:
            raise ValueError(f"Input '{name}' no definido. Usar define_input primero.")
        
        box_names, function = self._input_mapping[name]
        
        # Aplicar función de conversión si existe
        processed_value = function(value) if function is not None else value
        
        for box_name in box_names:
            self.cajas[box_name].alpha = processed_value

    # ------------------------------------------------------------------
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

    def set_all_synaptic_params(
        self,
        tau_rec: Optional[float] = None,
        U: Optional[float] = None,
        u_initial: Optional[float] = None,
        sigma: Optional[float] = None,
        tau_jitter: Optional[float] = None,
    ) -> None:
        """
        Establece parámetros de las perturbaciones de comunicación para
        todas las conexiones del sistema. Solo se modifican los parámetros
        que NO sean None.

        Nota: si se modifica `u_initial` mid-simulación, las muestras ya
        almacenadas en el buffer fueron producidas con la `u` anterior;
        la inconsistencia se diluye tras τ + buffer_margin segundos.

        Args:
            tau_rec: Constante de tiempo de recuperación del pool (s).
            U: Fracción de uso por unidad de actividad presináptica.
            u_initial: Valor inicial del recurso ∈ [0, 1] (también resetea
                       el estado actual `u` de cada conexión).
            sigma: Amplitud del ruido estocástico multiplicativo (≥0).
            tau_jitter: Amplitud relativa del jitter del timestamp (≥0,
                        valores típicos 0.05–0.2).
        """
        for c in self.conexiones:
            if tau_rec is not None:
                c.tau_rec = tau_rec
            if U is not None:
                c.U = U
            if u_initial is not None:
                c.u = float(np.clip(u_initial, 0.0, 1.0))
            if sigma is not None:
                c.sigma = sigma
            if tau_jitter is not None:
                c.tau_jitter = tau_jitter

    def __repr__(self) -> str:
        return (
            f"JohanssonBalkeniusSystem(dt={self.dt}, "
            f"n_cajas={len(self.cajas)}, "
            f"n_conexiones={len(self.conexiones)})"
        )

class ConfigurableJBS(JohanssonBalkeniusSystem):
    """
    Sistema Johansson-Balkenius configurable con parámetros pupilar como atributos de instancia.
    
    Hereda de JohanssonBalkeniusSystem y permite personalizar los parámetros
    de diámetro pupilar (D_MIN, D_MAX, D_REF) para cada instancia del sistema,
    evitando problemas con variables estáticas.
    """
    
    # def __init__(self, system_config: Union[Dict, str] = "default",
    #              D_min: float = 2.0, D_max: float = 8.0, D_ref: float = 4.0,
    #              L_gain: float = 1.0, L_gamma: float = 1.0):
    def __init__(self, system_config: Union[Dict, str] = "default",
                 D_min: float = 4.99, D_max: float = 8.63, D_ref: float = 5.87,
                 L_gain: float = 0.6458, L_gamma: float = 0.5127):
        """
        Inicializa el sistema con parámetros pupilar personalizables.
        
        Args:
            system_config: Configuración del sistema (dict o "default")
            D_min: Diámetro pupilar mínimo (mm)
            D_max: Diámetro pupilar máximo (mm)
            D_ref: Diámetro pupilar de referencia (mm)
            L_gain: Ganancia de la entrada lumínica (escala la magnitud de L_retina)
            L_gamma: Exponente de la ley de potencia (compresión no lineal de la entrada)
                     L_gamma < 1 → compresión logarítmica (respuesta visual real)
                     L_gamma = 1 → lineal (comportamiento original)
                     L_gamma > 1 → expansión
        """
        # Guardar parámetros como atributos de instancia
        self.D_min = D_min
        self.D_max = D_max
        self.D_ref = D_ref
        self.L_gain = L_gain
        self.L_gamma = L_gamma
        
        # Inicializar sistema padre
        super().__init__(system_config)

        # configurar entradas y salidas
        self.setup_standard_inputs()
        self.setup_standard_outputs()
        self.set_cortical_baseline()
        
        # Estado interno para seguimiento de variables
        self._state = {
            'luminance_left': {
                'raw': None,
                'normalized': None,
                'candela': None,
                'perceived': None
            },
            'luminance_right': {
                'raw': None,
                'normalized': None,
                'candela': None,
                'perceived': None
            },
            'pupil_left': None,
            'pupil_right': None,
            'cortical_excitatory': None,
            'cortical_emotional': None,
            'cortical_novelty': None,
            'box': {}  # Para almacenar outputs de todas las boxes
        }

        # Variable de configuracion
        self._config = {
            'enable_bilateral_noise' : False,
            'enable_stochastic_noise' : False,
            'enable_emotional' : False,
            'emotional_threshold' : 0.5,
            # Constantes para conversión de porcentajes a candelas/m²
            'L_MIN' : 1e-4,  # 0.0001 cd/m² (oscuridad completa)
            'L_MAX' : 1e2 ,  # 100 cd/m² (luz máxima) [original 1e4]            
            # Parámetros de ruido para simulaciones realistas
            'L_BILATERAL_NOISE_STD' : 0.05,  # SD de ruido aditivo para bilateralidad (<5%)
            'L_STOCHASTIC_NOISE_STD' : 0.03  # SD de ruido aditivo para estocasticidad (<5%)
        }
    
    def config(self, key, value) -> None:
        """ modifica la configuracion del par clave:valor"""
        if key not in self._config:
            raise ValueError(f"Clave '{key}' no reconocida. Claves válidas: {list(self._config.keys())}")
        self._config[key] = value

    @staticmethod
    def stanley_davies_diameter(L_cand:float, area_deg:float) -> float:
        """ Calcula el diametro pupilar utilizando el metodo de Stanley and Davis (1995) 
        area degree ranges from 0.4 to 25.4° in Watson & Yellot 2012"""

        LA_factor = (L_cand*area_deg/846)**0.41
        diameter = 7.75 - 5.75*(LA_factor/(LA_factor + 2))
        return float(diameter)
    
    def pupil_diameter(self, cg_o: float, scg_o: float) -> float:
        """
        Calcula el diámetro pupilar usando los parámetros de esta instancia.
        
        Args:
            cg_o: Salida del Ganglio Ciliar (esfínter)
            scg_o: Salida del Ganglio Cervical Superior (dilatador)
            
        Returns:
            float: Diámetro pupilar (mm)
        """
        d_basal = self.D_ref
        rango_constriccion = d_basal - self.D_min
        rango_dilatacion = self.D_max - d_basal
        
        # Mapeo bipolar→unipolar: salida de caja en [-2, 2] → activación muscular en [0, 1]
        # NOTA: Normalización y clipping de salidas musculares - DISEÑO ORIGINAL DEL PAPER
        # -----------------------------------------------------------------------
        # El diseño original del paper no incluye estos clips/normalizaciones.
        # Se mantienen por compatibilidad con código existente, pero podrían ser
        # candidatos para modificar si el sistema no logra constricción suficiente.
        #
        # Posibles modificaciones futuras:
        # 1. cg_o_clipped = normalize(cg_o, [-2, 2], [0, 1])  # Limita activación máxima
        #    -> Cambiar a: cg_o_clipped = normalize(cg_o, [-10, 10], [0, 1])  # Mayor rango dinámico
        #
        # 2. scg_o_clipped = normalize(scg_o, [-2, 2], [0, 1])  # Limita activación máxima  
        #    -> Cambiar a: scg_o_clipped = normalize(scg_o, [-10, 10], [0, 1])  # Mayor rango dinámico
        #
        # 3. return float(np.clip(d, self.D_min, self.D_max))  # Clip final del diámetro
        #    -> Cambiar a: return float(d)  # Sin clip para permitir mayor constricción

        def normalize(value: float, in_range: list = [0.0,1.0], out_range: list = [-2.0,2.0]) -> float:
            """Función de normalización local."""
            in_min = np.min(in_range)
            in_max = np.max(in_range)
            value = np.clip(value, in_min, in_max)
            
            out_min = np.min(out_range)
            out_max = np.max(out_range)

            mid_value = (value - in_min)/(in_max - in_min)
            normalized_value = mid_value * (out_max - out_min) + out_min

            return float(normalized_value)
        
        # Normalización actual (limita activación muscular a [0,1])
        # cg_o_clipped = normalize(cg_o, [-2, 2], [0, 1])
        # scg_o_clipped = normalize(scg_o, [-2, 2], [0, 1])
        cg_o_clipped = cg_o
        scg_o_clipped = scg_o
        
        
        d = d_basal - (rango_constriccion * cg_o_clipped) + (rango_dilatacion * scg_o_clipped)
        # return float(np.clip(d, self.D_min, self.D_max))
        return float(d)
    
    def _normalized_luminance_to_candels(self, L_normalized: float) -> float:
        """
        Convierte luminancia normalizada [0-1] a candelas/m².
        """
        l_min = self._config['L_MIN']
        l_max = self._config['L_MAX']
        return l_min+ (l_max - l_min) * L_normalized
    
    def light_input(self, L_normalized: float, d: float) -> float:
        """
        Calcula entrada óptica usando porcentaje de luminancia [0-1].
        
        Método alternativo a candel_input que trabaja directamente con rangos
        normalizados de luminancia, simplificando el uso en tests.
        
        La señal de entrada se calcula en dos etapas:
          1. Conversión a candelas/m²: L_cand = percentage_to_candelas(L_percentage)
          2. Conversión a trolands: L_troland = L_cand × π × (d/2)²
          3. Compresión no lineal: L_retina = L_gain × (L_troland / 278) ^ L_gamma

        Este método es ideal para:
        - Tests que trabajan con porcentajes [0-1]
        - Simulaciones con rangos normalizados
        - Entradas controladas sin necesidad de conversión manual

        Args:
            L_percentage: Porcentaje de luminancia (0.0 = oscuridad, 1.0 = máximo)
            d: Diámetro pupilar actual (mm)

        Returns:
            float: Entrada óptica normalizada (adimensional)
        """
        return self.candel_input(self._normalized_luminance_to_candels(L_normalized), d)

    def step_simulation(self, L: float, emotional_value:float = 0.0, inputs:dict = None) -> None:
        """
        Ejecuta un paso completo de simulación con entrada de luz unificada y procesamiento cortical.
        
        Este método encapsula un paso de simulación utilizado en los tests,
        incluyendo:
        - Entrada óptica unificada [0-1] con ruido bilateral y estocástico opcional
        - Procesamiento cortical (excitatorio y emocional opcional)
        - Paso de simulación
        - Actualización de estado interno
        
        Args:
            L: Luminancia entrada unificada [0-1]
            enable_bilateral_noise: Si agregar ruido aditivo para bilateralidad (<5% SD)
            enable_stochastic_noise: Si agregar ruido aditivo para estocasticidad (<5% SD)
            avg_luminance: Luminancia promedio (calculada si es None)
            enable_emotional: Si activar entrada emocional basada en umbral
            emotional_threshold: Umbral para activar entrada emocional
            emotional_value: Valor para entrada emocotional cuando se activa
            
        Returns:
            None: Solo modifica el estado interno del sistema
        """
        # Obtener salidas actuales
        pupil_left = self.get_output("pupil_left")
        pupil_right = self.get_output("pupil_right")
        
        # Aplicar ruido bilateral y estocástico a la luz base
        L_left = L
        L_right = L

        # State update
        self._state['luminance_left']['raw'] = L_left
        self._state['luminance_right']['raw'] = L_right
        

        # Noises
        L_BILATERAL_NOISE_STD = self._config['L_BILATERAL_NOISE_STD']
        L_STOCHASTIC_NOISE_STD = self._config['L_STOCHASTIC_NOISE_STD']

        if self._config['enable_bilateral_noise']:
            # Ruido aditivo para separación de pupilas (<5% SD)
            bilateral_noise_left = np.random.normal(0, L_BILATERAL_NOISE_STD)
            bilateral_noise_right = np.random.normal(0, L_BILATERAL_NOISE_STD)
            L_left += bilateral_noise_left
            L_right += bilateral_noise_right
            
        if self._config['enable_stochastic_noise']:
            # Ruido aditivo para estocasticidad de la fuente (<5% SD)
            stochastic_noise = np.random.normal(0, L_STOCHASTIC_NOISE_STD)
            L_left += stochastic_noise
            L_right += stochastic_noise

        # Asegurar que L sea estrictamente ≥ 0 (físicamente correcto)
        # El ruido puede hacer L temporalmente negativo, pero la luz base nunca
        L_left = max(0.0, L_left)
        L_right = max(0.0, L_right)
        # State update
        self._state['luminance_left']["normalized"] = L_left
        self._state['luminance_right']["normalized"] = L_right
        
        # Mantener linealidad del ruido sin clips
        # Los valores pueden salir del rango [0,1] temporalmente para preservar
        # la naturaleza estocástica del ruido aditivo
        
        # Aplicar entrada óptica usando el método del sistema
        alpha_left = self.light_input(L_left, pupil_left)
        alpha_right = self.light_input(L_right, pupil_right)
        self.set_input("retina_left", alpha_left)
        self.set_input("retina_right", alpha_right)

        # State update
        self._state['luminance_left']["candela"] = self._normalized_luminance_to_candels(L_left)
        self._state['luminance_right']["candela"] = self._normalized_luminance_to_candels(L_right)
        self._state['luminance_left']["perceived"] = alpha_left
        self._state['luminance_right']["perceived"] = alpha_left

        # Entradas corticales
        avg_luminance = (L_left + L_right) / 2.0
        
        # Normalización simple para entrada cortical
        cortex_excitatory = avg_luminance * 2.0 - 1.0  # Mapeo [0,1] → [-1,1]
        self.set_input("cortex_excitatory", cortex_excitatory)
        
        # Entrada emocional opcional basada en umbral
        if self._config['enable_emotional']:
            if avg_luminance > self._config['emotional_threshold']:
                cortex_emotional = emotional_value
            else:
                cortex_emotional = 0.0
            self.set_input("cortex_emotional", cortex_emotional)
        else:
            cortex_emotional = 0.0
            self.set_input("cortex_emotional", 0.0)
        
        # Aplicar los inputs customs
        if isinstance(inputs,dict) and len(inputs):
            for key,value in inputs.items():
                self.set_input(key, value)

        # Paso de simulación
        self.step()
        
        # Obtener salidas actualizadas y actualizar estado interno
        pupil_left = self.get_output("pupil_left")
        pupil_right = self.get_output("pupil_right")
        
        # Actualizar estado interno
        self._state['pupil_left'] = pupil_left
        self._state['pupil_right'] = pupil_right
        self._state['cortical_excitatory'] = cortex_excitatory
        self._state['cortical_emotional'] = cortex_emotional
        
        # Obtener output de todas las boxes
        try:
            for box in self.cajas.keys():
                self._state['box'][box] = self.cajas[box].o
        except:
            pass
        
        # simulation_loop no retorna nada, solo modifica el estado interno
        # Los valores se pueden obtener con view_state() o get_output() después

    def view_state(self, variable: str = "all") -> dict:
        """
        Método trivial para visualizar el estado actual del sistema.
        
        Accede directamente al estado interno _state que se actualiza automáticamente
        durante cada paso de simulación. Es extremadamente eficiente y directo.
        
        Args:
            variable: Variable específica a visualizar o "all" para todas
                     Opciones disponibles:
                     - "luminance_candelas": Luminancia en candelas/m²
                     - "luminance_normalized": Luminancia normalizada [0-1]
                     - "luminance_perceived": Luminancia percibida (alpha)
                     - "pupil_left", "pupil_right": Diámetro pupilar individual
                     - "cortical_excitatory", "cortical_emotional": Estados corticales
                     - "muscular_cg_left", etc.: Estados musculares individuales
                     - "all": Todo el estado interno
        
        Returns:
            dict: Diccionario con los valores solicitados
        """
        if variable == "all":
            # Retornar todo el estado interno + info del sistema
            result = self._state.copy()
            result["system_info"] = {
                "time": self.t,
                "parameters": {
                    "D_min": self.D_min,
                    "D_max": self.D_max,
                    "D_ref": self.D_ref,
                    "L_gain": self.L_gain,
                    "L_gamma": self.L_gamma,
                },
                "config": self._config
            }
            return result
        elif variable in self._state:
            # Retornar variable específica
            return {variable: self._state[variable]}
        else:
            # Variable no encontrada
            return {variable: None, "error": f"Variable '{variable}' no encontrada en el estado"}

    def candel_input(self, L_cand: float, d: float) -> float:
        """
        Calcula entrada óptica usando parámetros de esta instancia.

        La señal de entrada al modelo se calcula en dos etapas:
          1. Conversión a trolands estándar: L_troland = L_cand × π × (d/2)²
          2. Compresión no lineal (ley de potencia):
             L_retina = L_gain × (L_troland / 278) ^ L_gamma

        Con L_gamma = 1 y L_gain = 1 recupera el comportamiento original lineal.
        Con L_gamma < 1 se obtiene compresión logarítmica, que aproxima la
        respuesta compresiva del sistema visual ante el rango dinámico de luminancia.

        Args:
            L_cand: Luminancia en candelas/m²
            d: Diámetro pupilar actual (mm)

        Returns:
            float: Entrada óptica normalizada (adimensional)
        """
        # Clipping del diámetro (comentado: puede limitar constricción en alta luminosidad)
        # d = np.clip(d, self.D_min, self.D_max)

        # Paso 1: iluminancia retinal en trolands (siempre positivo)
        # L_cand > 0, d > 0, π > 0 → L_troland siempre > 0
        L_troland = L_cand * np.pi * (d / 2) ** 2

        # Paso 2: ley de potencia con protección contra valores negativos
        # L_troland > 0, 278 > 0 → división siempre positiva
        # self.L_gamma > 0 → exponente bien definido
        ratio = L_troland / 278
        L_retina = self.L_gain * ratio ** self.L_gamma

        # Depuración: imprimir valores que causan complejos
        if np.iscomplex(L_retina):
            print(f"DEBUG: L_retina complejo detectado!")
            print(f"  L_cand = {L_cand}")
            print(f"  d = {d}")
            print(f"  L_troland = {L_troland}")
            print(f"  ratio = {ratio}")
            print(f"  self.L_gain = {self.L_gain}")
            print(f"  self.L_gamma = {self.L_gamma}")
            print(f"  L_retina = {L_retina}")
            print(f"  L_retina.real = {np.real(L_retina)}")
            print(f"  L_retina.imag = {np.imag(L_retina)}")

        # Retornar directamente (no debería ser complejo si las entradas son físicas)
        try:
            resultado = float(L_retina)
            return resultado
        except (TypeError, ValueError) as e:
            print(f"ERROR: No se puede convertir L_retina a float: {e}")
            print(f"  Usando parte real: {np.real(L_retina)}")
            return float(np.real(L_retina))

    def setup_standard_inputs(self) -> None:
        """
        Configura los inputs estándar del sistema para un test básico.

        Define inputs para:
        - Retinas (izquierda y derecha)
        - Corteza (excitatory, emotional, novelty)
        """
        self.define_input("retina_left", ['left_retinae_l', 'left_retinae_r'])
        self.define_input("retina_right", ['right_retinae_l', 'right_retinae_r'])
        self.define_input("cortex_excitatory", ['cortex_excitatory'])
        self.define_input("cortex_emotional", ['cortex_emotional'])
        self.define_input("cortex_novelty", ['cortex_novelty'])

    def set_cortical_baseline(self, excitatory: float = 0.0,
                              emotional: float = 0.0,
                              novelty: float = 0.0) -> None:
        """
        Establece los inputs corticales en valores de linea base.
        Util para tests de reflejo autonomico puro (sin activacion cortical).

        Args:
            excitatory: Valor para cortex_excitatory (default 0.0).
            emotional: Valor para cortex_emotional (default 0.0).
            novelty: Valor para cortex_novelty (default 0.0).
        """
        self.set_input("cortex_excitatory", excitatory)
        self.set_input("cortex_emotional", emotional)
        self.set_input("cortex_novelty", novelty)

    def setup_standard_outputs(self) -> None:
        """
        Configura los outputs estandar del sistema para un test basico.
        Define outputs para diametro pupilar izquierdo y derecho.
        """
        self.define_output("pupil_left",  ['CG_l', 'SCG_l'], self.pupil_diameter)
        self.define_output("pupil_right", ['CG_r', 'SCG_r'], self.pupil_diameter)
