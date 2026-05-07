# Modelo computacional de control pupilar — Johansson & Balkenius (2018)

## Documentación técnica del módulo `JohanssonBalkenius.py`

---

> **Referencia del paper:**
> Johansson, B., & Balkenius, C. (2018). A computational model of pupil dilation. *Connection Science*, 30(1), 5–19. https://doi.org/10.1080/09540091.2016.1271401

---

## Tabla de contenidos

1. [Fundamento biológico](#1-fundamento-biológico)
2. [Modelo matemático](#2-modelo-matemático)
3. [Arquitectura del código](#3-arquitectura-del-código)
4. [Perturbaciones sinápticas](#4-perturbaciones-sinápticas)
5. [Guía de uso](#5-guía-de-uso)

---

# 1. Fundamento biológico

## 1.1 ¿Qué controla el tamaño de la pupila?

El diámetro pupilar no es fijo: cambia constantemente en respuesta a la cantidad de luz que llega al ojo, al estado emocional y al nivel de alerta del individuo. Detrás de ese ajuste continuo existen dos músculos antagonistas en el iris:

- **Músculo esfínter** (constrictor): reduce el diámetro pupilar (miosis). Está controlado por el sistema nervioso **parasimpático**.
- **Músculo dilatador**: aumenta el diámetro pupilar (midriasis). Está controlado por el sistema nervioso **simpático**.

El modelo de Johansson y Balkenius (2018) representa computacionalmente los circuitos neuronales que gobiernan ambas vías, desde la retina hasta los ganglios que inervan el ojo.

---

## 1.2 La vía parasimpática (constrictora)

Cuando la luz estimula la retina, la señal viaja por este circuito:

```
Retina
  └─► PTA  (Pretectal Area / Área Pretectal)
        └─► EWpg  (Edinger-Westphal, subdivisión pregangliolar)
               └─► CG  (Ganglio Ciliar)
                     └─► Músculo esfínter → constricción pupilar
```

| Núcleo | Nombre completo | Función en el circuito |
|--------|-----------------|------------------------|
| PTA | Área Pretectal | Primera estación que recibe la señal lumínica bilateral (de ambos ojos) y activa la vía parasimpática. |
| EWpg | Núcleo de Edinger-Westphal (subdivisión pregangliolar) | Centro integrador parasimpático; coordina la constricción según la intensidad lumínica y la modulación de otras áreas. |
| CG | Ganglio Ciliar | Último relevo antes del músculo; transmite la orden de constricción directamente al esfínter del iris. |

> **Nota sobre bilateralidad.** Aunque cada ojo posee su propia retina y sus propios PTA, EWpg y CG (sufijos `_l` y `_r` en el código), las señales se mezclan entre hemisferios: la iluminación de un ojo produce constricción en ambas pupilas (reflejo consensual). El modelo implementa esta cruce mediante conexiones entre los núcleos lateralizados.

---

## 1.3 La vía simpática (dilatadora)

La dilatación pupilar responde principalmente al nivel de alerta (arousal) y a estímulos emocionales. Su circuito es:

```
LC  (Locus Coeruleus)
  └─► IML  (Columna Intermediolateral de la médula espinal)
        └─► SCG  (Ganglio Cervical Superior)
               └─► Músculo dilatador → dilatación pupilar
```

| Núcleo | Nombre completo | Función en el circuito |
|--------|-----------------|------------------------|
| LC | Locus Coeruleus | Principal fuente de noradrenalina del encéfalo; se activa ante estrés, novedad o esfuerzo cognitivo y desencadena la dilatación. |
| IML | Columna Intermediolateral | Núcleo de la médula espinal que recibe la orden del LC y la transmite hacia la periferia. |
| SCG | Ganglio Cervical Superior | Último relevo simpático; inerva el músculo dilatador del iris. |

---

## 1.4 Las áreas moduladoras

Además de las vías directas, el modelo incluye estructuras que modulan la respuesta pupilar según contexto:

| Área | Rol |
|------|-----|
| **Hipotálamo** (SCN, VLPO, LH, DMH, PVN) | Integra el ciclo circadiano y el estado de vigilia/sueño; regula al LC. El PVN tiene actividad basal sostenida (α = 1.0) que mantiene un tono simpático de fondo. |
| **Amígdala** (AMY) | Procesa la relevancia emocional de estímulos; amplifica la respuesta del LC ante situaciones de alta carga afectiva. Es una estructura **plástica**: aprende asociaciones entre un estímulo condicionado y una respuesta emocional. |
| **Cerebelo** (CB) | Genera predicciones anticipatorias de la actividad de EWpg; permite que la pupila se adapte antes de que cambie la luz (respuesta predictiva). También es **plástico**. |
| **Corteza** | Provee entradas de contexto externo: excitación cognitiva (`cortex_excitatory`), estado emocional (`cortex_emotional`) y detección de novedad (`cortex_novelty`). |

---

## 1.5 Diagrama general del sistema

```
           [CORTEZA]
          /    |    \
   excit. emoc. novel.
         |     |     |
        CB    LC   AMY
         |   / \    |
  EWpg_pred  |  IML-SCG (simpática)
         |   |
 Retinas→PTA→EWpg→CG  (parasimpática)
              |
         [HIPOTÁLAMO]
        SCN-VLPO-DMH-LH-PVN
```

El diámetro pupilar final resulta del balance entre la activación del CG (constricción) y del SCG (dilatación).

---

# 2. Modelo matemático

El modelo representa cada núcleo cerebral como una **"caja"** con un estado interno que evoluciona en el tiempo. Hay tres ecuaciones fundamentales.

---

## 2.1 Ecuación 1 — Dinámica del estado interno

Describe cómo cambia el estado interno *x* de un núcleo a lo largo del tiempo:

$$\varepsilon \cdot \frac{dx}{dt} = \alpha + \frac{\beta}{1 + |S|} \cdot \sum_i E_i \cdot w_i \;\;-\;\; \gamma \cdot \sum_j I_j \;\;-\;\; x$$

En palabras simples: el estado de un núcleo sube si recibe señales excitatorias y baja si recibe señales inhibitorias. El valor actual *x* actúa como un freno que lleva al núcleo de vuelta a su punto de reposo.

### Parámetros y su análogo biológico

| Símbolo | Nombre en el código | Significado biológico |
|---------|---------------------|-----------------------|
| **ε** (epsilon) | `epsilon` | **Inercia del núcleo.** Un núcleo con ε grande tarda más en cambiar (responde lentamente a sus entradas). Un ε pequeño responde casi al instante. Biológicamente refleja la constante de tiempo de la membrana neuronal. |
| **α** (alpha) | `alpha` | **Actividad basal.** Nivel de activación cuando no llega ninguna señal del exterior. El PVN tiene α = 1.0 para modelar el tono simpático basal permanente del sistema nervioso autónomo. En los nodos de entrada (retinas, corteza) α es la señal de entrada que el experimentador controla. |
| **β** (beta) | `beta` | **Ganancia excitatoria.** Cuánto peso relativo tienen las entradas excitatorias. Por defecto se calcula como 1/N (donde N es el número de conexiones excitatorias entrantes), normalizando la suma para que el estado no dependa de cuántas fuentes llegan. |
| **γ** (gamma) | `gamma` | **Ganancia inhibitoria.** Equivalente a β pero para las entradas inhibitorias. Por defecto = 1/M (M = conexiones inhibitorias). |
| **E_i · w_i** | entradas excitatorias × pesos | Cada señal excitatoria entrante multiplicada por su peso sináptico. En núcleos no plásticos los pesos son fijos (w = 1.0). En núcleos plásticos (AMY, CB) los pesos empiezan en 0 y crecen con el aprendizaje. |
| **I_j** | entradas inhibitorias | Señales que suprimen la actividad del núcleo. |
| **S** | inhibición shunting | Tipo especial de inhibición que reduce la *eficacia* de las entradas excitatorias sin eliminarlas. Equivale biológicamente a sinapsis inhibitorias ubicadas muy cerca del soma (inhibición perisomal o "en derivación"). El LC ejerce este tipo de influencia sobre EWpg. |

> **Sobre el término `1/(1 + |S|)`:** cuando S = 0 (sin shunting), las entradas excitatorias pasan completas. Cuando S crece, el factor se acerca a 0 y las excitatorias quedan silenciadas aunque sean grandes. Se usa el valor absoluto de S porque la salida de las cajas puede ser negativa (ver Ecuación 2), y el shunting solo tiene sentido como atenuador no negativo.

---

## 2.2 Ecuación 2 — Función de salida

Convierte el estado interno *x* (que puede crecer sin límite) en una señal de salida acotada *o*:

$$o = \varphi \cdot \arctan(x) \qquad \text{con } \varphi = \frac{4}{\pi} \approx 1.27$$

El factor φ garantiza que la curva tenga **pendiente 1 en el origen** (respuesta lineal para señales pequeñas) y que la salida pase exactamente por (1, 1). La función arctan satura progresivamente:

| Estado *x* | Salida *o* (aprox.) |
|:----------:|:-------------------:|
| 0 | 0 |
| 1 | 1 (zona quasi-lineal) |
| 5 | ≈ 1.22 |
| → ∞ | → 2 (límite teórico) |

Biológicamente, esta curva reproduce la **saturación de la tasa de disparo neuronal**: a señales débiles la respuesta es proporcional; a señales muy intensas el núcleo ya no puede disparar más rápido.

> Las salidas de las cajas viven en el rango (−2, 2). Valores positivos representan activación; valores negativos, inhibición dominante.

---

## 2.3 Ecuación 3 — Regla de aprendizaje (solo AMY y CB)

Los núcleos plásticos (Amígdala y Cerebelo) modifican sus pesos sinápticos mediante una **regla delta**:

$$\varepsilon \cdot \frac{dw_i}{dt} = \lambda \cdot E_i \cdot (US - o)$$

Donde:

| Símbolo | Nombre en el código | Significado biológico |
|---------|---------------------|-----------------------|
| **λ** (lambda) | `lambda_rate` | **Tasa de aprendizaje.** Qué tan rápido se modifican los pesos. AMY usa λ = 0.05; CB usa λ = 0.1. |
| **US** | señal de entrada tipo `'us'` | **Estímulo incondicionado** (*Unconditioned Stimulus*). Es la señal "maestra" que enseña al núcleo cuánto debería activarse. Para AMY, el US es la novedad cortical; para CB, es la salida de EWpg (la activación real del núcleo parasimpático). |
| **E_i** | entrada excitatoria | Señal que llega por la sinapsis que se está modulando. Solo se refuerzan las conexiones activas en el momento del aprendizaje. |
| **(US − o)** | señal de error | La diferencia entre "lo que debería ocurrir" (US) y "lo que está ocurriendo" (o). Si el núcleo ya predice correctamente, la diferencia es cero y no hay cambio de peso. |

**El aprendizaje ocurre solo cuando US > o**, es decir, cuando la activación real del núcleo queda por debajo de lo que el estímulo incondicionado exige.

> **US como máximo:** si múltiples fuentes envían señal tipo `us`, el valor efectivo es el **máximo** de ellas (no el promedio). Esto garantiza que basta con que una vía esté activa para desencadenar aprendizaje, sin que las vías silentes lo atenúen.

---

## 2.4 El retraso de transmisión (τ)

Las conexiones entre núcleos no son instantáneas. Cada conexión tiene un parámetro τ (`tau`) que representa el **tiempo que tarda la señal en viajar** desde el núcleo fuente hasta el núcleo destino (retardo axonal).

Por defecto τ = 20 ms en todo el sistema, lo que es consistente con los tiempos de conducción del sistema nervioso autónomo. El buffer de cada conexión guarda el historial reciente de la señal e interpola linealmente para obtener el valor exacto en el tiempo (t − τ).

---

## 2.5 La entrada lumínica: de candelas a señal retinal

Antes de que la señal lumínica entre al modelo, se aplica una conversión biofísica en dos pasos:

**Paso 1 — Trolands:** la iluminancia retinal (en trolands) se calcula como:

$$T = L_{cd/m^2} \times \pi \times \left(\frac{d}{2}\right)^2$$

donde *L* es la luminancia de la fuente en cd/m² y *d* es el diámetro pupilar actual en mm. Esto modela el **efecto de la propia pupila sobre la cantidad de luz que entra al ojo** (retroalimentación óptica de lazo cerrado).

**Paso 2 — Compresión no lineal (ley de potencia):**

$$L_{retina} = L_{gain} \times \left(\frac{T}{278}\right)^{L_{\gamma}}$$

| Parámetro | Efecto |
|-----------|--------|
| `L_gain` | Escala global de la señal retinal. |
| `L_gamma` | Controla la compresión. Con L_γ < 1 la curva es logarítmica (como la respuesta real del sistema visual); con L_γ = 1 la respuesta es lineal. |

El valor 278 es el valor de referencia de trolands que normaliza la escala. Los valores ajustados por defecto en `ConfigurableJBS` (L_gain = 0.646, L_gamma = 0.513) reproducen la compresión fotópica característica del ojo humano.

---

# 3. Arquitectura del código

El módulo `JohanssonBalkenius.py` organiza el modelo en cuatro clases apiladas en una jerarquía, donde cada nivel agrega funcionalidad sobre el anterior:

```
Conexion
  └── JohanssonBalkeniusBox
        └── JohanssonBalkeniusSystem
              └── ConfigurableJBS          ← punto de entrada recomendado
```

---

## 3.1 `Conexion` — El cable sináptico

Representa **una sola sinapsis** entre dos núcleos. Su responsabilidad es:

- Guardar un historial de la señal emitida por el núcleo fuente (lista FIFO con ventana deslizante: las muestras nuevas entran por el extremo derecho y las que superan el horizonte τ + margen se eliminan por el izquierdo).
- Entregar esa señal con el retraso τ cuando el núcleo destino la solicita (interpolación lineal).
- Aplicar opcionalmente perturbaciones biológicas a la señal antes de almacenarla (ver Sección 4).

**Tipos de conexión** (`tipo`):

| Tipo | Rol en la Ecuación 1 |
|------|----------------------|
| `'excitatory'` | Contribuye a la suma Σ(E_i · w_i) |
| `'inhibitory'` | Contribuye a la suma ΣI_j |
| `'shunting'` | Contribuye al factor 1/(1 + |S|) |
| `'us'` | Canal de enseñanza para la regla de aprendizaje (Ecuación 3) |

Cada conexión conoce quién es su fuente (`fuente`: un objeto `JohanssonBalkeniusBox`) pero **no** sabe quién es su destino — esa información la maneja el sistema.

---

## 3.2 `JohanssonBalkeniusBox` — Un núcleo cerebral

Representa **un único núcleo** (PTA, EWpg, LC, AMY, CB, etc.). Sus responsabilidades son:

- Mantener su estado interno *x* y calcular su salida *o = φ · arctan(x)*.
- Integrar la Ecuación 1 con el método **RK4** (Runge-Kutta de cuarto orden), que es más preciso que la integración de Euler simple para el mismo paso de tiempo.
- Si es plástico, actualizar sus pesos sinápticos con la Ecuación 3.

El método central es `compute_and_commit(entradas, dt)`. Recibe un diccionario con las señales ya recuperadas de los buffers y avanza el estado un paso `dt`:

```python
entradas = {
    'E': [val0, val1, ...],   # señales excitatorias
    'I': [val0, val1, ...],   # señales inhibitorias
    'S': float,               # señal shunting acumulada
    'U': [val0, val1, ...]    # señales de enseñanza (US)
}
```

> **Diseño deliberado:** `compute_and_commit` solo lee `entradas` y escribe `self.x` / `self.o`. Nunca toca los buffers de otras conexiones. Esto permite que el sistema actualice todos los núcleos con la misma "fotografía" del instante anterior (actualización síncrona), evitando que núcleos actualizados antes afecten a los que aún no se han actualizado en el mismo paso.

---

## 3.3 `JohanssonBalkeniusSystem` — El sistema interconectado

Es el **orquestador**: mantiene todos los núcleos y todas las conexiones, y ejecuta cada paso de simulación en cuatro fases estrictamente ordenadas:

```
Fase 1 — Leer entradas:   para cada núcleo, recuperar señales de los buffers (t − τ)
Fase 2 — Calcular:        cada núcleo integra su Ecuación 1 y actualiza pesos
Fase 3 — Publicar:        cada núcleo envía su nueva salida a sus conexiones salientes
Fase 4 — Registrar:       guardar historial y avanzar el reloj interno (t += dt)
```

Las fases 1 y 2 están separadas porque todos los núcleos deben leer el estado del instante *anterior*, no el del instante que se está calculando. Si la lectura y el cálculo se mezclaran, el resultado dependería del orden en que se procesen los núcleos, lo que no es biológicamente válido.

**Construcción desde configuración:**

El sistema se construye a partir de un diccionario `system_config` con tres secciones:

```python
config = {
    'dt': 0.002,          # paso de integración (s)
    'boxes': [...],       # lista de núcleos
    'connections': [...]  # lista de sinapsis
}
```

También puede instanciarse con el string `"default"` para usar la topología completa de J&B 2018.

**Resolución automática de β y γ:**

Si `beta` o `gamma` se especifican como `'auto'`, el sistema los calcula automáticamente al finalizar la construcción:

- β = 1 / (número de conexiones excitatorias entrantes)
- γ = 1 / (número de conexiones inhibitorias entrantes)

Esto normaliza la suma de entradas independientemente de cuántas fuentes tenga cada núcleo.

**Métodos de acceso en tiempo de ejecución:**

| Método | Uso |
|--------|-----|
| `define_input(name, box_names, fn)` | Asocia un nombre de entrada con uno o varios núcleos. Al llamar `set_input(name, value)`, se actualiza el `alpha` de esos núcleos. |
| `set_input(name, value)` | Cambia la activación basal de los núcleos asociados a esa entrada. |
| `define_output(name, box_names, fn)` | Define cómo calcular una salida del sistema combinando señales de varios núcleos. |
| `get_output(name)` | Evalúa y retorna el valor de esa salida. |
| `get_box(name)` | Retorna el objeto `JohanssonBalkeniusBox` para inspección o modificación directa. |
| `get_connection(src, dst)` | Retorna el objeto `Conexion` entre dos núcleos. |
| `export_weights()` / `load_weights()` | Guarda y carga los pesos aprendidos de las cajas plásticas (serialización del estado de aprendizaje). |
| `generate(t_total)` | Ejecuta la simulación durante `t_total` segundos y retorna el historial completo. |

---

## 3.4 `ConfigurableJBS` — La interfaz de alto nivel

Hereda de `JohanssonBalkeniusSystem` y agrega todo lo necesario para usar el sistema con señales físicas reales (luminancia en cd/m², diámetro en mm):

- Configura automáticamente los inputs estándar (retinas, corteza) y los outputs estándar (diámetro pupilar izquierdo y derecho).
- Expone `step_simulation(L)` como método único de avance: recibe una luminancia normalizada [0–1] y aplica internamente la conversión biofísica, el ruido opcional y la integración.
- Expone `view_state()` para inspeccionar el estado completo del sistema después de cada paso.
- Permite ajustar los parámetros pupilares (D_min, D_max, D_ref) y de conversión lumínica (L_gain, L_gamma) como argumentos del constructor, evitando modificar valores globales.

**Parámetros del constructor de `ConfigurableJBS`:**

| Parámetro | Valor por defecto | Significado |
|-----------|:-----------------:|-------------|
| `D_min` | 4.99 mm | Diámetro pupilar mínimo fisiológico. |
| `D_max` | 8.63 mm | Diámetro pupilar máximo fisiológico. |
| `D_ref` | 5.87 mm | Diámetro de referencia en condiciones basales (mesopio). |
| `L_gain` | 0.6458 | Ganancia de la señal retinal. |
| `L_gamma` | 0.5127 | Exponente de compresión fotópica. |

**Cálculo del diámetro pupilar:**

La salida del sistema es el diámetro pupilar (en mm), calculado como:

```
d = D_ref  −  (D_ref − D_min) × CG_o  +  (D_max − D_ref) × SCG_o
```

donde `CG_o` es la salida del Ganglio Ciliar (activa la constricción) y `SCG_o` es la salida del Ganglio Cervical Superior (activa la dilatación). Cuando ambos están en cero, la pupila queda en `D_ref`.

---

# 4. Perturbaciones sinápticas

El modelo base de J&B 2018 asume sinapsis ideales: la señal llega exactamente τ segundos después de emitirse, sin distorsión. En la realidad biológica, la transmisión sináptica es ruidosa, fatigable y temporalmente variable. El módulo implementa tres mecanismos independientes para modelar estas imperfecciones. **Por defecto los tres están desactivados**, de manera que el comportamiento es idéntico al modelo original del paper. Cada uno puede activarse individualmente.

Las tres perturbaciones actúan en el método `push()` de `Conexion`, que es el único punto donde la señal se escribe en el buffer. La lectura (`get_delayed_output()`) siempre es limpia.

---

## 4.1 Depleción del recurso sináptico (Tsodyks-Markram simplificado)

### ¿Qué modela?

Las neuronas liberan neurotransmisores almacenados en vesículas. Si la neurona dispara repetidamente a alta frecuencia, el pool vesicular se **agota** progresivamente: cada disparo libera menos neurotransmisor que el anterior. Cuando hay un período de silencio, las vesículas se recargan gradualmente.

Este fenómeno se llama **depresión sináptica de corto plazo** y produce que una sinapsis activa continuamente transmita señales cada vez más débiles, aunque el núcleo fuente siga disparando igual.

### La ecuación

El estado del pool vesicular se representa con la variable `u ∈ [0, 1]`, donde u = 1 significa pool lleno y u = 0 significa pool agotado:

$$\frac{du}{dt} = \frac{1 - u}{\tau_{rec}} \;-\; U \cdot u \cdot \max(o_{fuente}, 0)$$

- El primer término (`(1-u)/τ_rec`) es la **recuperación**: el pool se recarga con una constante de tiempo τ_rec.
- El segundo término (`U · u · o`) es la **depleción**: cada vez que la fuente dispara (o > 0), consume una fracción U del pool disponible.

La señal efectiva que llega al destino es `u × o`, no `o` directamente.

### Parámetros

| Parámetro | Nombre en código | Valor por defecto | Efecto biológico |
|-----------|-----------------|:-----------------:|------------------|
| `τ_rec` | `tau_rec` | 1.0 s | Tiempo de recuperación del pool vesicular. Valores pequeños (0.1–0.5 s) modelan sinapsis que se fatigan y recuperan rápido; valores grandes (> 1 s) modelan fatiga lenta. |
| `U` | `U` | 0.0 (desactivado) | Fracción del pool consumida por unidad de actividad. Valores fisiológicos típicos: 0.05–0.30 según el tipo de sinapsis. U = 0 desactiva la depleción. |
| `u_initial` | `u_initial` | 1.0 | Estado inicial del pool. u < 1 modela una sinapsis pre-fatigada o con daño parcial. |

> **Ejemplo A05:** `default_tau_rec = 0.5 s`, `default_U = 0.8`. Esto produce sinapsis que se agotan rápidamente ante actividad sostenida, lo que agrega variabilidad natural a la respuesta del sistema ante escalones de luminancia.

---

## 4.2 Ruido estocástico multiplicativo

### ¿Qué modela?

Incluso con el pool vesicular lleno, la liberación de neurotransmisores tiene un componente **probabilístico**: en cada disparo, la cantidad exacta liberada fluctúa aleatoriamente alrededor de un valor medio. Este ruido de liberación cuántica es una fuente de variabilidad intrínseca de cualquier sinapsis biológica.

En el modelo, el ruido se **escala con el agotamiento** del pool: cuando el pool está lleno (u ≈ 1) hay pocas fluctuaciones; cuando el pool está agotado (u ≈ 0) la liberación es muy errática. Esto reproduce el hecho de que la liberación vesicular se vuelve más variable cuando quedan pocas vesículas disponibles.

### La ecuación

$$o_{ruidosa} = o \cdot \left(1 + \sigma \cdot (1 - u) \cdot \mathcal{N}(0, 1)\right)$$

La señal `o` se multiplica por un factor aleatorio centrado en 1 (sin distorsión media) con amplitud σ · (1 − u).

### Parámetros

| Parámetro | Nombre en código | Valor por defecto | Efecto |
|-----------|-----------------|:-----------------:|--------|
| `σ` | `sigma` | 0.0 (desactivado) | Amplitud del ruido. σ = 0.1 agrega ~10% de variabilidad; σ = 0.5 agrega variabilidad muy alta. |

> **Ejemplo A05:** `default_sigma = 0.6`, lo que produce señales con variabilidad del 60% escalada por el agotamiento — apropiado para modelar ruido sináptico biológicamente plausible en un sistema con alta depleción (U = 0.8).

---

## 4.3 Jitter temporal (variabilidad en la velocidad de conducción)

### ¿Qué modela?

El retraso τ no es exactamente el mismo en cada disparo. La velocidad de conducción de un axón varía ligeramente de un potencial de acción al siguiente, y puede verse afectada por el estado de mielinización, la temperatura y el nivel de actividad previa. Esto produce que la señal no llegue siempre a exactamente (t − τ), sino con una pequeña fluctuación temporal.

Al igual que el ruido de amplitud, este jitter temporal **aumenta con el agotamiento del pool**: cuando el sistema trabaja al límite, la transmisión no solo es más débil sino también más impredecible en tiempo.

### La ecuación

El timestamp con que se almacena la muestra en el buffer recibe un desplazamiento:

$$\delta_t = \tau \cdot \tau_{jitter} \cdot (1 - u) \cdot \mathcal{N}(0, 1)$$
$$t_{efectivo} = t + \max(\delta_t, \;-\tau)$$

El clamp (`max(δ_t, −τ)`) garantiza que el retraso efectivo nunca sea negativo (la señal no puede llegar antes de ser emitida). El jitter solo añade latencia o la reduce, pero no puede crear causalidad inversa.

### Parámetros

| Parámetro | Nombre en código | Valor por defecto | Efecto |
|-----------|-----------------|:-----------------:|--------|
| `τ_jitter` | `tau_jitter` | 0.0 (desactivado) | Amplitud relativa del jitter como fracción de τ. Valores típicos: 0.05–0.20. |

> **Ejemplo A05:** `default_tau_jitter = 0.2`, lo que permite que el timestamp varíe hasta ±20% de τ (escalado por el agotamiento) en cada transmisión.

---

## 4.4 Variabilidad en la constante de tiempo (jitter de ε)

Además de las perturbaciones sinápticas, el sistema permite modelar variabilidad **entre núcleos** en su constante de tiempo ε. Biológicamente, distintas neuronas de un mismo núcleo tienen constantes de membrana algo diferentes.

Si se especifica `default_jitter_epsilon > 0` (o `jitter_epsilon` en un núcleo individual), el valor de ε para cada caja se sortea de una distribución **log-normal** con media `epsilon` y coeficiente de variación `jitter_epsilon / epsilon`. La log-normal garantiza que ε siempre sea positivo.

> **Ejemplo A05:** `default_jitter_epsilon = 0.05`, es decir, una variabilidad del 5% en la velocidad de respuesta de los núcleos.

---

## 4.5 Resumen: activación de perturbaciones

| Perturbación | Parámetro clave | = 0 → desactivado | > 0 → activado |
|-------------|-----------------|:-----------------:|:--------------:|
| Depleción sináptica | `U` | Sinapsis ideal (peso constante) | Fatiga proporcional a la actividad |
| Ruido de canal | `sigma` | Transmisión determinista | Variabilidad de liberación vesicular |
| Jitter temporal | `tau_jitter` | Retardo τ exacto | Variabilidad en velocidad de conducción |
| Variabilidad de ε | `jitter_epsilon` | ε igual para todos | ε sorteado por núcleo (log-normal) |

Todos los parámetros se pueden especificar **globalmente** en `system_config` (aplican a todas las conexiones) o **por conexión individual** sobreescribiendo el campo correspondiente en el dict de esa conexión.

---

# 5. Guía de uso

Esta sección muestra cómo crear, configurar y ejecutar el sistema. El punto de entrada recomendado es siempre `ConfigurableJBS`, que encapsula los detalles de bajo nivel y expone una interfaz orientada al experimento.

---

## 5.1 Instalación y dependencias

El módulo requiere únicamente:

```bash
pip install numpy matplotlib
```

Para importarlo desde un script ubicado fuera del directorio del proyecto, agregar la ruta raíz al path de Python:

```python
import sys
sys.path.insert(0, '/ruta/al/proyecto')

from lib.generadores.JohanssonBalkenius import ConfigurableJBS, DEFAULT_CONFIG
```

---

## 5.2 Uso básico — Topología estándar J&B

El caso más simple usa la topología completa del paper con parámetros por defecto (sin ruido, sin depleción):

```python
from lib.generadores.JohanssonBalkenius import ConfigurableJBS

# 1. Crear sistema con topología estándar J&B 2018
sistema = ConfigurableJBS("default")

# 2. Ejecutar pasos de simulación con una luminancia constante
for i in range(5000):                 # 5000 pasos × 2 ms = 10 s de simulación
    sistema.step_simulation(L=0.5)    # L normalizada [0-1]: 0=oscuridad, 1=máximo

# 3. Obtener el diámetro pupilar al final
diametro_izq = sistema.get_output("pupil_left")   # en mm
diametro_der = sistema.get_output("pupil_right")  # en mm
print(f"Pupila izq: {diametro_izq:.2f} mm | der: {diametro_der:.2f} mm")
```

---

## 5.3 Flujo detallado — Creación, estabilización y simulación

En experimentos reales conviene **estabilizar** el sistema antes de comenzar la simulación experimental, de la misma forma que en un experimento real se espera a que el sujeto se adapte a las condiciones basales:

```python
import copy
import numpy as np
from lib.generadores.JohanssonBalkenius import ConfigurableJBS, DEFAULT_CONFIG

# ---------------------------------------------------------------------------
# PASO 1: Configurar el sistema
# ---------------------------------------------------------------------------

# Partir de la configuración estándar y modificar solo los parámetros deseados
config = copy.deepcopy(DEFAULT_CONFIG)

# Activar perturbaciones biológicas (ver Sección 4)
config['default_tau_rec']    = 0.5   # Tiempo de recuperación del pool vesicular (s)
config['default_U']          = 0.8   # Fracción de uso por actividad
config['default_sigma']      = 0.6   # Amplitud del ruido de canal
config['default_tau_jitter'] = 0.2   # Variabilidad temporal del retardo
config['default_jitter_epsilon'] = 0.05  # Variabilidad en constante de tiempo

# Crear el sistema con parámetros pupilares del sujeto (o valores por defecto)
sistema = ConfigurableJBS(
    system_config = config,
    D_min  = 4.99,   # mm — constricción máxima
    D_max  = 8.63,   # mm — dilatación máxima
    D_ref  = 5.87,   # mm — diámetro basal mesopio
    L_gain = 0.6458, # ganancia retinal
    L_gamma= 0.5127  # compresión fotópica
)

# Activar ruido de fuente lumínica (opcional)
sistema.config('enable_bilateral_noise', True)   # diferencia inter-ocular
sistema.config('enable_stochastic_noise', True)  # fluctuación de la fuente

# ---------------------------------------------------------------------------
# PASO 2: Estabilizar con luminancia de fondo
# ---------------------------------------------------------------------------
# Se deshabilita el historial durante la estabilización para no guardar datos
# que no son parte del experimento.

L_fondo = 0.0           # oscuridad total como condición inicial
T_estabilizacion = 60   # segundos — tiempo suficiente para el estado estacionario

sistema.enable_history = False                             # desactivar registro
n_pasos = int(round(T_estabilizacion / sistema.dt))

for _ in range(n_pasos):
    sistema.step_simulation(L=L_fondo)

# Resetear el reloj y reactivar el historial
sistema.t = 0.0
sistema.enable_history = True

# ---------------------------------------------------------------------------
# PASO 3: Ejecutar el experimento
# ---------------------------------------------------------------------------
# En este ejemplo se aplican escalones de luminancia secuenciales.

ESCALONES = [
    (5.0,  0.0),   # t=0–5 s:   oscuridad
    (15.0, 0.3),   # t=5–15 s:  luz media
    (25.0, 0.7),   # t=15–25 s: luz alta
    (30.0, 0.1),   # t=25–30 s: luz baja
]

T_total = 30.0
n_pasos = int(round(T_total / sistema.dt))

registros = {'tiempo': [], 'pupila_izq': [], 'pupila_der': []}

for _ in range(n_pasos):
    t = sistema.t

    # Determinar nivel de luz según escalón actual
    L_actual = next(
        (nivel for t_fin, nivel in ESCALONES if t <= t_fin),
        0.0
    )

    # Ejecutar un paso de simulación
    sistema.step_simulation(L=L_actual)

    # Guardar resultados
    registros['tiempo'].append(t)
    registros['pupila_izq'].append(sistema.get_output("pupil_left"))
    registros['pupila_der'].append(sistema.get_output("pupil_right"))

# ---------------------------------------------------------------------------
# PASO 4: Inspeccionar el estado
# ---------------------------------------------------------------------------

# Estado completo del último paso
estado = sistema.view_state("all")
print(f"Tiempo: {estado['system_info']['time']:.3f} s")
print(f"Pupila izq: {estado['pupil_left']:.2f} mm")
print(f"Pupila der: {estado['pupil_right']:.2f} mm")

# Acceso a un núcleo específico
ewpg_l = sistema.get_box('EWpg_l')
print(f"EWpg_l — estado x: {ewpg_l.x:.4f}, salida o: {ewpg_l.o:.4f}")
```

---

## 5.4 Extensión de la topología — Añadir un nodo externo

La configuración del sistema es un diccionario de Python, por lo que puede modificarse antes de instanciar el sistema para agregar núcleos o conexiones no presentes en el paper original.

El siguiente ejemplo (tomado del test A05) agrega un nodo `disturbance` que simula una perturbación externa conectada inhibitoriamente sobre EWpg. Esto permite inyectar señales arbitrarias en el sistema sin modificar la topología base:

```python
import copy
from lib.generadores.JohanssonBalkenius import ConfigurableJBS, DEFAULT_CONFIG

# Partir de la configuración estándar
config = copy.deepcopy(DEFAULT_CONFIG)

# --- Añadir el núcleo perturbador ---
# beta=0 y gamma=0 porque este nodo no recibe conexiones del sistema;
# solo recibe la señal externa a través de su alpha.
config['boxes'].append({
    'name': 'disturbance',
    'alpha': 0.0,
    'beta': 0,
    'gamma': 0
})

# --- Conectarlo como inhibitorio sobre EWpg ---
# Esto permite que una señal negativa en 'disturbance' reduzca la actividad
# del núcleo parasimpático, simulando dilatación por perturbación externa.
config['connections'].append({
    'from': 'disturbance',
    'to': 'EWpg_l',
    'tipo': 'inhibitory'
})
config['connections'].append({
    'from': 'disturbance',
    'to': 'EWpg_r',
    'tipo': 'inhibitory'
})

# Instanciar el sistema con la topología extendida
sistema = ConfigurableJBS(config)

# Registrar la entrada externa
sistema.define_input("disturbance", ['disturbance'])

# En cada paso, inyectar una señal de ruido gaussiano
import numpy as np
for i in range(5000):
    ruido = np.random.normal(0, 0.5)         # ruido con SD=0.5
    sistema.step_simulation(
        L=0.3,
        inputs={'disturbance': ruido}         # se pasa como alpha del nodo
    )
```

---

## 5.5 Guardar y cargar pesos aprendidos

Para experimentos de condicionamiento (AMY y CB), los pesos aprendidos pueden exportarse y reutilizarse en una sesión posterior:

```python
import json

# --- Exportar después del entrenamiento ---
pesos = sistema.export_weights()
# pesos tiene la forma: {'AMY': {0: 0.23, 1: 0.47}, 'CB': {0: 0.11, ...}}

with open('pesos_entrenados.json', 'w') as f:
    json.dump(pesos, f)

# --- Cargar en una nueva instancia ---
sistema_nuevo = ConfigurableJBS("default")

with open('pesos_entrenados.json', 'r') as f:
    pesos_cargados = json.load(f)

# json convierte las claves a string; convertir de vuelta a int
pesos_int = {
    caja: {int(k): v for k, v in ws.items()}
    for caja, ws in pesos_cargados.items()
}
sistema_nuevo.load_weights(pesos_int)
```

---

## 5.6 Referencia rápida de parámetros configurables

### Parámetros de `ConfigurableJBS`

| Parámetro | Unidad | Descripción |
|-----------|--------|-------------|
| `D_min` | mm | Diámetro pupilar mínimo |
| `D_max` | mm | Diámetro pupilar máximo |
| `D_ref` | mm | Diámetro basal de referencia |
| `L_gain` | — | Ganancia de la señal retinal |
| `L_gamma` | — | Exponente de compresión fotópica (< 1 = compresión) |

### Parámetros de `system_config` (nivel de sistema)

| Clave | Unidad | Descripción |
|-------|--------|-------------|
| `dt` | s | Paso de integración |
| `default_epsilon` | — | Constante de tiempo de los núcleos |
| `default_jitter_epsilon` | — | Variabilidad de ε entre núcleos |
| `default_tau` | s | Retardo de transmisión por defecto |
| `default_tau_rec` | s | Constante de recuperación sináptica |
| `default_U` | — | Fracción de uso del pool vesicular |
| `default_sigma` | — | Amplitud del ruido de canal |
| `default_tau_jitter` | — | Variabilidad relativa del retardo |

### Opciones de `sistema.config(key, value)`

| Clave | Tipo | Descripción |
|-------|------|-------------|
| `enable_bilateral_noise` | bool | Ruido aditivo entre ojo izquierdo y derecho |
| `enable_stochastic_noise` | bool | Ruido aditivo compartido en la fuente lumínica |
| `enable_emotional` | bool | Activar entrada emocional según umbral |
| `emotional_threshold` | float | Umbral de luminancia para activar cortex_emotional |
| `L_MIN` | float (cd/m²) | Luminancia mínima del rango de mapeo |
| `L_MAX` | float (cd/m²) | Luminancia máxima del rango de mapeo |
| `L_BILATERAL_NOISE_STD` | float | SD del ruido bilateral (fracción de L) |
| `L_STOCHASTIC_NOISE_STD` | float | SD del ruido estocástico de fuente (fracción de L) |
