import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from lib.generadores.pupila_base import simular_pupila
from lib.analisis.estadistica import calcular_media

# Configuración global
fs = 60.0
duracion = 30.0
t = np.linspace(0, duracion, int(fs * duracion)) 

#  Grafico

# TODO: funcion para graficar

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(t, simular_pupila(es_mv=True,t=t),  label="Migraña Vestibular (Patológico)", color='red', lw=1.2)
ax.plot(t, simular_pupila(es_mv=False,t=t), label="Control (No Patológico)", color='blue', lw=1.2, alpha=0.8)
ax.plot(t, simular_pupila(es_mv=True,t=t),  label="Migraña Vestibular (Patológico)", color='lightcoral', lw=1.2)
ax.plot(t, simular_pupila(es_mv=False,t=t), label="Control (No Patológico)", color='lightblue', lw=1.2, alpha=0.8)

ax.set_title('Simulacion de señal Hippus', fontsize=13)
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Diámetro [mm]')
ax.set_ylim([2.5, 5.5])
ax.legend(loc='upper right')


print(f"media MV: {calcular_media(simular_pupila(es_mv=True,t=t))}")
print(f"media Control: {calcular_media(simular_pupila(es_mv=False,t=t))}")

plt.show()
