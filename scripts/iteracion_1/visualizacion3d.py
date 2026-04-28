import pandas as pd
import matplotlib.pyplot as plt

# Resultados
df = pd.read_csv('data/processed/analisis_resultados.csv')
df.columns = df.columns.str.strip()

# Configurar el espacio 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Separar grupos
sanos = df[df['Diagnostico'] == 'Control']
enfermos = df[df['Diagnostico'] == 'Migraña Vestibular']

# 3 mejores variables según el ranking
ax.scatter(sanos['Dfi'], sanos['Velocidad_Media'], sanos['Frecuencia_Dom'], c='blue', label='Control', alpha=0.5)
ax.scatter(enfermos['Dfi'], enfermos['Velocidad_Media'], enfermos['Frecuencia_Dom'], c='orange', label='Migraña', alpha=0.5)

# Etiquetas de los ejes
ax.set_xlabel('Dfi (Diámetro Final)')
ax.set_ylabel('Velocidad Media')
ax.set_zlabel('Frecuencia Dominante')
ax.set_title('Top 3 Variables del Ranking')
ax.legend()

plt.show()