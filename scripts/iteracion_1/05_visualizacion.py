import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os

# Cargar ranking para elegir los ejes automaticamente
ruta_ranking = 'scripts/iteracion_1/metricas_completas.csv'
ruta_datos = 'data/processed/analisis_resultados.csv'

if not os.path.exists(ruta_ranking):
    print("Error: Ejecuta primero el script de clasificacion para generar el ranking.")
    exit()

# Leer el top 1 del ranking
df_ranking = pd.read_csv(ruta_ranking)
top_vars = ast.literal_eval(df_ranking.iloc[0]['Variables'])

# Definir que variables graficar segun el ranking
v1 = top_vars[0]
v2 = top_vars[1] if len(top_vars) > 1 else top_vars[0]

# Cargar metricas de los pacientes
datos = pd.read_csv(ruta_datos)
datos.columns = datos.columns.str.strip()

# Configuracion del grafico
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

sns.scatterplot(
    data=datos,
    x=v1,
    y=v2,
    hue='Diagnostico',
    palette='viridis',
    alpha=0.8,
    s=80
)

# Etiquetas basadas en el ranking real
plt.title(f'Top 1 Ranking: {v1} vs {v2}')
plt.xlabel(f'{v1} (Mejor parametro)')
plt.ylabel(f'{v2} (Segundo mejor parametro)')

# Guardar resultado
if not os.path.exists('docs'):
    os.makedirs('docs')

plt.savefig('docs/grafico_top_ranking.png', dpi=300)
plt.show()

print(f"Grafico generado con: {v1} y {v2}")