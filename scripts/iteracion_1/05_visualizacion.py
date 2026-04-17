import pandas as pd
import matplotlib.pyplot as plt
import ast
import os

# Configuración de rutas
ruta_ranking = 'scripts/iteracion_1/metricas_completas.csv'
ruta_datos = 'data/processed/analisis_resultados.csv'

if not os.path.exists(ruta_ranking):
    print("Error: Ejecuta primero el script de clasificación para generar el ranking.")
    exit()

# Cargar ranking y seleccionar las TOP 3 variables
df_ranking = pd.read_csv(ruta_ranking)
top_vars = ast.literal_eval(df_ranking.iloc[0]['Variables'])

v1 = top_vars[0]
v2 = top_vars[1]
v3 = top_vars[2] if len(top_vars) > 2 else top_vars[0]

# Cargar datos
datos = pd.read_csv(ruta_datos)
datos.columns = datos.columns.str.strip()

# Crear el gráfico 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Separar grupos para asignar colores
mv = datos[datos['Diagnostico'] == 'Migraña Vestibular']
ctrl = datos[datos['Diagnostico'] == 'Control']

# Graficar cada grupo
ax.scatter(mv[v1], mv[v2], mv[v3], c='#440154', label='Migraña Vestibular', s=50, alpha=0.6, edgecolors='w')
ax.scatter(ctrl[v1], ctrl[v2], ctrl[v3], c='#22a884', label='Control', s=50, alpha=0.6, edgecolors='w')

# Configurar etiquetas y título
ax.set_title(f'Espacio 3D de Biomarcadores\nTop 3: {v1}, {v2} y {v3}', pad=20)
ax.set_xlabel(v1)
ax.set_ylabel(v2)
ax.set_zlabel(v3)

ax.legend()

# Ajustar ángulo de visión para mejor perspectiva
ax.view_init(elev=20, azim=45)

# Guardar resultado
if not os.path.exists('docs'):
    os.makedirs('docs')

plt.tight_layout()
plt.savefig('docs/grafico_3D_ranking.png', dpi=300)
plt.show()

print(f"Gráfico 3D generado con: {v1}, {v2} y {v3}")
