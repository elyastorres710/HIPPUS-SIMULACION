import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los resultados que calculamos antes
datos = pd.read_csv('data/processed/analisis_resultados.csv')
datos.columns = datos.columns.str.strip()

# Configurar el gráfico (algo sencillo)
sns.set_theme(style="ticks")
plt.figure(figsize=(9, 6))

# Graficar Desviación vs PUI para ver si los grupos se mezclan o no
sns.scatterplot(
    data=datos, 
    x='Desviacion', 
    y='PUI', 
    hue='Diagnostico',
    alpha=0.7
)

# Títulos y etiquetas claras para la tesis
plt.title('Distribución de Pacientes: Desviación vs PUI')
plt.xlabel('Desviación Estándar (Varianza)')
plt.ylabel('PUI (Inestabilidad Pupilar)')

# Guardar la imagen para ponerla en el informe
plt.savefig('docs/grafico_analisis.png')
plt.show()

print("El gráfico se guardó en la carpeta docs.")