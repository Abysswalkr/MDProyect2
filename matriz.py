import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def matriz():
    # 1. Carga del DataFrame
    try:
        df = pd.read_csv('train.csv')
    except FileNotFoundError:
        print("El archivo 'train.csv' no se encontró. Asegúrate de tenerlo en el directorio de trabajo.")
        return

    print("Dimensiones del dataset:", df.shape)
    print(df.head(), "\n")

    # 2. Seleccionar solo las columnas numéricas para la matriz de correlación
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()

    # 3. Crear una máscara para la parte superior de la matriz de correlación
    #    De esta forma, solo se muestra la diagonal y la parte inferior.
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # 4. Configuración del tamaño y estilo de la figura
    plt.figure(figsize=(12, 10))
    sns.set_style("white")  # Estilo de fondo, opcional

    # 5. Graficar el heatmap de la matriz de correlación
    #    - mask=mask: oculta la parte superior
    #    - annot=False: si deseas mostrar/ocultar valores numéricos
    #    - fmt=".2f": formato de las anotaciones
    #    - square=True: cada celda es cuadrada
    #    - linewidths=.5: línea divisoria entre celdas
    #    - cbar_kws={"shrink": .8}: ajusta el tamaño de la barra de color
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap='coolwarm',
        annot=True,  # Cambia a False si no deseas valores numéricos
        fmt=".2f",
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8}
    )

    plt.title('Matriz de Correlación (Mostrando Solo la Mitad Inferior)')
    plt.xticks(rotation=45)  # Rotar etiquetas del eje X para evitar superposición
    plt.yticks(rotation=0)  # Mantener etiquetas del eje Y horizontales
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    matriz()
