import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
import random

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings('ignore')

def set_seed(seed=42):
    """
    Establece semillas aleatorias para garantizar que los resultados sean reproducibles.
    """
    random.seed(seed)
    np.random.seed(seed)

def simple_linear_regression(X_train, X_test, y_train, y_test, feature_name):
    """
    Realiza una regresión lineal simple con una sola variable y analiza el modelo.
    """
    # Extraer la característica específica
    X_train_simple = X_train[[feature_name]]
    X_test_simple = X_test[[feature_name]]
    
    # Ajustar el modelo
    model_simple = LinearRegression()
    model_simple.fit(X_train_simple, y_train)
    
    # Predicciones y evaluación
    y_pred_simple = model_simple.predict(X_test_simple)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_simple))
    mae = mean_absolute_error(y_test, y_pred_simple)
    r2 = r2_score(y_test, y_pred_simple)
    
    # Análisis estadístico con statsmodels
    X_sm = sm.add_constant(X_train_simple)
    model_sm = sm.OLS(y_train, X_sm).fit()
    
    # Visualización de la regresión y residuos
    # [Código de visualización aquí]
    
    return model_simple, rmse, mae, r2

def check_multicollinearity(X):
    """
    Calcula el Factor de Inflación de Varianza (VIF) para evaluar multicolinealidad.
    """
    X_with_const = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_with_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) 
                       for i in range(X_with_const.shape[1])]
    
    return vif_data

def optimized_model(X_train, X_test, y_train, y_test, feature_names, alpha=1.0):
    """
    Crea un modelo regularizado (Ridge) para reducir sobreajuste.
    """
    # Ajustar modelo regularizado
    model_ridge = Ridge(alpha=alpha)
    model_ridge.fit(X_train, y_train)
    
    # Evaluación
    y_pred_ridge = model_ridge.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
    mae = mean_absolute_error(y_test, y_pred_ridge)
    r2 = r2_score(y_test, y_pred_ridge)
    
    return model_ridge, rmse, mae, r2

def feature_engineering(df):
    """
    Realiza ingeniería de características para mejorar el modelo.
    """
    df_eng = df.copy()
    
    # Transformaciones logarítmicas para variables sesgadas
    df_eng['LogSalePrice'] = np.log1p(df_eng['SalePrice'])
    
    # Características derivadas importantes
    df_eng['TotalSF'] = df_eng['TotalBsmtSF'] + df_eng['1stFlrSF'] + df_eng['2ndFlrSF']
    df_eng['Age'] = df_eng['YrSold'] - df_eng['YearBuilt']
    df_eng['RemodAge'] = df_eng['YrSold'] - df_eng['YearRemodAdd']
    
    # Agregar TotalQuality como una combinación de OverallQual y OverallCond
    df_eng['TotalQuality'] = (df_eng['OverallQual'] * 2 + df_eng['OverallCond']) / 3
    
    return df_eng

def compare_models(models_info):
    """
    Compara los diferentes modelos según sus métricas.
    """
    # Extraer información para la comparación
    model_names = [info['name'] for info in models_info]
    rmse_values = [info['rmse'] for info in models_info]
    mae_values = [info['mae'] for info in models_info]
    r2_values = [info['r2'] for info in models_info]
    
    # Crear DataFrame para la comparación
    comparison_df = pd.DataFrame({
        'Modelo': model_names,
        'RMSE': rmse_values,
        'MAE': mae_values,
        'R²': r2_values
    })
    
    # Visualización de comparación de modelos
    plt.figure(figsize=(12, 8))
    sns.barplot(x='RMSE', y='Modelo', data=comparison_df, orient='h', palette='viridis')
    plt.title('Comparación de Modelos por RMSE')
    plt.xlabel('RMSE')
    plt.ylabel('Modelo')
    plt.tight_layout()
    plt.show()
    
    return comparison_df

def plot_saleprice_distribution(df):
    """
    Genera un histograma con curva de densidad (KDE) para la variable SalePrice.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df['SalePrice'], kde=True, bins=30, color='skyblue')
    plt.title('Distribución de SalePrice')
    plt.xlabel('Precio de Venta')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df):
    """
    Genera un heatmap que muestra la matriz de correlación
    entre las variables numéricas del DataFrame.
    """
    # Seleccionar solo columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.set_style("white")
    
    plt.figure(figsize=(15, 12))
    
    sns.heatmap(corr_matrix, 
                mask=mask,
                cmap='RdBu_r',  
                vmin=-1, vmax=1,  
                center=0,  
                square=True,  
                annot=True,  
                fmt='.2f',  
                cbar_kws={'label': 'Coeficiente de Correlación'},
                annot_kws={'size': 7},  
                linewidths=0.5)  
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.title('Matriz de Correlación entre Variables Numéricas', pad=20, size=14)
    plt.tight_layout()
    
    plt.show()

def plot_elbow_method(X_pca):
    """
    Genera el gráfico del método del codo (inercia vs número de clusters)
    para determinar el k óptimo en K-Means.
    """
    inertia = []
    K_range = range(1, 10)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_pca)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertia, marker='o', linestyle='--', color='teal')
    plt.title('Método del Codo para Determinar el Número Óptimo de Clusters')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Inercia')
    plt.tight_layout()
    plt.show()

def plot_clusters(X_pca, clusters):
    """
    Genera un scatter plot de los datos en el espacio reducido por PCA (2D),
    coloreando cada punto según el cluster asignado por K-Means.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis', s=100)
    plt.title('Clusters de Viviendas (PCA)')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()

def plot_real_vs_predicted(y_test, y_pred):
    """
    Genera un scatter plot comparando los valores reales de SalePrice
    vs los valores predichos por el modelo.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='purple', alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Valor Real de SalePrice')
    plt.ylabel('Valor Predicho de SalePrice')
    plt.title('Comparación: Valores Reales vs. Predichos')
    plt.tight_layout()
    plt.show()

def analyze_clusters(df, features, clusters, n_clusters=3):
    """
    Analiza las características de cada cluster identificado.
    
    Args:
        df: DataFrame con los datos originales
        features: Lista de características a analizar por cluster
        clusters: Array con las etiquetas de cluster asignadas
        n_clusters: Número total de clusters
    """
    # Crear una copia del DataFrame con la asignación de clusters
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = clusters
    
    # Imprimir estadísticas por cluster
    print("\nEstadísticas por Cluster:")
    for i in range(n_clusters):
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == i]
        print(f"\nCluster {i} (n={len(cluster_data)}):")
        print(cluster_data[features].describe().round(2).loc[['mean', 'std', 'min', 'max']])
    
    plt.figure(figsize=(14, 8))
    
    # Calcular medias por cluster para cada característica
    cluster_means = df_with_clusters.groupby('Cluster')[features].mean()
    
    # Normalizar para mejor visualización
    scaler = StandardScaler()
    cluster_means_scaled = pd.DataFrame(
        scaler.fit_transform(cluster_means),
        index=cluster_means.index,
        columns=cluster_means.columns
    )
    
    # Graficar heatmap de medias por cluster
    sns.heatmap(
        cluster_means_scaled, 
        annot=True, 
        cmap='coolwarm', 
        fmt='.2f',
        linewidths=0.5
    )
    plt.title('Características Promedio por Cluster (Valores Normalizados)')
    plt.ylabel('Cluster')
    plt.tight_layout()
    plt.savefig('cluster_characteristics.png')
    plt.show()
    
    # Boxplot de precios por cluster
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster', y='SalePrice', data=df_with_clusters)
    plt.title('Distribución de Precios por Cluster')
    plt.ylabel('Precio de Venta ($)')
    plt.xlabel('Cluster')
    plt.tight_layout()
    plt.savefig('price_by_cluster.png')
    plt.show()
    
    # Descripción de cada cluster
    print("\nInterpretación de Clusters:")
    cluster_interpretation = []
    
    # Comparar con la media global
    global_means = df[features].mean()
    
    for i in range(n_clusters):
        cluster_mean = cluster_means.loc[i]
        
        highlights_high = [feat for feat in features if cluster_mean[feat] > global_means[feat] * 1.2]
        highlights_low = [feat for feat in features if cluster_mean[feat] < global_means[feat] * 0.8]
        
        print(f"\nCluster {i}:")
        print(f"  Tamaño: {len(df_with_clusters[df_with_clusters['Cluster'] == i])} viviendas")
        print(f"  Precio promedio: ${cluster_mean['SalePrice']:.2f}")
        
        if highlights_high:
            print(f"  Características destacadas (por encima de la media):")
            for feat in highlights_high:
                print(f"    - {feat}: {cluster_mean[feat]:.2f} (media global: {global_means[feat]:.2f})")
        
        if highlights_low:
            print(f"  Características destacadas (por debajo de la media):")
            for feat in highlights_low:
                print(f"    - {feat}: {cluster_mean[feat]:.2f} (media global: {global_means[feat]:.2f})")
        
        if 'SalePrice' in cluster_mean.index:
            if cluster_mean['SalePrice'] > global_means['SalePrice'] * 1.2:
                print("  → Este cluster representa viviendas de alto valor en el mercado.")
            elif cluster_mean['SalePrice'] < global_means['SalePrice'] * 0.8:
                print("  → Este cluster representa viviendas de bajo valor en el mercado.")
            else:
                print("  → Este cluster representa viviendas de valor medio en el mercado.")

def multiple_linear_regression(X_train, X_test, y_train, y_test, feature_names):
    """
    Realiza una regresión lineal múltiple y analiza el modelo.
    
    Args:
        X_train: Conjunto de características de entrenamiento
        X_test: Conjunto de características de prueba
        y_train: Variable objetivo de entrenamiento
        y_test: Variable objetivo de prueba
        feature_names: Nombres de las características
        
    Returns:
        model_multiple: Modelo de regresión lineal múltiple ajustado
        rmse: Error cuadrático medio de raíz
        mae: Error absoluto medio
        r2: Coeficiente de determinación
    """
    print(f"\n\n{'=' * 50}")
    print(f"REGRESIÓN LINEAL MÚLTIPLE")
    print(f"{'=' * 50}")
    
    # Ajustar el modelo
    model_multiple = LinearRegression()
    model_multiple.fit(X_train, y_train)
    
    # Coeficientes
    print("Intercepto:", model_multiple.intercept_)
    print("\nCoeficientes:")
    for i, feature in enumerate(feature_names):
        print(f"  {feature}: {model_multiple.coef_[i]:.4f}")
    
    # Predicciones
    y_pred_train = model_multiple.predict(X_train)
    y_pred_test = model_multiple.predict(X_test)
    
    # Métricas en conjunto de prueba
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    
    # Métricas en conjunto de entrenamiento (para detectar sobreajuste)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)
    
    print("\nMétricas de desempeño en conjunto de prueba:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R²: {r2:.4f}")
    
    print("\nMétricas de desempeño en conjunto de entrenamiento:")
    print(f"  RMSE: {rmse_train:.2f}")
    print(f"  R²: {r2_train:.4f}")
    
    # Análisis de sobreajuste
    if r2_train - r2 > 0.1:
        print("\n⚠️ Posible sobreajuste detectado: R² en entrenamiento es significativamente mayor que en prueba.")
    
    X_sm = sm.add_constant(X_train)
    model_sm = sm.OLS(y_train, X_sm).fit()
    print("\nResumen estadístico del modelo:")
    print(model_sm.summary())
    
    # Análisis de residuos
    residuals = y_test - y_pred_test
    
    # Gráfico de residuos
    plt.figure(figsize=(12, 8))
    
    # Histograma de residuos
    plt.subplot(2, 2, 1)
    sns.histplot(residuals, kde=True, color='skyblue')
    plt.title('Distribución de Residuos')
    plt.xlabel('Residuo')
    
    plt.subplot(2, 2, 2)
    plt.scatter(y_pred_test, residuals, alpha=0.5, color='purple')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Residuos vs Valores Predichos')
    plt.xlabel('Valores Predichos')
    plt.ylabel('Residuos')
    
    plt.subplot(2, 2, 3)
    stats.probplot(residuals, plot=plt)
    plt.title('QQ Plot de Residuos')
    
    plt.subplot(2, 2, 4)
    plt.scatter(y_test, y_pred_test, alpha=0.5, color='green')
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title('Valores Reales vs. Predichos')
    plt.xlabel('Valores Reales')
    plt.ylabel('Valores Predichos')
    
    plt.tight_layout()
    plt.savefig('residuals_multiple.png')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, color='purple', alpha=0.6)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('Valor Real de SalePrice')
    plt.ylabel('Valor Predicho de SalePrice')
    plt.title('Comparación: Valores Reales vs. Predichos')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('real_vs_predicted_multiple.png')
    plt.show()
    
    return model_multiple, rmse, mae, r2

def main():
    # Sirve para que los resultados sean reproducibles
    set_seed(42)
    
    # -------------------------------------------------------------------------
    # 1. Carga de Datos
    # -------------------------------------------------------------------------
    try:
        df = pd.read_csv('train.csv')
        print("Dataset cargado correctamente.")
    except FileNotFoundError:
        print("El archivo 'train.csv' no se encontró. Asegúrate de tenerlo en el directorio de trabajo.")
        return
    
    print("Dimensiones del dataset:", df.shape)
    print(df.head(), "\n")
    
    # -------------------------------------------------------------------------
    # 2. Análisis Exploratorio de Datos (EDA)
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print("ANÁLISIS EXPLORATORIO DE DATOS")
    print("="*50)
    
    # 2.1. Distribución de SalePrice
    plot_saleprice_distribution(df)
    
    # 2.2. Matriz de Correlación
    plot_correlation_heatmap(df)
    
    # -------------------------------------------------------------------------
    # 3. Análisis de Clustering 
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print("ANÁLISIS DE AGRUPAMIENTO (CLUSTERING)")
    print("="*50)
    
    features_cluster = ['OverallQual', 'GrLivArea', 'YearBuilt', 'TotalBsmtSF']
    df_cluster = df[features_cluster].dropna()
    
    # Escalar datos para clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)
    
    # Reducir dimensionalidad para visualización
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Método del Codo para determinar número óptimo de clusters
    plot_elbow_method(X_pca)
    
    # Realizar clustering con K=3
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)  # Usar datos escalados completos
    
    # Visualizar clusters
    plot_clusters(X_pca, clusters)
    
    # Analizar características de cada cluster
    analyze_clusters(df, features_cluster + ['SalePrice'], clusters, n_clusters=3)
    
    # -------------------------------------------------------------------------
    # 4. Preprocesamiento e Ingeniería de Características
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print("PREPROCESAMIENTO E INGENIERÍA DE CARACTERÍSTICAS")
    print("="*50)
    
    # Aplicar ingeniería de características
    df_engineered = feature_engineering(df)
    
    # Seleccionar características relevantes basadas en correlación y dominio
    features_selected = [
        'OverallQual',  # Calidad general de la casa
        'GrLivArea',    # Superficie habitable
        'TotalBsmtSF',  # Superficie del sótano
        'GarageCars',   # Capacidad del garaje
        'YearBuilt',    # Año de construcción
        'TotalSF',      # Superficie total (característica derivada)
        'Age',          # Edad de la vivienda (característica derivada)
        'TotalQuality'  # Calidad total ponderada (característica derivada)
    ]
    
    # Seleccionar datos y eliminar filas con valores faltantes
    df_model = df_engineered[features_selected + ['SalePrice']].copy()
    df_model = df_model.dropna()
    
    # Eliminar datos atípicos que pueden afectar el modelo
    df_model = df_model[df_model['GrLivArea'] < 4500]
    
    print(f"\nDimensiones después del preprocesamiento: {df_model.shape}")
    
    # -------------------------------------------------------------------------
    # 5. División de Datos en Entrenamiento y Prueba
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print("DIVISIÓN DE DATOS EN ENTRENAMIENTO Y PRUEBA")
    print("="*50)
    
    X = df_model[features_selected]
    y = df_model['SalePrice']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Conjunto de prueba: {X_test.shape[0]} muestras")
    
    # -------------------------------------------------------------------------
    # 6. Regresión Lineal Simple (Univariada)
    # -------------------------------------------------------------------------
    # Seleccionar la variable con mayor correlación con SalePrice (OverallQual)
    best_single_feature = 'OverallQual'
    
    # Ajustar modelo univariado
    model_simple, rmse_simple, mae_simple, r2_simple = simple_linear_regression(
        X_train, X_test, y_train, y_test, best_single_feature
    )
    
    # -------------------------------------------------------------------------
    # 7. Regresión Lineal Múltiple con Todas las Variables Numéricas
    # -------------------------------------------------------------------------
    # Ajustar modelo con todas las variables seleccionadas
    model_multiple, rmse_multiple, mae_multiple, r2_multiple = multiple_linear_regression(
        X_train, X_test, y_train, y_test, features_selected
    )
    
    # -------------------------------------------------------------------------
    # 8. Análisis de Multicolinealidad
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print("ANÁLISIS DE MULTICOLINEALIDAD")
    print("="*50)
    
    vif_data = check_multicollinearity(X_train)
    
    # Identificar variables con alta multicolinealidad (VIF > 10)
    high_vif_features = vif_data[vif_data['VIF'] > 10]['Variable'].tolist()
    
    if 'const' in high_vif_features:
        high_vif_features.remove('const')
    
    print(f"Variables con alta multicolinealidad: {high_vif_features}")
    
    # -------------------------------------------------------------------------
    # 9. Modelo Optimizado (para corregir multicolinealidad/sobreajuste)
    # -------------------------------------------------------------------------
    # Determinar si hay sobreajuste
    train_pred = model_multiple.predict(X_train)
    test_pred = model_multiple.predict(X_test)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    overfitting = train_r2 - test_r2 > 0.1  # Diferencia significativa entre R² de train y test
    
    print(f"\nDiferencia entre R² entrenamiento ({train_r2:.3f}) y prueba ({test_r2:.3f}): {train_r2 - test_r2:.3f}")
    
    if overfitting:
        print("Se detectó sobreajuste en el modelo. Aplicando regularización.")
    else:
        print("No se detectó sobreajuste significativo en el modelo.")
    
    # Crear un modelo regularizado para resolver multicolinealidad y/o sobreajuste
    model_ridge, rmse_ridge, mae_ridge, r2_ridge = optimized_model(
        X_train, X_test, y_train, y_test, features_selected, alpha=1.0
    )
    
    # -------------------------------------------------------------------------
    # 10. Aplicación de Modelos al Conjunto de Prueba
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print("EVALUACIÓN DE MODELOS EN CONJUNTO DE PRUEBA")
    print("="*50)
    
    # -------------------------------------------------------------------------
    # 11. Comparación de Modelos
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print("COMPARACIÓN DE MODELOS")
    print("="*50)
    
    models_info = [
        {'name': f'Regresión Simple ({best_single_feature})', 'rmse': rmse_simple, 'mae': mae_simple, 'r2': r2_simple},
        {'name': 'Regresión Múltiple', 'rmse': rmse_multiple, 'mae': mae_multiple, 'r2': r2_multiple},
        {'name': 'Ridge Regression', 'rmse': rmse_ridge, 'mae': mae_ridge, 'r2': r2_ridge}
    ]
    
    comparison_df = compare_models(models_info)
    
    # -------------------------------------------------------------------------
    # 12. Selección del Mejor Modelo
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print("SELECCIÓN DEL MEJOR MODELO")
    print("="*50)
    
    # Seleccionar el mejor modelo basado en RMSE
    best_model_idx = comparison_df['RMSE'].idxmin()
    best_model_name = comparison_df.loc[best_model_idx, 'Modelo']
    
    print(f"El mejor modelo basado en RMSE es: {best_model_name}")
    print(f"  RMSE: {comparison_df.loc[best_model_idx, 'RMSE']:.2f}")
    print(f"  MAE: {comparison_df.loc[best_model_idx, 'MAE']:.2f}")
    print(f"  R²: {comparison_df.loc[best_model_idx, 'R²']:.3f}")
    
    print("\nCONCLUSIÓN:")
    if best_model_name == f'Regresión Simple ({best_single_feature})':
        print(f"  El modelo univariado usando solamente '{best_single_feature}' ofrece el mejor balance entre simplicidad y precisión.")
    elif best_model_name == 'Regresión Múltiple':
        print("  El modelo de regresión múltiple con todas las variables seleccionadas ofrece la mejor precisión.")
    else:  # Ridge
        print("  El modelo regularizado (Ridge) ofrece el mejor balance entre precisión y control de sobreajuste.")

if __name__ == '__main__':
    main()
