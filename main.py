import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
import random

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from naive_bayes_models import naive_bayes_regression, naive_bayes_classification, tune_naive_bayes_regression, \
    tune_naive_bayes_classification, naive_bayes_cross_validation

import warnings

warnings.filterwarnings('ignore')


def set_seed(seed=42):
    """Establece semillas aleatorias para garantizar la reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)


def simple_linear_regression(X_train, X_test, y_train, y_test, feature_name):
    """Realiza una regresión lineal simple y muestra resumen y gráfica."""
    X_train_simple = X_train[[feature_name]]
    X_test_simple = X_test[[feature_name]]

    model_simple = LinearRegression()
    model_simple.fit(X_train_simple, y_train)

    y_pred_simple = model_simple.predict(X_test_simple)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_simple))
    mae = mean_absolute_error(y_test, y_pred_simple)
    r2 = r2_score(y_test, y_pred_simple)

    X_sm = sm.add_constant(X_train_simple)
    model_sm = sm.OLS(y_train, X_sm).fit()
    print(model_sm.summary())

    plt.figure(figsize=(8, 6))
    plt.scatter(X_test_simple, y_test, color='blue', alpha=0.6, label='Datos Reales')
    plt.plot(X_test_simple, y_pred_simple, color='red', label='Recta de Regresión')
    plt.title(f'Regresión Lineal Simple: {feature_name} vs. SalePrice')
    plt.xlabel(feature_name)
    plt.ylabel('SalePrice')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model_simple, rmse, mae, r2


def check_multicollinearity(X):
    """Calcula el VIF para detectar multicolinealidad."""
    X_with_const = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_with_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i)
                       for i in range(X_with_const.shape[1])]
    return vif_data


def optimized_model(X_train, X_test, y_train, y_test, feature_names, alpha=1.0):
    """Ajusta un modelo Ridge para mitigar sobreajuste y multicolinealidad."""
    model_ridge = Ridge(alpha=alpha)
    model_ridge.fit(X_train, y_train)

    y_pred_ridge = model_ridge.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
    mae = mean_absolute_error(y_test, y_pred_ridge)
    r2 = r2_score(y_test, y_pred_ridge)

    return model_ridge, rmse, mae, r2


def feature_engineering(df):
    """Realiza transformaciones y crea variables derivadas."""
    df_eng = df.copy()
    df_eng['LogSalePrice'] = np.log1p(df_eng['SalePrice'])
    df_eng['TotalSF'] = df_eng['TotalBsmtSF'] + df_eng['1stFlrSF'] + df_eng['2ndFlrSF']
    df_eng['Age'] = df_eng['YrSold'] - df_eng['YearBuilt']
    df_eng['RemodAge'] = df_eng['YrSold'] - df_eng['YearRemodAdd']
    df_eng['TotalQuality'] = (df_eng['OverallQual'] * 2 + df_eng['OverallCond']) / 3
    return df_eng


def create_price_category(df, price_column='SalePrice'):
    """
    Crea la variable categórica 'PriceCategory' con los siguientes límites:
      - Económicas: Precio < 150,000
      - Intermedias: 150,000 <= Precio <= 300,000
      - Caras: Precio > 300,000
    La elección se basa en la distribución observada de precios.
    """
    conditions = [
        (df[price_column] < 150000),
        (df[price_column] >= 150000) & (df[price_column] <= 300000),
        (df[price_column] > 300000)
    ]
    choices = ['Económicas', 'Intermedias', 'Caras']
    df['PriceCategory'] = pd.Series(np.select(conditions, choices, default='Desconocido'), dtype='object')
    return df


def plot_feature_engineering_effects(df):
    """Compara la distribución original de SalePrice y LogSalePrice."""
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df['SalePrice'], bins=30, kde=True, color='skyblue')
    plt.title('Distribución de SalePrice (Original)')
    plt.xlabel('SalePrice')
    plt.subplot(1, 2, 2)
    sns.histplot(df['LogSalePrice'], bins=30, kde=True, color='lightgreen')
    plt.title('Distribución de Log(SalePrice)')
    plt.xlabel('Log(SalePrice)')
    plt.tight_layout()
    plt.show()


def decision_tree_regression(X_train, X_test, y_train, y_test, max_depth=None):
    """Ajusta un árbol de decisión para regresión y muestra gráfica de desempeño."""
    dt_reg = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    dt_reg.fit(X_train, y_train)
    y_pred = dt_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Árbol de Decisión (max_depth={max_depth}) - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='darkorange', alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f"Árbol de Decisión para Regresión (max_depth={max_depth})")
    plt.xlabel("Valor Real de SalePrice")
    plt.ylabel("Valor Predicho de SalePrice")
    plt.tight_layout()
    plt.show()

    return dt_reg, rmse, mae, r2


def decision_tree_classifier(X_train, X_test, y_train, y_test, max_depth=None):
    """Ajusta un árbol de decisión para clasificación usando PriceCategory y muestra matriz de confusión y árbol."""
    # Eliminar filas con valores NaN
    mask = ~(X_train.isna().any(axis=1) | y_train.isna())
    X_train_clean = X_train[mask]
    y_train_clean = y_train[mask]

    mask_test = ~(X_test.isna().any(axis=1) | y_test.isna())
    X_test_clean = X_test[mask_test]
    y_test_clean = y_test[mask_test]

    dt_clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt_clf.fit(X_train_clean, y_train_clean)
    y_pred = dt_clf.predict(X_test_clean)

    cm = confusion_matrix(y_test_clean, y_pred)
    cr = classification_report(y_test_clean, y_pred)

    print(f"Árbol de Decisión Clasificador (max_depth={max_depth})")
    print("Matriz de Confusión:")
    print(cm)
    print("Reporte de Clasificación:")
    print(cr)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Matriz de Confusión - Árbol Clasificador (max_depth={max_depth})")
    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    plot_tree(dt_clf, feature_names=X_train.columns, class_names=dt_clf.classes_, filled=True, rounded=True)
    plt.title(f"Árbol de Decisión - Clasificador (max_depth={max_depth})")
    plt.tight_layout()
    plt.show()

    return dt_clf, cm, cr


def random_forest_regression(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=None):
    """Ajusta un modelo de Random Forest para regresión y muestra desempeño gráfico."""
    rf_reg = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_reg.fit(X_train, y_train)
    y_pred = rf_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(
        f"Random Forest Regressor (n_estimators={n_estimators}, max_depth={max_depth}) - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='darkgreen', alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f"Random Forest para Regresión (max_depth={max_depth})")
    plt.xlabel("Valor Real de SalePrice")
    plt.ylabel("Valor Predicho de SalePrice")
    plt.tight_layout()
    plt.show()

    return rf_reg, rmse, mae, r2


def random_forest_classifier(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=None):
    """Ajusta un modelo de Random Forest para clasificación y muestra matriz de confusión."""
    # Eliminar filas con valores NaN
    mask = ~(X_train.isna().any(axis=1) | y_train.isna())
    X_train_clean = X_train[mask]
    y_train_clean = y_train[mask]

    mask_test = ~(X_test.isna().any(axis=1) | y_test.isna())
    X_test_clean = X_test[mask_test]
    y_test_clean = y_test[mask_test]

    rf_clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_clf.fit(X_train_clean, y_train_clean)
    y_pred = rf_clf.predict(X_test_clean)

    cm = confusion_matrix(y_test_clean, y_pred)
    cr = classification_report(y_test_clean, y_pred)

    print(f"Random Forest Clasificador (n_estimators={n_estimators}, max_depth={max_depth})")
    print("Matriz de Confusión:")
    print(cm)
    print("Reporte de Clasificación:")
    print(cr)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title(f"Matriz de Confusión - Random Forest Clasificador (max_depth={max_depth})")
    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")
    plt.tight_layout()
    plt.show()

    return rf_clf, cm, cr


def plot_train_test_R2(train_r2, test_r2):
    """Genera gráfico de barras comparando R² en entrenamiento y prueba."""
    labels = ['Entrenamiento', 'Prueba']
    values = [train_r2, test_r2]
    plt.figure(figsize=(6, 4))
    sns.barplot(x=labels, y=values, palette='magma')
    plt.title('Comparación de R²: Entrenamiento vs. Prueba')
    plt.ylim(0, 1)
    plt.ylabel('R²')
    plt.tight_layout()
    plt.show()


def plot_real_vs_predicted(y_test, y_pred):
    """
    Genera un scatter plot comparando los valores reales de SalePrice
    con los valores predichos por el modelo.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='purple', alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Valor Real de SalePrice')
    plt.ylabel('Valor Predicho de SalePrice')
    plt.title('Comparación: Valores Reales vs. Predichos')
    plt.tight_layout()
    plt.show()


def compare_models(models_info):
    """Genera gráfico comparativo de RMSE entre modelos."""
    model_names = [info['name'] for info in models_info]
    rmse_values = [info['rmse'] for info in models_info]
    mae_values = [info['mae'] for info in models_info]
    r2_values = [info['r2'] for info in models_info]

    comparison_df = pd.DataFrame({
        'Modelo': model_names,
        'RMSE': rmse_values,
        'MAE': mae_values,
        'R²': r2_values
    })

    plt.figure(figsize=(12, 8))
    sns.barplot(x='RMSE', y='Modelo', data=comparison_df, orient='h', palette='viridis')
    plt.title('Comparación de Modelos por RMSE')
    plt.xlabel('RMSE')
    plt.ylabel('Modelo')
    plt.tight_layout()
    plt.show()

    return comparison_df


# ==================== NUEVAS GRÁFICAS ====================

def plot_regression_metrics_comparison(models_info, save_path=None):
    """
    Genera un gráfico comparativo (subplots) de las métricas RMSE, MAE y R²
    para los modelos de regresión.
    Los modelos_info deben ser una lista de diccionarios con las llaves:
      'name', 'rmse', 'mae', 'r2'
    """
    model_names = [info['name'] for info in models_info]
    rmse_values = [info['rmse'] for info in models_info]
    mae_values = [info['mae'] for info in models_info]
    r2_values = [info['r2'] for info in models_info]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.barplot(x=rmse_values, y=model_names, ax=axes[0], palette='viridis')
    axes[0].set_title('RMSE')
    axes[0].set_xlabel('RMSE')

    sns.barplot(x=mae_values, y=model_names, ax=axes[1], palette='magma')
    axes[1].set_title('MAE')
    axes[1].set_xlabel('MAE')

    sns.barplot(x=r2_values, y=model_names, ax=axes[2], palette='coolwarm')
    axes[2].set_title('R²')
    axes[2].set_xlabel('R²')
    axes[2].set_xlim(0, 1)

    fig.suptitle('Comparación de Métricas de Regresión', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_cv_scores(cv_scores, save_path=None):
    """
    Genera un boxplot y un histograma de las puntuaciones obtenidas en la validación cruzada
    para el modelo de Naïve Bayes en regresión.
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.boxplot(y=cv_scores, color='lightblue')
    plt.title('Distribución de CV Scores (Boxplot)')
    plt.ylabel('Puntuación (R²)')

    plt.subplot(1, 2, 2)
    sns.histplot(cv_scores, bins=10, kde=True, color='salmon')
    plt.title('Distribución de CV Scores (Histograma)')
    plt.xlabel('Puntuación (R²)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_classification_accuracy_comparison(dt_cm, rf_cm, nb_cm, save_path=None):
    """
    Calcula la exactitud a partir de las matrices de confusión de tres modelos de clasificación
    (Árbol de Decisión, Random Forest y Naïve Bayes) y genera un gráfico de barras comparativo.
    """

    def accuracy_from_cm(cm):
        return np.trace(cm) / np.sum(cm)

    dt_acc = accuracy_from_cm(dt_cm)
    rf_acc = accuracy_from_cm(rf_cm)
    nb_acc = accuracy_from_cm(nb_cm)

    models = ['Árbol Decisión', 'Random Forest', 'Naïve Bayes']
    accuracies = [dt_acc, rf_acc, nb_acc]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=models, y=accuracies, palette='pastel')
    plt.title('Comparación de Exactitud de Modelos de Clasificación')
    plt.ylabel('Exactitud')
    plt.ylim(0, 1)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_saleprice_distribution(df):
    """Genera histograma y KDE para SalePrice."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df['SalePrice'], kde=True, bins=30, color='skyblue')
    plt.title('Distribución de SalePrice')
    plt.xlabel('SalePrice')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df):
    """Genera heatmap de la matriz de correlación (mitad inferior)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.set_style("white")
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', vmin=-1, vmax=1, center=0,
                square=True, annot=True, fmt='.2f', cbar_kws={'label': 'Coeficiente de Correlación'},
                annot_kws={'size': 7}, linewidths=0.5)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title('Matriz de Correlación entre Variables Numéricas', pad=20, size=14)
    plt.tight_layout()
    plt.show()


def plot_elbow_method(X_pca):
    """Genera gráfico del método del codo para K-Means."""
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
    """Genera scatter plot de clusters en espacio de PCA."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis', s=100)
    plt.title('Clusters de Viviendas (PCA)')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()


def analyze_clusters(df, features, clusters, n_clusters=3):
    """Analiza y grafica características de cada cluster."""
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = clusters

    print("\nEstadísticas por Cluster:")
    for i in range(n_clusters):
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == i]
        print(f"\nCluster {i} (n={len(cluster_data)}):")
        print(cluster_data[features].describe().round(2).loc[['mean', 'std', 'min', 'max']])

    plt.figure(figsize=(14, 8))
    cluster_means = df_with_clusters.groupby('Cluster')[features].mean()
    scaler = StandardScaler()
    cluster_means_scaled = pd.DataFrame(
        scaler.fit_transform(cluster_means),
        index=cluster_means.index,
        columns=cluster_means.columns
    )
    sns.heatmap(cluster_means_scaled, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Características Promedio por Cluster (Valores Normalizados)')
    plt.ylabel('Cluster')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster', y='SalePrice', data=df_with_clusters)
    plt.title('Distribución de Precios por Cluster')
    plt.ylabel('Precio de Venta ($)')
    plt.xlabel('Cluster')
    plt.tight_layout()
    plt.show()

    print("\nInterpretación de Clusters:")
    global_means = df[features].mean()
    for i in range(n_clusters):
        cluster_mean = cluster_means.loc[i]
        highlights_high = [feat for feat in features if cluster_mean[feat] > global_means[feat] * 1.2]
        highlights_low = [feat for feat in features if cluster_mean[feat] < global_means[feat] * 0.8]
        print(f"\nCluster {i}:")
        print(f"  Tamaño: {len(df_with_clusters[df_with_clusters['Cluster'] == i])} viviendas")
        print(f"  Precio promedio: ${cluster_mean['SalePrice']:.2f}")
        if highlights_high:
            print("  Características destacadas (por encima de la media):")
            for feat in highlights_high:
                print(f"    - {feat}: {cluster_mean[feat]:.2f} (media global: {global_means[feat]:.2f})")
        if highlights_low:
            print("  Características destacadas (por debajo de la media):")
            for feat in highlights_low:
                print(f"    - {feat}: {cluster_mean[feat]:.2f} (media global: {global_means[feat]:.2f})")
        if 'SalePrice' in cluster_mean.index:
            if cluster_mean['SalePrice'] > global_means['SalePrice'] * 1.2:
                print("  → Representa viviendas de alto valor.")
            elif cluster_mean['SalePrice'] < global_means['SalePrice'] * 0.8:
                print("  → Representa viviendas de bajo valor.")
            else:
                print("  → Representa viviendas de valor medio.")


def main():
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
    # 2. Análisis Exploratorio de Datos (EDA) - Primer Avance
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("ANÁLISIS EXPLORATORIO DE DATOS")
    print("=" * 50)
    plot_saleprice_distribution(df)
    plot_correlation_heatmap(df)

    # -------------------------------------------------------------------------
    # 3. Análisis de Agrupamiento (Clustering) - Primer Avance
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("ANÁLISIS DE AGRUPAMIENTO (CLUSTERING)")
    print("=" * 50)
    features_cluster = ['OverallQual', 'GrLivArea', 'YearBuilt', 'TotalBsmtSF']
    df_cluster = df[features_cluster].dropna()

    scaler = StandardScaler()
    X_scaled_cluster = scaler.fit_transform(df_cluster)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled_cluster)

    plot_elbow_method(X_pca)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled_cluster)
    plot_clusters(X_pca, clusters)
    analyze_clusters(df, features_cluster + ['SalePrice'], clusters, n_clusters=3)

    # -------------------------------------------------------------------------
    # 4. Preprocesamiento e Ingeniería de Características - Primer Avance
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("PREPROCESAMIENTO E INGENIERÍA DE CARACTERÍSTICAS")
    print("=" * 50)
    df_engineered = feature_engineering(df)

    # Nueva gráfica: Efecto de la transformación logarítmica en SalePrice
    plot_feature_engineering_effects(df_engineered)

    features_selected = [
        'OverallQual',
        'GrLivArea',
        'TotalBsmtSF',
        'GarageCars',
        'YearBuilt',
        'TotalSF',
        'Age',
        'TotalQuality'
    ]
    df_model = df_engineered[features_selected + ['SalePrice']].copy()
    df_model = df_model.dropna()
    df_model = df_model[df_model['GrLivArea'] < 4500]
    print(f"\nDimensiones después del preprocesamiento: {df_model.shape}")

    # -------------------------------------------------------------------------
    # 5. División de Datos en Entrenamiento y Prueba - Primer Avance
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("DIVISIÓN DE DATOS EN ENTRENAMIENTO Y PRUEBA")
    print("=" * 50)
    X = df_model[features_selected]
    y = df_model['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Conjunto de prueba: {X_test.shape[0]} muestras")

    print("\nDistribución de SalePrice en el conjunto de entrenamiento:")
    print(y_train.describe())
    print("\nDistribución de SalePrice en el conjunto de prueba:")
    print(y_test.describe())

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(y_train, bins=30, kde=True, color='skyblue')
    plt.title('Distribución de SalePrice - Entrenamiento')
    plt.xlabel('SalePrice')
    plt.ylabel('Frecuencia')
    plt.subplot(1, 2, 2)
    sns.histplot(y_test, bins=30, kde=True, color='salmon')
    plt.title('Distribución de SalePrice - Prueba')
    plt.xlabel('SalePrice')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 6. Modelo Univariado de Regresión - Primer Avance
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("REGRESIÓN LINEAL SIMPLE (MODELO UNIVARIADO)")
    print("=" * 50)
    best_single_feature = 'OverallQual'
    model_simple, rmse_simple, mae_simple, r2_simple = simple_linear_regression(
        X_train, X_test, y_train, y_test, best_single_feature
    )

    # -------------------------------------------------------------------------
    # 7. Modelo de Regresión Lineal Múltiple / Árbol de Decisión - Primer Avance
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("REGRESIÓN LINEAL MÚLTIPLE (Árbol de Decisión)")
    print("=" * 50)
    # Aquí se utiliza un árbol de decisión para regresión como ejemplo.
    model_multiple, rmse_multiple, mae_multiple, r2_multiple = decision_tree_regression(
        X_train, X_test, y_train, y_test, max_depth=5
    )

    # -------------------------------------------------------------------------
    # 8. Análisis de Multicolinealidad y Sobreajuste - Primer Avance
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("ANÁLISIS DE MULTICOLINEALIDAD")
    print("=" * 50)
    vif_data = check_multicollinearity(X_train)
    high_vif_features = vif_data[vif_data['VIF'] > 10]['Variable'].tolist()
    if 'const' in high_vif_features:
        high_vif_features.remove('const')
    print(f"Variables con alta multicolinealidad: {high_vif_features}")

    train_pred = model_multiple.predict(X_train)
    test_pred = model_multiple.predict(X_test)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    print(f"\nR² entrenamiento: {train_r2:.3f}")
    print(f"R² prueba: {test_r2:.3f}")
    if train_r2 - test_r2 > 0.1:
        print("⚠️ Se detectó sobreajuste.")
    else:
        print("No se detectó sobreajuste significativo.")

    plot_train_test_R2(train_r2, test_r2)

    # -------------------------------------------------------------------------
    # 9. Modelo Optimizado (Ridge) - Primer Avance
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("MODELO REGULARIZADO (RIDGE)")
    print("=" * 50)
    model_ridge, rmse_ridge, mae_ridge, r2_ridge = optimized_model(
        X_train, X_test, y_train, y_test, features_selected, alpha=1.0
    )

    # -------------------------------------------------------------------------
    # 10. Evaluación del Desempeño en el Conjunto de Prueba - Primer Avance
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("EVALUACIÓN DE MODELOS EN CONJUNTO DE PRUEBA")
    print("=" * 50)
    print("Modelo de Regresión Simple:")
    print(f"  RMSE: {rmse_simple:.2f}, MAE: {mae_simple:.2f}, R²: {r2_simple:.2f}")
    print("Modelo de Árbol de Decisión:")
    print(f"  RMSE: {rmse_multiple:.2f}, MAE: {mae_multiple:.2f}, R²: {r2_multiple:.2f}")
    print("Modelo Ridge:")
    print(f"  RMSE: {rmse_ridge:.2f}, MAE: {mae_ridge:.2f}, R²: {r2_ridge:.2f}")

    plot_real_vs_predicted(y_test, test_pred)

    # -------------------------------------------------------------------------
    # 11. Comparación de Modelos y Selección del Mejor - Primer Avance
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("COMPARACIÓN DE MODELOS")
    print("=" * 50)
    models_info = [
        {'name': f'Regresión Simple ({best_single_feature})', 'rmse': rmse_simple, 'mae': mae_simple, 'r2': r2_simple},
        {'name': 'Árbol de Decisión', 'rmse': rmse_multiple, 'mae': mae_multiple, 'r2': r2_multiple},
        {'name': 'Ridge Regression', 'rmse': rmse_ridge, 'mae': mae_ridge, 'r2': r2_ridge}
    ]
    comparison_df = compare_models(models_info)
    best_model_idx = comparison_df['RMSE'].idxmin()
    best_model_name = comparison_df.loc[best_model_idx, 'Modelo']
    print(f"El mejor modelo basado en RMSE es: {best_model_name}")
    print(f"  RMSE: {comparison_df.loc[best_model_idx, 'RMSE']:.2f}")
    print(f"  MAE: {comparison_df.loc[best_model_idx, 'MAE']:.2f}")
    print(f"  R²: {comparison_df.loc[best_model_idx, 'R²']:.3f}")

    # NUEVA GRÁFICA: Comparación directa de métricas de regresión (se guardará como 'regression_metrics_comparison.png')
    plot_regression_metrics_comparison(models_info, save_path='regression_metrics_comparison.png')

    # -------------------------------------------------------------------------
    # 12. Validación Cruzada para Naïve Bayes Regresión
    # -------------------------------------------------------------------------
    # Escalar características para Naïve Bayes
    scaler_cv = StandardScaler()
    X_train_scaled = scaler_cv.fit_transform(X_train)
    X_test_scaled = scaler_cv.transform(X_test)

    # Convertir arrays escalados de nuevo a DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    # Naive Bayes para Regresión
    print("\n" + "=" * 50)
    print("MODELOS DE NAIVE BAYES PARA REGRESIÓN")
    print("=" * 50)
    model_nb_reg, rmse_nb_reg, mae_nb_reg, r2_nb_reg = naive_bayes_regression(
        X_train_scaled, X_test_scaled, y_train, y_test, n_bins=15, var_smoothing=1e-9
    )

    # Validación cruzada para Naïve Bayes Regresión
    cv_scores = naive_bayes_cross_validation(X, y, n_bins=15, var_smoothing=1e-9, cv=5)
    print("Puntuaciones de Validación Cruzada (Naïve Bayes Regresión):", cv_scores)
    # NUEVA GRÁFICA: Distribución de los CV Scores (se guardará como 'cv_scores_distribution.png')
    plot_cv_scores(cv_scores, save_path='cv_scores_distribution.png')

    # Ajuste de hiperparámetros para Naïve Bayes Regresión
    print("\n" + "=" * 50)
    print("AJUSTE DE HIPERPARÁMETROS PARA NAIVE BAYES REGRESIÓN")
    print("=" * 50)
    best_nb_reg, best_rmse_nb_reg, best_mae_nb_reg, best_r2_nb_reg = tune_naive_bayes_regression(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    # -------------------------------------------------------------------------
    # 13. Creación de la Variable Categórica para Clasificación
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("CREACIÓN DE LA VARIABLE CATEGÓRICA (PriceCategory)")
    print("=" * 50)
    df_model = create_price_category(df_model, price_column='SalePrice')
    print("Distribución de la nueva variable categórica:")
    print(df_model['PriceCategory'].value_counts())

    # -------------------------------------------------------------------------
    # 14. Árboles de Decisión para Predicción y Clasificación
    # -------------------------------------------------------------------------
    # 14.1 Árbol de Decisión para Regresión
    print("\n" + "=" * 50)
    print("ÁRBOLES DE DECISIÓN PARA REGRESIÓN")
    print("=" * 50)
    depths = [3, 5, 7]
    dt_reg_results = {}
    for depth in depths:
        dt_model, dt_rmse, dt_mae, dt_r2 = decision_tree_regression(X_train, X_test, y_train, y_test, max_depth=depth)
        dt_reg_results[depth] = (dt_rmse, dt_mae, dt_r2)

    # 14.2 Árbol de Decisión para Clasificación (usando PriceCategory)
    print("\n" + "=" * 50)
    print("ÁRBOLES DE DECISIÓN PARA CLASIFICACIÓN")
    print("=" * 50)
    X_clf = df_model[features_selected]
    y_clf = df_model['PriceCategory']
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
    dt_clf, clf_cm, clf_cr = decision_tree_classifier(X_train_clf, X_test_clf, y_train_clf, y_test_clf, max_depth=5)

    # -------------------------------------------------------------------------
    # 15. Random Forest para Regresión y Clasificación
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("RANDOM FOREST PARA REGRESIÓN")
    print("=" * 50)
    rf_reg, rf_rmse, rf_mae, rf_r2 = random_forest_regression(X_train, X_test, y_train, y_test, n_estimators=100,
                                                              max_depth=7)

    print("\n" + "=" * 50)
    print("RANDOM FOREST PARA CLASIFICACIÓN")
    print("=" * 50)
    rf_clf, rf_cm, rf_cr = random_forest_classifier(X_train_clf, X_test_clf, y_train_clf, y_test_clf, n_estimators=100,
                                                    max_depth=7)

    # -------------------------------------------------------------------------
    # 16. Comparación Final de Algoritmos para Predicción del Precio
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("COMPARACIÓN FINAL DE ALGORITMOS PARA PREDICCIÓN DEL PRECIO")
    print("=" * 50)
    best_depth = min(dt_reg_results, key=lambda d: dt_reg_results[d][0])
    best_dt_rmse, best_dt_mae, best_dt_r2 = dt_reg_results[best_depth]
    print("\nResumen de Modelos para Predicción:")
    print(
        f"Regresión Lineal Simple ({best_single_feature}): RMSE={rmse_simple:.2f}, MAE={mae_simple:.2f}, R²={r2_simple:.2f}")
    print(
        f"Árbol de Decisión (max_depth={best_depth}): RMSE={best_dt_rmse:.2f}, MAE={best_dt_mae:.2f}, R²={best_dt_r2:.2f}")
    print(f"Random Forest (max_depth=7): RMSE={rf_rmse:.2f}, MAE={rf_mae:.2f}, R²={rf_r2:.2f}")
    models_pred_info = [
        {'name': f'Regresión Simple ({best_single_feature})', 'rmse': rmse_simple, 'mae': mae_simple, 'r2': r2_simple},
        {'name': f'Árbol de Decisión (max_depth={best_depth})', 'rmse': best_dt_rmse, 'mae': best_dt_mae,
         'r2': best_dt_r2},
        {'name': 'Random Forest', 'rmse': rf_rmse, 'mae': rf_mae, 'r2': rf_r2}
    ]

    # NUEVA GRÁFICA: Comparación directa de métricas de regresión incluyendo también los modelos NB,
    # se sugiere nombrarla "regression_metrics_comparison.png"
    # Para incluir los modelos NB, puedes extender la lista models_pred_info con:
    models_pred_info.extend([
        {'name': 'Naive Bayes (n_bins=15)', 'rmse': rmse_nb_reg, 'mae': mae_nb_reg, 'r2': r2_nb_reg},
        {'name': 'Naive Bayes Ajustado', 'rmse': best_rmse_nb_reg, 'mae': best_mae_nb_reg, 'r2': best_r2_nb_reg}
    ])
    plot_regression_metrics_comparison(models_pred_info, save_path='regression_metrics_comparison.png')

    # -------------------------------------------------------------------------
    # 17. Naive Bayes para Clasificación y Comparación de Exactitud
    # -------------------------------------------------------------------------
    # Escalar características para clasificación
    scaler_clf = StandardScaler()
    X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
    X_test_clf_scaled = scaler_clf.transform(X_test_clf)

    # Convertir arrays escalados de nuevo a DataFrames
    X_train_clf_scaled = pd.DataFrame(X_train_clf_scaled, columns=X_train_clf.columns, index=X_train_clf.index)
    X_test_clf_scaled = pd.DataFrame(X_test_clf_scaled, columns=X_test_clf.columns, index=X_test_clf.index)

    # Naive Bayes para Clasificación
    print("\n" + "=" * 50)
    print("MODELOS DE NAIVE BAYES PARA CLASIFICACIÓN")
    print("=" * 50)
    nb_clf, nb_cm, nb_cr = naive_bayes_classification(
        X_train_clf_scaled, X_test_clf_scaled, y_train_clf, y_test_clf, var_smoothing=1e-9
    )

    # Ajuste de hiperparámetros para Naive Bayes Clasificación
    print("\n" + "=" * 50)
    print("AJUSTE DE HIPERPARÁMETROS PARA NAIVE BAYES CLASIFICACIÓN")
    print("=" * 50)
    best_nb_clf, best_nb_cm, best_nb_cr = tune_naive_bayes_classification(
        X_train_clf_scaled, X_test_clf_scaled, y_train_clf, y_test_clf
    )

    # NUEVA GRÁFICA: Comparación de exactitud de modelos de clasificación.
    # Se calculará la exactitud a partir de las matrices de confusión de:
    # Árbol de Decisión (dt_clf), Random Forest (rf_clf) y Naïve Bayes (nb_clf)
    # Se sugiere nombrar esta gráfica "classification_accuracy_comparison.png"
    plot_classification_accuracy_comparison(clf_cm, rf_cm, nb_cm, save_path='classification_accuracy_comparison.png')


if __name__ == '__main__':
    main()
