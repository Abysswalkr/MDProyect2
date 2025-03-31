import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, accuracy_score

import warnings

warnings.filterwarnings('ignore')


# --------------------- Funciones para NB Regresión ---------------------
def nb_regression(X_train, X_test, y_train, y_test, n_bins=15, var_smoothing=1e-9, save_plots=True):
    """
    Entrena un modelo de Naïve Bayes para regresión:
      - Discretiza la variable respuesta en n_bins.
      - Usa GaussianNB con el parámetro var_smoothing.
      - Calcula la esperanza condicional usando los centros de los bins correspondientes a las clases presentes.
    Genera:
      • Scatter plot de valores reales vs. predichos.
      • Histograma de los residuos.
    Retorna: (rmse, mae, r2, y_pred, y_test, residuals)
    """
    # Discretización de y_train en n_bins y obtención de los centros
    y_train_bins, bin_edges = pd.cut(y_train, bins=n_bins, retbins=True, labels=False)
    y_test_bins = pd.cut(y_test, bins=bin_edges, labels=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Seleccionar solo los bins presentes en el conjunto de entrenamiento
    unique_bins = np.unique(y_train_bins)
    new_bin_centers = bin_centers[unique_bins.astype(int)]

    # Entrenamiento del modelo NB
    nb = GaussianNB(var_smoothing=var_smoothing)
    nb.fit(X_train, y_train_bins)

    # Predicción: calcular la esperanza condicional con los centros de los bins presentes
    probs = nb.predict_proba(X_test)
    y_pred = np.dot(probs, new_bin_centers)

    residuals = y_test - y_pred

    # Gráfico de dispersión de valores reales vs. predichos
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Valores Reales (SalePrice)')
    plt.ylabel('Valores Predichos (NB Regresión)')
    plt.title('NB Regresión: Valores Reales vs. Predichos')
    plt.tight_layout()
    if save_plots:
        plt.savefig('NB_regression_scatter.png')
    plt.show()

    # Histograma de residuos
    plt.figure(figsize=(8,6))
    sns.histplot(residuals, bins=30, kde=True, color='skyblue')
    plt.xlabel('Errores (Residuales)')
    plt.title('NB Regresión: Histograma de Errores')
    plt.tight_layout()
    if save_plots:
        plt.savefig('NB_regression_residuals_histogram.png')
    plt.show()

    return (np.sqrt(mean_squared_error(y_test, y_pred)),
            mean_absolute_error(y_test, y_pred),
            r2_score(y_test, y_pred),
            y_pred, y_test, residuals)


def plot_nb_regression_residuals_boxplot(residuals, save_plots=True):
    """
    Genera un boxplot de los residuos del modelo NB de regresión.
    (Actividad 2: análisis de dispersión de residuos)
    """
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=residuals, color='lightgreen')
    plt.xlabel('Errores (Residuales)')
    plt.title('NB Regresión: Boxplot de Residuos')
    plt.tight_layout()
    if save_plots:
        plt.savefig('NB_regression_residuals_boxplot.png')
    plt.show()


# --------------------- Funciones para Comparación de Modelos de Regresión ---------------------
def regression_comparison_models(X_train, X_test, y_train, y_test, nb_params={'n_bins': 15, 'var_smoothing': 1e-9}):
    """
    Entrena y obtiene predicciones de:
      - Naïve Bayes (usando nb_regression)
      - Regresión Lineal
      - Árbol de Decisión (max_depth=5)
      - Random Forest (n_estimators=100, max_depth=7)
    Retorna un diccionario con métricas y predicciones para cada modelo.
    """
    metrics = {}

    # NB Regresión
    rmse_nb, mae_nb, r2_nb, y_pred_nb, y_true_nb, _ = nb_regression(X_train, X_test, y_train, y_test,
                                                                    n_bins=nb_params['n_bins'],
                                                                    var_smoothing=nb_params['var_smoothing'],
                                                                    save_plots=False)
    metrics['Naïve Bayes'] = {'rmse': rmse_nb, 'mae': mae_nb, 'r2': r2_nb, 'y_pred': y_pred_nb}

    # Regresión Lineal
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    mae_lr = np.mean(np.abs(y_test - y_pred_lr))
    r2_lr = r2_score(y_test, y_pred_lr)
    metrics['Regresión Lineal'] = {'rmse': rmse_lr, 'mae': mae_lr, 'r2': r2_lr, 'y_pred': y_pred_lr}

    # Árbol de Decisión
    dt = DecisionTreeRegressor(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
    mae_dt = np.mean(np.abs(y_test - y_pred_dt))
    r2_dt = r2_score(y_test, y_pred_dt)
    metrics['Árbol de Decisión'] = {'rmse': rmse_dt, 'mae': mae_dt, 'r2': r2_dt, 'y_pred': y_pred_dt}

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    mae_rf = np.mean(np.abs(y_test - y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)
    metrics['Random Forest'] = {'rmse': rmse_rf, 'mae': mae_rf, 'r2': r2_rf, 'y_pred': y_pred_rf}

    return metrics


def plot_regression_metrics_comparison(metrics, save_plots=True):
    """
    Genera un diagrama de barras comparativo de RMSE, MAE y R² para los modelos de regresión.
    (Actividad 3)
    """
    modelos = list(metrics.keys())
    rmse_vals = [metrics[m]['rmse'] for m in modelos]
    mae_vals = [metrics[m]['mae'] for m in modelos]
    r2_vals = [metrics[m]['r2'] for m in modelos]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.barplot(x=modelos, y=rmse_vals, ax=axes[0], palette='viridis')
    axes[0].set_title('RMSE')
    axes[0].set_ylabel('RMSE')

    sns.barplot(x=modelos, y=mae_vals, ax=axes[1], palette='magma')
    axes[1].set_title('MAE')
    axes[1].set_ylabel('MAE')

    sns.barplot(x=modelos, y=r2_vals, ax=axes[2], palette='coolwarm')
    axes[2].set_title('R²')
    axes[2].set_ylabel('R²')
    axes[2].set_ylim(0, 1)

    fig.suptitle('Comparación de Métricas de Regresión', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_plots:
        plt.savefig('regression_metrics_comparison.png')
    plt.show()


def plot_regression_real_vs_predicted_comparison(X_test, y_test, metrics, save_plots=True):
    """
    Genera un gráfico con subplots que muestra los valores reales vs. predichos para cada modelo.
    (Actividad 5)
    """
    modelos = list(metrics.keys())
    n_models = len(modelos)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    for ax, m in zip(axes, modelos):
        y_pred = metrics[m]['y_pred']
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_title(f"{m}\n(R²={metrics[m]['r2']:.2f})")
        ax.set_xlabel('Valores Reales')
        ax.set_ylabel('Valores Predichos')
    plt.tight_layout()
    if save_plots:
        plt.savefig('regression_real_vs_predicted_comparison.png')
    plt.show()


# --------------------- Funciones para NB Clasificación ---------------------
def nb_classification(X_train, X_test, y_train, y_test, var_smoothing=1e-9, save_plots=True):
    """
    Entrena un clasificador Naïve Bayes para clasificación con la variable categórica PriceCategory.
    Genera:
      • Matriz de confusión.
    Retorna: (accuracy, cm)
    """
    nb_clf = GaussianNB(var_smoothing=var_smoothing)
    nb_clf.fit(X_train, y_train)
    y_pred = nb_clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión - NB Clasificación')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.tight_layout()
    if save_plots:
        plt.savefig('NB_classification_confusion_matrix.png')
    plt.show()

    return acc, cm


def plot_classification_accuracy_comparison(acc_dict, save_plots=True):
    """
    Genera un gráfico de barras comparativo de exactitud para modelos de clasificación.
    acc_dict: diccionario con {modelo: exactitud}.
    (Actividades 4 y 10)
    """
    modelos = list(acc_dict.keys())
    accuracies = [acc_dict[m] for m in modelos]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=modelos, y=accuracies, palette='pastel')
    plt.title('Comparación de Exactitud - Clasificación')
    plt.ylabel('Exactitud')
    plt.ylim(0, 1)
    plt.tight_layout()
    if save_plots:
        plt.savefig('classification_accuracy_comparison.png')
    plt.show()


# --------------------- Funciones para Validación Cruzada y Sobreajuste ---------------------
def plot_cv_scores_distribution(X_train, y_train, n_bins=15, var_smoothing=1e-9, cv=5, save_plots=True):
    """
    Realiza validación cruzada (sobre la clasificación de bins) para NB en regresión y genera:
      • Boxplot e Histograma de las puntuaciones.
    (Actividad 8)
    """
    # Discretización de y_train
    y_train_bins, _ = pd.cut(y_train, bins=n_bins, retbins=True, labels=False)
    nb = GaussianNB(var_smoothing=var_smoothing)
    # Usamos KFold para obtener scores
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    for train_index, val_index in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_tr, y_val = y_train_bins.iloc[train_index], y_train_bins.iloc[val_index]
        nb.fit(X_tr, y_tr)
        score = nb.score(X_val, y_val)
        scores.append(score)
    scores = np.array(scores)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=scores, color='lightblue')
    plt.title('CV Scores - Boxplot')
    plt.ylabel('Exactitud')
    plt.subplot(1, 2, 2)
    sns.histplot(scores, bins=10, kde=True, color='salmon')
    plt.title('CV Scores - Histograma')
    plt.xlabel('Exactitud')
    plt.tight_layout()
    if save_plots:
        plt.savefig('cv_scores_distribution.png')
    plt.show()


def plot_train_test_R2_comparison(X_train, X_test, y_train, y_test, model, save_plots=True):
    """
    Calcula el R² para entrenamiento y prueba con el modelo dado y genera un gráfico de barras.
    (Actividad 7)
    """
    r2_train = model.score(X_train, y_train)
    r2_test = model.score(X_test, y_test)
    labels = ['Entrenamiento', 'Prueba']
    r2_values = [r2_train, r2_test]
    plt.figure(figsize=(6, 4))
    sns.barplot(x=labels, y=r2_values, palette='magma')
    plt.title('Comparación de R²: Entrenamiento vs. Prueba')
    plt.ylim(0, 1)
    plt.ylabel('R²')
    plt.tight_layout()
    if save_plots:
        plt.savefig('train_test_R2_comparison.png')
    plt.show()


# --------------------- Funciones para Afinación de Hiperparámetros ---------------------
def nb_regression_hyperparameter_tuning(X, y, bins_list=[5, 10, 15, 20], vs_list=[1e-9, 1e-8, 1e-7], cv=3,
                                        save_plots=True):
    """
    Realiza un barrido (grid search) sobre dos hiperparámetros: número de bins y var_smoothing,
    evaluando el RMSE (usando una validación simple con KFold) para NB en regresión.
    Genera un heatmap del RMSE promedio para cada combinación.
    (Actividad 9 para regresión)
    """
    from sklearn.model_selection import KFold
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import mean_squared_error
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    results = np.zeros((len(bins_list), len(vs_list)))

    for i, bins in enumerate(bins_list):
        # Discretización global: se calculan los bin_edges y los centros con todo el conjunto
        _, bin_edges = pd.cut(y, bins=bins, retbins=True, labels=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        for j, vs in enumerate(vs_list):
            rmses = []
            for train_index, val_index in kf.split(X):
                X_tr, X_val = X.iloc[train_index], X.iloc[val_index]
                y_tr, y_val = y.iloc[train_index], y.iloc[val_index]
                # Discretizamos usando los bin_edges globales
                y_tr_bins = pd.cut(y_tr, bins=bin_edges, labels=False)
                # Eliminar posibles NaN (si hay valores fuera del rango)
                y_tr_bins = y_tr_bins.dropna()
                # Obtener los índices únicos (clases) presentes en el conjunto de entrenamiento
                unique_bins = np.unique(y_tr_bins)
                # Seleccionar los centros correspondientes a las clases presentes
                new_bin_centers = bin_centers[unique_bins.astype(int)]

                # Entrenar NB con los datos correspondientes (se usa solo X_tr para los índices válidos)
                nb = GaussianNB(var_smoothing=vs)
                nb.fit(X_tr.loc[y_tr_bins.index], y_tr_bins)

                # Predicción en el conjunto de validación
                probs = nb.predict_proba(X_val)
                # Ahora, el número de columnas de probs debe coincidir con len(new_bin_centers)
                y_pred = np.dot(probs, new_bin_centers)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                rmses.append(rmse)
            results[i, j] = np.mean(rmses)

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(results, annot=True, fmt=".2f", xticklabels=vs_list, yticklabels=bins_list, cmap='YlGnBu')
    ax.set_xlabel('var_smoothing')
    ax.set_ylabel('Número de bins')
    plt.title('Afinación NB Regresión: RMSE promedio')
    plt.tight_layout()
    if save_plots:
        plt.savefig('NB_regression_hyperparameter_tuning.png')
    plt.show()


def nb_classification_hyperparameter_tuning(X_train, X_test, y_train, y_test, vs_list=[1e-9, 1e-8, 1e-7],
                                            save_plots=True):
    """
    Varía el parámetro var_smoothing para NB en clasificación y evalúa la exactitud en el conjunto de prueba.
    Genera un gráfico de línea de exactitud vs. var_smoothing.
    (Actividad 9 para clasificación)
    """
    accuracies = []
    for vs in vs_list:
        nb = GaussianNB(var_smoothing=vs)
        nb.fit(X_train, y_train)
        y_pred = nb.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

    plt.figure(figsize=(8, 6))
    plt.plot(vs_list, accuracies, marker='o', linestyle='--')
    plt.xscale('log')
    plt.xlabel('var_smoothing')
    plt.ylabel('Exactitud')
    plt.title('Afinación NB Clasificación: Exactitud vs. var_smoothing')
    plt.tight_layout()
    if save_plots:
        plt.savefig('NB_classification_hyperparameter_tuning.png')
    plt.show()


# --------------------- Main ---------------------
def main():
    # Cargar datos
    df = pd.read_csv('train.csv')
    # Para regresión se usan las siguientes características
    features_reg = ['OverallQual', 'GrLivArea', 'TotalBsmtSF']
    X_reg = df[features_reg]
    y_reg = df['SalePrice']
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    # Para clasificación se crea la variable PriceCategory
    conditions = [
        (df['SalePrice'] < 150000),
        (df['SalePrice'] >= 150000) & (df['SalePrice'] <= 300000),
        (df['SalePrice'] > 300000)
    ]
    choices = ['Económicas', 'Intermedias', 'Caras']
    df['PriceCategory'] = np.select(conditions, choices, default='Desconocido')
    X_clf = df[features_reg]
    y_clf = df['PriceCategory']
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

    # ---------------- Actividad 1 y 2: NB Regresión ----------------
    print("Actividad 1 y 2: Modelo de Regresión con Naïve Bayes")
    nb_rmse, nb_mae, nb_r2, y_pred_nb, y_test_actual, nb_residuals = nb_regression(X_train_reg, X_test_reg, y_train_reg,
                                                                                   y_test_reg)
    plot_nb_regression_residuals_boxplot(nb_residuals)

    # ---------------- Actividad 3 y 5: Comparación de Modelos de Regresión ----------------
    print("Actividad 3 y 5: Comparación con Modelos de Regresión")
    metrics = regression_comparison_models(X_train_reg, X_test_reg, y_train_reg, y_test_reg)
    plot_regression_metrics_comparison(metrics)
    plot_regression_real_vs_predicted_comparison(X_test_reg, y_test_reg, metrics)

    # ---------------- Actividad 4 y 10: NB Clasificación ----------------
    print("Actividad 4 y 10: Modelo de Clasificación con Naïve Bayes")
    nb_clf_acc, nb_cm = nb_classification(X_train_clf, X_test_clf, y_train_clf, y_test_clf)
    # Entrenar otros clasificadores para comparar exactitud
    dt_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_clf.fit(X_train_clf, y_train_clf)
    y_pred_dt = dt_clf.predict(X_test_clf)
    acc_dt = accuracy_score(y_test_clf, y_pred_dt)

    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_clf.fit(X_train_clf, y_train_clf)
    y_pred_rf = rf_clf.predict(X_test_clf)
    acc_rf = accuracy_score(y_test_clf, y_pred_rf)

    acc_dict = {'Naïve Bayes': nb_clf_acc,
                'Árbol de Decisión': acc_dt,
                'Random Forest': acc_rf}
    plot_classification_accuracy_comparison(acc_dict)

    # ---------------- Actividad 7: Análisis de Sobreajuste ----------------
    print("Actividad 7: Análisis de Sobreajuste")
    dt_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
    dt_reg.fit(X_train_reg, y_train_reg)
    plot_train_test_R2_comparison(X_train_reg, X_test_reg, y_train_reg, y_test_reg, dt_reg)

    # ---------------- Actividad 8: Validación Cruzada para NB Regresión ----------------
    print("Actividad 8: Validación Cruzada para NB Regresión")
    plot_cv_scores_distribution(X_train_reg, y_train_reg)

    # ---------------- Actividad 9: Afinación de Hiperparámetros ----------------
    print("Actividad 9: Afinación de Hiperparámetros para NB")
    nb_regression_hyperparameter_tuning(X_train_reg, y_train_reg)
    nb_classification_hyperparameter_tuning(X_train_clf, X_test_clf, y_train_clf, y_test_clf)


if __name__ == '__main__':
    main()
