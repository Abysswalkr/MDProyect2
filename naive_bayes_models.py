import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report


class GaussianNBRegressor:
    """
    Implementación de Naive Bayes Gaussiano para regresión.
    Esta clase adapta el clasificador GaussianNB para tareas de regresión
    discretizando la variable objetivo y luego prediciendo el valor esperado.
    """
    def __init__(self, n_bins=10, var_smoothing=1e-9):
        self.n_bins = n_bins
        self.var_smoothing = var_smoothing
        self.classifier = GaussianNB(var_smoothing=var_smoothing)
        
    def fit(self, X, y):
        # Crear bins equidistantes
        bin_edges = np.linspace(y.min(), y.max(), self.n_bins + 1)
        
        # Asignar cada valor a un bin
        y_binned = np.digitize(y, bin_edges[1:])
        
        # Entrenar el clasificador
        self.classifier.fit(X, y_binned)
        
        # Calcular los centros de bins para cada clase que el clasificador conoce
        self.bin_means = {}
        for bin_idx in np.unique(y_binned):
            self.bin_means[bin_idx] = y[y_binned == bin_idx].mean()
        
        return self
    
    def predict(self, X):
        # Predecir probabilidades para cada clase
        proba = self.classifier.predict_proba(X)
        
        # Inicializar array de predicciones
        y_pred = np.zeros(X.shape[0])
        
        # Para cada muestra
        for i in range(X.shape[0]):
            # Para cada clase
            for j, class_idx in enumerate(self.classifier.classes_):
                # Sumar el valor medio del bin ponderado por su probabilidad
                y_pred[i] += proba[i, j] * self.bin_means.get(class_idx, 0)
        
        return y_pred
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)


def naive_bayes_regression(X_train, X_test, y_train, y_test, n_bins=10, var_smoothing=1e-9):
    """
    Entrena un modelo Naive Bayes para regresión y evalúa su rendimiento.
    
    Parámetros:
    -----------
    X_train, X_test, y_train, y_test : arrays
        Datos de entrenamiento y prueba
    n_bins : int, default=10
        Número de bins para discretizar la variable objetivo
    var_smoothing : float, default=1e-9
        Porción de la varianza más grande de todas las características 
        que se añade a las varianzas para la estabilidad del cálculo
        
    Retorna:
    --------
    model : GaussianNBRegressor
        Modelo entrenado
    rmse, mae, r2 : float
        Métricas de rendimiento
    """
    # Crear y entrenar el modelo
    model = GaussianNBRegressor(n_bins=n_bins, var_smoothing=var_smoothing)
    model.fit(X_train, y_train)
    
    # Hacer predicciones
    y_pred = model.predict(X_test)
    
    # Calcular métricas de rendimiento
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Mostrar resultados
    print(f"Naive Bayes Regresión (n_bins={n_bins}, var_smoothing={var_smoothing}):")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R²: {r2:.3f}")
    
    # Graficar valores predichos vs reales
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='purple')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Naive Bayes Regresión: Valores Reales vs Predichos')
    plt.xlabel('Valor Real de SalePrice')
    plt.ylabel('Valor Predicho de SalePrice')
    plt.tight_layout()
    plt.show()
    
    return model, rmse, mae, r2


def naive_bayes_classification(X_train, X_test, y_train, y_test, var_smoothing=1e-9):
    """
    Entrena un modelo Naive Bayes para clasificación y evalúa su rendimiento.
    
    Parámetros:
    -----------
    X_train, X_test, y_train, y_test : arrays
        Datos de entrenamiento y prueba
    var_smoothing : float, default=1e-9
        Porción de la varianza más grande de todas las características 
        que se añade a las varianzas para la estabilidad del cálculo
        
    Retorna:
    --------
    model : GaussianNB
        Modelo entrenado
    cm : array
        Matriz de confusión
    cr : string
        Reporte de clasificación
    """
    # Eliminar filas con valores NaN
    mask = ~(X_train.isna().any(axis=1) | y_train.isna())
    X_train_clean = X_train[mask]
    y_train_clean = y_train[mask]
    
    mask_test = ~(X_test.isna().any(axis=1) | y_test.isna())
    X_test_clean = X_test[mask_test]
    y_test_clean = y_test[mask_test]
    
    # Crear y entrenar el modelo
    model = GaussianNB(var_smoothing=var_smoothing)
    model.fit(X_train_clean, y_train_clean)
    
    # Hacer predicciones
    y_pred = model.predict(X_test_clean)
    
    # Calcular métricas
    cm = confusion_matrix(y_test_clean, y_pred)
    cr = classification_report(y_test_clean, y_pred)
    
    # Mostrar resultados
    print(f"Naive Bayes Clasificación (var_smoothing={var_smoothing}):")
    print("Matriz de Confusión:")
    print(cm)
    print("Reporte de Clasificación:")
    print(cr)
    
    # Graficar matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión - Naive Bayes Clasificación')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.tight_layout()
    plt.show()
    
    return model, cm, cr


def tune_naive_bayes_regression(X_train, X_test, y_train, y_test):
    """
    Ajusta hiperparámetros para el modelo Naive Bayes de regresión.
    
    Parámetros:
    -----------
    X_train, X_test, y_train, y_test : arrays
        Datos de entrenamiento y prueba
        
    Retorna:
    --------
    best_model : GaussianNBRegressor
        Mejor modelo después del ajuste de hiperparámetros
    best_rmse, best_mae, best_r2 : float
        Mejores métricas conseguidas
    """
    # Definir grid de hiperparámetros
    n_bins_range = [5, 10, 15, 20, 25, 30]
    var_smoothing_range = [1e-11, 1e-9, 1e-7, 1e-5, 1e-3, 1e-1]
    
    # Inicializar mejores valores
    best_rmse = float('inf')
    best_model = None
    best_mae = None
    best_r2 = None
    best_params = None
    
    # Búsqueda en grid
    results = []
    for n_bins in n_bins_range:
        for var_smoothing in var_smoothing_range:
            model = GaussianNBRegressor(n_bins=n_bins, var_smoothing=var_smoothing)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results.append({
                'n_bins': n_bins,
                'var_smoothing': var_smoothing,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            })
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_mae = mae
                best_r2 = r2
                best_model = model
                best_params = {'n_bins': n_bins, 'var_smoothing': var_smoothing}
    
    # Mostrar resultados
    results_df = pd.DataFrame(results)
    print("Resultados del ajuste de hiperparámetros para Naive Bayes Regresión:")
    print(f"Mejores Parámetros: {best_params}")
    print(f"Mejor RMSE: {best_rmse:.2f}")
    print(f"Mejor MAE: {best_mae:.2f}")
    print(f"Mejor R²: {best_r2:.3f}")
    
    # Graficar heatmap de hiperparámetros
    if len(n_bins_range) > 1 and len(var_smoothing_range) > 1:
        pivot_rmse = results_df.pivot(index='n_bins', columns='var_smoothing', values='rmse')
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_rmse, annot=True, fmt='.2f', cmap='viridis_r')
        plt.title('RMSE por Hiperparámetros - Naive Bayes Regresión')
        plt.xlabel('var_smoothing')
        plt.ylabel('n_bins')
        plt.tight_layout()
        plt.show()
    
    return best_model, best_rmse, best_mae, best_r2


def tune_naive_bayes_classification(X_train, X_test, y_train, y_test):
    """
    Ajusta hiperparámetros para el modelo Naive Bayes de clasificación.
    
    Parámetros:
    -----------
    X_train, X_test, y_train, y_test : arrays
        Datos de entrenamiento y prueba
        
    Retorna:
    --------
    best_model : GaussianNB
        Mejor modelo después del ajuste de hiperparámetros
    best_cm : array
        Matriz de confusión para el mejor modelo
    best_cr : string
        Reporte de clasificación para el mejor modelo
    """
    # Eliminar filas con valores NaN
    mask = ~(X_train.isna().any(axis=1) | y_train.isna())
    X_train_clean = X_train[mask]
    y_train_clean = y_train[mask]
    
    mask_test = ~(X_test.isna().any(axis=1) | y_test.isna())
    X_test_clean = X_test[mask_test]
    y_test_clean = y_test[mask_test]
    
    # Definir grid de hiperparámetros
    param_grid = {
        'var_smoothing': np.logspace(-11, -1, 11)
    }
    
    # Crear el modelo
    model = GaussianNB()
    
    # Búsqueda en grid
    grid_search = GridSearchCV(
        model, param_grid, cv=5, 
        scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train_clean, y_train_clean)
    
    # Obtener mejor modelo
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Evaluar mejor modelo
    y_pred = best_model.predict(X_test_clean)
    best_cm = confusion_matrix(y_test_clean, y_pred)
    best_cr = classification_report(y_test_clean, y_pred)
    
    # Mostrar resultados
    print("Resultados del ajuste de hiperparámetros para Naive Bayes Clasificación:")
    print(f"Mejores Parámetros: {best_params}")
    print("Matriz de Confusión:")
    print(best_cm)
    print("Reporte de Clasificación:")
    print(best_cr)
    
    # Graficar matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión - Naive Bayes Clasificación Ajustado')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.tight_layout()
    plt.show()
    
    return best_model, best_cm, best_cr


def compare_models(models_info):
    """
    Compara el rendimiento de todos los modelos de regresión.
    
    Parámetros:
    -----------
    models_info : lista de diccionarios
        Lista con información de modelos (nombre, rmse, mae, r2)
    
    Retorna:
    --------
    comparison_df : DataFrame
        DataFrame con resultados de la comparación
    """
    comparison_df = pd.DataFrame(models_info)
    
    # Graficar comparación de RMSE
    plt.figure(figsize=(12, 8))
    sns.barplot(x='rmse', y='name', data=comparison_df, orient='h', palette='viridis')
    plt.title('Comparación de Modelos por RMSE')
    plt.xlabel('RMSE')
    plt.ylabel('Modelo')
    plt.tight_layout()
    plt.show()
    
    # Graficar comparación de R²
    plt.figure(figsize=(12, 8))
    sns.barplot(x='r2', y='name', data=comparison_df, orient='h', palette='magma')
    plt.title('Comparación de Modelos por R²')
    plt.xlabel('R²')
    plt.ylabel('Modelo')
    plt.tight_layout()
    plt.show()
    
    return comparison_df


def naive_bayes_cross_validation(X, y, n_bins=10, var_smoothing=1e-9, cv=5):
    """
    Realiza validación cruzada para el modelo Naive Bayes de regresión.
    
    Parámetros:
    -----------
    X, y : arrays
        Datos de características y objetivo
    n_bins : int, default=10
        Número de bins para discretizar la variable objetivo
    var_smoothing : float, default=1e-9
        Parámetro de suavizado de varianza
    cv : int, default=5
        Número de folds para validación cruzada
        
    Retorna:
    --------
    cv_scores : array
        Puntuaciones de validación cruzada (R²)
    """
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = GaussianNBRegressor(n_bins=n_bins, var_smoothing=var_smoothing)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        scores.append(score)
    
    # Mostrar resultados
    print(f"Validación Cruzada ({cv} folds) - Naive Bayes Regresión:")
    print(f"  R² medio: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    
    return np.array(scores)