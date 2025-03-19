import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from src.models.neural_network import NeuralNetworkROM
from src.models.gaussian_process import GaussianProcessROM
from src.models.rbf import RBFROM
from src.models.polynomial_response_surface import PolynomialResponseSurface
from src.models.svr import SVRROM

def search_best_hyperparameters(model_name, X_train, y_train, search_type='grid', n_iter=10):
    """
    Realiza una búsqueda de hiperparámetros para el modelo especificado.
    
    model_name: str: El nombre del modelo ('neural_network', 'gaussian_process', 'rbf')
    X_train: np.array: Los datos de entrada de entrenamiento.
    y_train: np.array: Las etiquetas de salida de entrenamiento.
    search_type: str: El tipo de búsqueda ('grid' para GridSearch, 'random' para RandomizedSearch).
    n_iter: int: Número de iteraciones si se usa RandomizedSearch.
    
    return: dict: Los mejores hiperparámetros encontrados.
    """
    # Print shapes of X_train and y_train
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    
    if model_name.lower() == "neural_network":
        # Definir el modelo
        model = NeuralNetworkROM(input_dim=X_train.shape[1], output_dim=y_train.shape[1])
        
        # Definir el espacio de hiperparámetros
        param_grid = {
            'hidden_layers': [1, 2, 4],
            'neurons_per_layer': [5, 10, 20],
            'learning_rate': [1e-3, 1e-4],
            'activation_function': ['ReLU', 'Tanh', 'Sigmoid'],
            'optimizer': ['Adam', 'SGD', 'RMSprop'],
            'num_epochs': [1000]
        }
        
        if search_type == 'grid':
            search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
        else:
            search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=n_iter, cv=5, scoring='neg_mean_squared_error')
        
    elif model_name.lower() == "gaussian_process":
        # Definir el modelo
        model = GaussianProcessROM()
        
        # Definir el espacio de hiperparámetros
        param_grid = {
            'kernel': ['RBF', 'Matern'],
            'noise': [1e-3, 1e-2, 1e-1],
        }
        
        if search_type == 'grid':
            search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
        else:
            search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=n_iter, cv=5, scoring='neg_mean_squared_error')
    
    elif model_name.lower() == "rbf":
        # Definir el modelo
        model = RBFROM()
        
        # Definir el espacio de hiperparámetros
        param_grid = {
            'gamma': [0.1, 1.0, 10.0]
        }
        
        if search_type == 'grid':
            search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
        else:
            search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=n_iter, cv=5, scoring='neg_mean_squared_error')

    elif model_name.lower() == "response_surface":
        # Definir el modelo
        model = PolynomialResponseSurface()
        
        # Definir el espacio de hiperparámetros
        param_grid = {
            'degree': [2, 3, 4, 5, 6, 7]
        }
        
        if search_type == 'grid':
            search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
        else:
            search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=n_iter, cv=5, scoring='neg_mean_squared_error')

    elif model_name.lower() == "svr":
        # Definir el modelo
        model = SVRROM()
        
        # Definir el espacio de hiperparámetros
        param_grid = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [0.1, 5, 50],
            'epsilon': [0.1, 0.2, 0]
        }
        
        if search_type == 'grid':
            search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
        else:
            search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=n_iter, cv=5, scoring='neg_mean_squared_error')


    # Realizar la búsqueda de hiperparámetros
    search.fit(X_train, y_train)
    
    # Imprimir los mejores parámetros y el mejor score
    print("Mejores hiperparámetros encontrados: ", search.best_params_)
    print("Mejor score encontrado: ", search.best_score_)
    
    # Devolver los mejores hiperparámetros
    return search.best_params_
