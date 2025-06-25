# -*- coding: utf-8 -*-

import json
import joblib
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
#from tqdm_joblib import tqdm_joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import TransformedTargetRegressor
#from skorch import NeuralNetRegressor
from sklearn.model_selection import ParameterGrid


# ===========================
# Clase del modelo feedforward
# ===========================
class SkorchFeedforwardNN(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, n_neurons, activation='relu', dropout=0.0):
        super().__init__()
        layers = []
        current_dim = input_dim
        act = {'relu': nn.ReLU(),
               'tanh': nn.Tanh(),
               'leakyrelu': nn.LeakyReLU()}[activation.lower()]

        target_neurons = n_neurons if n_layers > 0 else output_dim
        layers.append(nn.Linear(current_dim, target_neurons))
        layers.append(act)
        layers.append(nn.Dropout(dropout))
        current_dim = target_neurons

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(current_dim, n_neurons))
            layers.append(act)
            layers.append(nn.Dropout(dropout))
            current_dim = n_neurons

        if n_layers > 0:
            layers.append(nn.Linear(current_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)


# ===========================
# Filtro de Outliers (IQR)
# ===========================
def remove_outliers(X, y=None, factor=1.5):
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    mask = np.all((X >= Q1 - factor * IQR) & (X <= Q3 + factor * IQR), axis=1)
    return (X[mask], y[mask]) if y is not None else X[mask]


# ===========================
# Main
# ===========================
def main():
    # Cargar datos
    X = pd.read_csv(
        r'C:\Users\aalvarezsanz\OneDrive - DanobatGroup\Documentos\idk_framework\DOE_datos\post\inputs.csv'
    )
    y = pd.read_csv(
        r'C:\Users\aalvarezsanz\OneDrive - DanobatGroup\Documentos\idk_framework\DOE_datos\post\outputs.csv'
    )

   # Quitar ciertas columnas que no quieres usar
    columnas_a_quitar = ['CO2', 'dofS', 'TCP_X_pos_list', 'TCP_Y_pos_list', 'TCP_Z_pos_list','TCP_A_pos_list', 'TCP_B_pos_list', 'TCP_C_pos_list']
    y = y.drop(columns=columnas_a_quitar, errors='ignore')


    # Filtrar outliers
    X_filtered, y_filtered = remove_outliers(X.values, y.values)

    # Separar train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_filtered, test_size=0.2, random_state=42
    )

    # Red
    net = NeuralNetRegressor(
        module=SkorchFeedforwardNN,
        module__input_dim=X_train.shape[1],
        module__output_dim=y_train.shape[1],
        max_epochs=1000,
        lr=1e-3,
        optimizer=optim.Adam,
        device='cpu',
        verbose=0  # <- Silencia los logs de cada epoch
    )


    # Pipeline para preprocesar X
    pipe_X = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler()),  # Escalado de X
    ])

    # Usar TransformedTargetRegressor para escalar y
    regressor = TransformedTargetRegressor(
        regressor=net,
        transformer=MinMaxScaler()  # Escalado también de y
    )

    pipe = Pipeline([
        ('preprocessing', pipe_X),
        ('model', regressor),
    ])

    # Param grid (atención a los nuevos prefijos)
    param_grid = {
        'model__regressor__module__n_layers': [2, 3, 4, 5],
        'model__regressor__module__n_neurons': [16, 32, 64, 128],
        'model__regressor__module__activation': ['relu'],
        'model__regressor__module__dropout': [0.1, 0.2, 0.3]
    }

    # GridSearch
    gs = GridSearchCV(
        pipe,
        param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )

    n_combinations = len(list(ParameterGrid(param_grid)))
    total_fits = n_combinations * 3  # cv=3 en tu GridSearchCV

    with tqdm_joblib(tqdm(desc="GridSearchCV", total=total_fits)) as _:
        gs.fit(X_train.astype('float32'), y_train.astype('float32'))

    print('Mejores params:', gs.best_params_)
    print('Mejor score CV:', gs.best_score_)

    # Evaluar en test
    best_model = gs.best_estimator_
    y_pred_test = best_model.predict(X_test.astype('float32'))
    mse_test = np.mean((y_pred_test - y_test) ** 2)
    print(f'MSE en test: {mse_test:.4f}')

    # ===========================
    # Guardar predicciones y mejor modelo
    # ===========================

    with open('best_params_and_score.json', 'w') as f:
        json.dump(
            {'best_params': gs.best_params_, 'best_score': gs.best_score_},
            f, indent=4
        )
    print(f"Mejores params y score guardados en best_params_and_score.json")

    # ===========================
    # Guardar mejor modelo
    # ===========================
    best_model = gs.best_estimator_
    joblib.dump(best_model, 'best_model.pkl')
    print("Mejor modelo guardado en best_model.pkl")

    # ===========================
    # (Opcional) Guardar predicciones del test en CSV
    # ===========================
    y_pred_test = best_model.predict(X_test.astype('float32'))
    df_results = pd.DataFrame({
        'y_pred': y_pred_test.flatten(),
        'y_true': y_test.flatten()
    })
    df_results.to_csv('test_predictions.csv', index=False)
    print("Predicciones en test_predictions.csv")


if __name__ == "__main__":
    #main()

    # Cargar datos


    import pandas as pd
    import ast  # Para convertir strings tipo "[1, 2, 3]" en listas

    # Cargar CSV original
    y = pd.read_csv(
        r'C:\Users\aalvarezsanz\OneDrive - DanobatGroup\Documentos\idk_framework\DOE_datos\post\outputs.csv'
    )

    # Quitar columnas innecesarias
    columnas_a_quitar = ['CO2', 'dofS']
    y = y.drop(columns=columnas_a_quitar, errors='ignore')

    # Crear una copia del DataFrame que se irá actualizando
    y_expandido = pd.DataFrame()

    for col in y.columns:
        # Intentar convertir el primer valor a lista para saber si es una lista representada como string
        try:
            val = ast.literal_eval(str(y[col].iloc[0]))
            if isinstance(val, list):
                # Expandir columna en múltiples columnas
                col_expandida = y[col].apply(ast.literal_eval).apply(pd.Series)
                col_expandida.columns = [f"{col}_{i}" for i in col_expandida.columns]
                y_expandido = pd.concat([y_expandido, col_expandida], axis=1)
            else:
                y_expandido[col] = y[col]
        except (ValueError, SyntaxError):
            # Si no es una lista o da error al convertir, dejar la columna como está
            y_expandido[col] = y[col]

    # Guardar CSV procesado
    y_expandido.to_csv(
        r'C:\Users\aalvarezsanz\OneDrive - DanobatGroup\Documentos\idk_framework\DOE_datos\post\outputs_filtrados_2.csv',
        index=False
    )


