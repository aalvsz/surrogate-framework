import os
import json
import joblib
import pandas as pd
import numpy as np
import ast
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from skorch import NeuralNetRegressor
from skorch.callbacks import LRScheduler
import torch.nn as nn
import torch.optim as optim
import torch


# 1. Expansión de columnas lista como TCP_X_pos_list
def expand_list_columns(df):
    new_columns = {}
    for col in df.columns:
        sample_val = df[col].iloc[0]
        if isinstance(sample_val, str):
            try:
                sample_val = ast.literal_eval(sample_val)
            except:
                continue
        if isinstance(sample_val, list):
            col_lists = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            max_len = col_lists.apply(len).max()
            for i in range(max_len):
                new_col_name = f"{col}_{i}"
                df[new_col_name] = col_lists.apply(lambda x: x[i] if i < len(x) else None)
            new_columns[col] = True
    df.drop(columns=list(new_columns.keys()), inplace=True)
    return df


# 2. Filtro outliers (IQR)
def remove_outliers(X, y, factor=1.5):
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)

    IQR = Q3 - Q1
    mask = np.all((X >= Q1 - factor * IQR) & (X <= Q3 + factor * IQR), axis=1)
    return X[mask], y[mask]


# 3. Red feedforward
class SkorchFeedforwardNN(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, n_neurons, activation='relu', dropout=0.0):
        super().__init__()
        layers = []
        current_dim = input_dim
        act = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'leakyrelu': nn.LeakyReLU()}[activation.lower()]

        layers.append(nn.Linear(current_dim, n_neurons))
        layers.append(act)
        layers.append(nn.Dropout(dropout))
        current_dim = n_neurons

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(current_dim, n_neurons))
            layers.append(act)
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(current_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)


# 4. MAIN
def main():
    # === Cargar y preparar ===
    X = pd.read_csv(r'C:\Users\aalvarezsanz\OneDrive - DanobatGroup\Documentos\idk_framework\DOE_datos\post\inputs.csv')
    y = pd.read_csv(r'C:\Users\aalvarezsanz\OneDrive - DanobatGroup\Documentos\idk_framework\DOE_datos\post\outputs_filtrados_2.csv')

    # y = expand_list_columns(y)
    # columnas_a_quitar = ['CO2', 'dofS']
    # y.drop(columns=columnas_a_quitar, errors='ignore', inplace=True)

    X_filtered, y_filtered = remove_outliers(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

    # === Preprocesamiento inputs ===
    preprocessing_X = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])

    # === Escalado outputs por variable ===
    output_columns = pd.DataFrame(y_train).columns
    y_scalers = {}
    y_train_scaled = pd.DataFrame(index=range(len(y_train)))
    y_test_scaled = pd.DataFrame(index=range(len(y_test)))

    scaled_train_cols = []
    scaled_test_cols = []

    for col in output_columns:
        scaler = RobustScaler() if col.startswith("TCP") else MinMaxScaler()
        scaler.fit(y_train[[col]])
        y_scalers[col] = scaler

        scaled_train_cols.append(pd.Series(scaler.transform(y_train[[col]]).flatten(), name=col))
        scaled_test_cols.append(pd.Series(scaler.transform(y_test[[col]]).flatten(), name=col))

    # Juntar todo de golpe
    y_train_scaled = pd.concat(scaled_train_cols, axis=1)
    y_test_scaled = pd.concat(scaled_test_cols, axis=1)


    # === Red base ===
    net = NeuralNetRegressor(
        module=SkorchFeedforwardNN,
        module__input_dim=X_train.shape[1],
        module__output_dim=y_train_scaled.shape[1],
        module__n_layers=2,              
        module__n_neurons=123,           
        module__activation='relu',       
        max_epochs=1000,
        iterator_train__shuffle=True,
        verbose=1
    )

    # === Pipeline final ===
    pipeline = Pipeline([
        ('preprocessing', preprocessing_X),
        ('model', net)
    ])

    # === Grid de parámetros ===
    step_sizes = [10, 50, 100]
    gammas = [0.1, 0.5, 0.9]
    patiences = [10, 50, 100]
    factors = [0.1, 0.5]

    # Generar combinaciones de StepLR
    step_lr_callbacks = [
        [LRScheduler(policy=torch.optim.lr_scheduler.StepLR, step_size=ss, gamma=gm)]
        for ss in step_sizes
        for gm in gammas
    ]

    # Generar combinaciones de ReduceLROnPlateau
    reduce_lr_callbacks = [
        [LRScheduler(policy=torch.optim.lr_scheduler.ReduceLROnPlateau, patience=pt, factor=fc)]
        for pt in patiences
        for fc in factors
    ]

    all_callbacks = step_lr_callbacks + reduce_lr_callbacks


    param_grid = {
        'model__lr': [1e-1, 1e-2, 1e-3, 1e-4],
        'model__optimizer': [optim.Adam, optim.SGD],
        'model__module__dropout': [0.1, 0.2, 0.3],
        'model__callbacks': all_callbacks
    }


    gs = GridSearchCV(
        pipeline,
        param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    # === Fit grid ===
    gs.fit(X_train.astype('float32'), y_train_scaled.to_numpy().astype('float32'))

    # === Mejor modelo ===
    best_model = gs.best_estimator_
    print("Mejores parámetros:", gs.best_params_)

    # === Evaluar en test ===
    y_pred_test_scaled = best_model.predict(X_test.astype('float32'))
    y_pred_test = pd.DataFrame(index=range(len(y_pred_test_scaled)))

    for i, col in enumerate(output_columns):
        scaler = y_scalers[col]
        y_pred_test[col] = scaler.inverse_transform(y_pred_test_scaled[:, [i]]).flatten()

    mse = np.mean((y_pred_test.values - y_test) ** 2)
    print(f"MSE test: {mse:.4f}")

    # === Guardado ===
    joblib.dump(best_model, 'best_model_skorch.pkl')
    joblib.dump(y_scalers, 'output_scalers.pkl')
    pd.DataFrame(y_pred_test).to_csv("y_pred_test.csv", index=False)
    pd.DataFrame(y_test, columns=output_columns).to_csv("y_true_test.csv", index=False)

    # Limpiar los callbacks antes de guardar
    def serialize_param(value):
        if isinstance(value, list):
            return [str(v) for v in value]
        elif callable(value):
            return value.__name__ if hasattr(value, '__name__') else str(value)
        else:
            return str(value)

    serializable_params = {
        k: serialize_param(v) for k, v in gs.best_params_.items()
    }

    # Guardar como JSON
    with open("best_params.json", "w") as f:
        json.dump(serializable_params, f, indent=2)

    print("Todo guardado correctamente.")


if __name__ == "__main__":
    #main()

    import joblib
    from collections import defaultdict

    # Cargar el mejor modelo guardado
    torch.serialization.add_safe_globals([SkorchFeedforwardNN])
# Registra todas las clases que usa tu red
    with torch.serialization.safe_globals(
        [SkorchFeedforwardNN, nn.Sequential, nn.Linear, nn.ReLU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU, nn.Dropout, nn.MSELoss, optim.Adam, defaultdict, dict,
    torch.optim.lr_scheduler.StepLR,
    torch.optim.lr_scheduler.ReduceLROnPlateau]
    ):
        best_model = joblib.load('best_model_skorch.pkl')
    # Acceder a los callbacks del modelo Skorch
    # Extraer los callbacks del modelo
    model_callbacks = best_model.named_steps['model'].callbacks_

    for cb in model_callbacks:
        # cb es una lista que contiene el LRScheduler
        if isinstance(cb, (list, tuple)) and len(cb) > 0:
            scheduler = cb[0]  # Primer elemento dentro
        else:
            scheduler = cb

        if isinstance(scheduler, LRScheduler):
            policy_name = scheduler.policy.__name__
            scheduler_kwargs = scheduler.kwargs
            print(f"Tipo scheduler: {policy_name}")
            print("Parámetros:", scheduler_kwargs)
            print("-" * 40)

    # Cargar los params que guardaste en best_params.json
    import json
    with open('best_params.json', 'r') as f:
        params = json.load(f)

    print(params['model__callbacks'])

    import ast
    callback_list = ast.literal_eval(params['model__callbacks'][0])  # primer callback elegido
    scheduler_info = callback_list[0]  # el LRScheduler

    print('Tipo scheduler:', scheduler_info.policy)  # StepLR o ReduceLROnPlateau
    print('Parámetros:', scheduler_info.kwargs)
