from sklearn.svm import SVR
from idkROM.idkROM import idkROM
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

class SVRROM(idkROM.Modelo):
    """
    Support Vector Regression (SVR) model compatible with idkROM framework.
    """
    def __init__(self, rom_config, random_state):
        super().__init__(rom_config, random_state)

        self.kernel = rom_config['hyperparams']['kernel_svr']
        self.degree = rom_config['hyperparams']['degree_svr']
        self.gamma = rom_config['hyperparams']['gamma_svr']
        self.tolerance = rom_config['hyperparams']['tolerance']
        self.C = rom_config['hyperparams']['C']
        self.epsilon = rom_config['hyperparams']['epsilon']

        self.random_state = random_state
        self.model_name = rom_config['model_name']
        self.rom_config = rom_config

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.train_losses = []
        self.val_losses = []

        self.base_model = SVR(kernel=self.kernel, degree=self.degree, gamma=self.gamma,
                              tol=self.tolerance, C=self.C, epsilon=self.epsilon)

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame = None, y_val: pd.DataFrame = None, validation_mode='cross'):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.models = []
        self.train_losses = []
        self.val_losses = []

        perform_cv = validation_mode == 'cross' or X_val is None or y_val is None

        if perform_cv:
            print("Iniciando Cross-Validation para cada salida...")
            for i in range(y_train.shape[1]):
                print(f"\n--- Cross-validation para salida {i+1} ---")
                y_column = y_train.iloc[:, i].values.ravel()
                kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
                fold_losses = []

                for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                    X_fold_train = X_train.iloc[train_idx]
                    y_fold_train = y_column[train_idx]
                    X_fold_val = X_train.iloc[val_idx]
                    y_fold_val = y_column[val_idx]

                    model = SVR(kernel=self.kernel, degree=self.degree, gamma=self.gamma,
                                tol=self.tolerance, C=self.C, epsilon=self.epsilon)
                    model.fit(X_fold_train, y_fold_train)
                    y_val_pred = model.predict(X_fold_val)
                    fold_loss = mean_squared_error(y_fold_val, y_val_pred)
                    fold_losses.append(fold_loss)
                    print(f"  Fold {fold+1}/5 MSE: {fold_loss:.6f}")

                avg_loss = np.mean(fold_losses)
                self.val_losses.append(avg_loss)
                print(f"Promedio CV MSE para salida {i+1}: {avg_loss:.6f}")

                # Entrenar modelo final en todo el dataset
                final_model = SVR(kernel=self.kernel, degree=self.degree, gamma=self.gamma,
                                  tol=self.tolerance, C=self.C, epsilon=self.epsilon)
                final_model.fit(X_train, y_column)
                self.models.append(final_model)
        else:
            print("Entrenamiento con validación explícita...")
            for i in range(y_train.shape[1]):
                print(f"\nEntrenando modelo para la salida {i+1}")
                y_train_column = y_train.iloc[:, i].values.ravel()

                model = SVR(kernel=self.kernel, degree=self.degree, gamma=self.gamma,
                            tol=self.tolerance, C=self.C, epsilon=self.epsilon)
                model.fit(X_train, y_train_column)
                self.models.append(model)

                y_train_pred = model.predict(X_train)
                mse_train = mean_squared_error(y_train_column, y_train_pred)
                self.train_losses.append(mse_train)
                print(f"Training MSE (salida {i+1}): {mse_train}")

                y_val_pred = model.predict(X_val)
                mse_val = mean_squared_error(y_val.iloc[:, i], y_val_pred)
                self.val_losses.append(mse_val)
                print(f"Validation MSE (salida {i+1}): {mse_val}")

        # Guardar modelos
        output_folder = os.path.join(os.getcwd(), 'results', self.model_name)
        os.makedirs(output_folder, exist_ok=True)
        model_path = os.path.join(output_folder, 'svr_model.pkl')
        with open(model_path, 'wb') as f:
            joblib.dump(self.models, f)
        print(f"\nModelos guardados en: {model_path}")

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, 'models') or not self.models:
            raise ValueError("Model is not trained yet!")
        predictions = np.zeros((X_test.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X_test)
        return predictions

    def predict_single_output(self, X_test: pd.DataFrame, output_index: int) -> np.ndarray:
        if not hasattr(self, 'models') or not self.models:
            raise ValueError("Model is not trained yet!")
        if output_index < 0 or output_index >= len(self.models):
            raise ValueError(f"Output index {output_index} is out of range.")
        return self.models[output_index].predict(X_test)

    def idk_run(self, X_params_dict):
        if hasattr(self, "X_train") and hasattr(self.X_train, "columns"):
            if len(X_params_dict) != len(self.X_train.columns):
                raise ValueError(f"Número de variables de entrada incorrecto. Se esperaban {len(self.X_train.columns)}, se recibieron {len(X_params_dict)}.")
        else:
            print("Advertencia: No se pudo verificar el número de variables de entrada.")

        X = np.array([list(X_params_dict.values())])
        y_pred = self.predict(pd.DataFrame(X, columns=self.X_train.columns))

        if y_pred.ndim > 1:
            y_pred_flat = y_pred.flatten() if y_pred.shape[0] == 1 else y_pred[0]
        else:
            y_pred_flat = y_pred

        if hasattr(self, "y_train") and hasattr(self.y_train, "columns"):
            target_keys = list(self.y_train.columns)
            if y_pred_flat.size != len(target_keys):
                raise ValueError("El número de resultados predichos no coincide con el número de columnas en y_train.")
        else:
            print("Advertencia: No se pudieron obtener los nombres de salida. Usando etiquetas genéricas.")
            target_keys = [f"result{i+1}" for i in range(y_pred_flat.size)]

        results = {key: value for key, value in zip(target_keys, y_pred_flat)}
        return results
