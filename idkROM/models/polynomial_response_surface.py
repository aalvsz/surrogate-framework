import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import joblib
from idkROM.idkROM import idkROM

class PolynomialResponseSurface(idkROM.Modelo):
    def __init__(self, rom_config, random_state):
        super().__init__(rom_config, random_state)

        # Extraer parámetros de configuración
        self.degree = rom_config['hyperparams']['degree_rs']
        self.interaction_only = rom_config['hyperparams']['interaction_only']
        self.include_bias = rom_config['hyperparams']['include_bias']
        self.order = rom_config['hyperparams']['order']
        self.fit_intercept = rom_config['hyperparams']['fit_intercept']
        self.positive = rom_config['hyperparams']['positive']

        self.random_state = random_state
        self.model_name = rom_config['model_name']
        self.poly = PolynomialFeatures(degree=self.degree, interaction_only=self.interaction_only,
                                       include_bias=self.include_bias, order=self.order)
        self.model = LinearRegression(fit_intercept=self.fit_intercept, positive=self.positive)

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.train_losses = []
        self.val_losses = []

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame = None, y_val: pd.DataFrame = None, validation_mode='cross'):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        perform_cv = True
        if validation_mode == 'single' and X_val is not None and y_val is not None:
            perform_cv = False
            print("Usando conjunto de validación explícito proporcionado.")
        else:
            print("Iniciando Cross-Validation.")

        if perform_cv:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_val_losses = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                print(f"--- Fold {fold+1}/5 ---")
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_val = y_train.iloc[val_idx]

                X_poly_fold_train = self.poly.fit_transform(X_fold_train)
                self.model.fit(X_poly_fold_train, y_fold_train)

                X_poly_fold_val = self.poly.transform(X_fold_val)
                val_preds = self.model.predict(X_poly_fold_val)
                val_loss = mean_squared_error(y_fold_val, val_preds)
                fold_val_losses.append(val_loss)

                print(f"  Fold {fold+1} Val Loss: {val_loss:.6f}")

            avg_val_loss = np.mean(fold_val_losses)
            self.val_losses.append(avg_val_loss)
            print(f"\nPromedio de la pérdida de validación en CV: {avg_val_loss:.6f}")
        else:
            X_poly_train = self.poly.fit_transform(X_train)
            self.model.fit(X_poly_train, y_train)

            y_train_pred = self.predict(X_train)
            mse_train = mean_squared_error(y_train, y_train_pred)
            self.train_losses.append(mse_train)
            print(f"Training MSE: {mse_train}")

            y_val_pred = self.predict(X_val)
            mse_val = mean_squared_error(y_val, y_val_pred)
            self.val_losses.append(mse_val)
            print(f"Validation MSE: {mse_val}")

        output_folder = os.path.join(os.getcwd(), 'results', self.model_name)
        os.makedirs(output_folder, exist_ok=True)
        model_path = os.path.join(output_folder, 'polynomial_model.pkl')
        with open(model_path, 'wb') as f:
            joblib.dump(self, f)
        print(f"Model saved at: {model_path}")

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained yet!")
        X_poly_test = self.poly.transform(X_test)
        return self.model.predict(X_poly_test)

    def idk_run(self, X_params_dict):
        if hasattr(self, "X_train") and hasattr(self.X_train, "columns"):
            if len(X_params_dict) != len(self.X_train.columns):
                raise ValueError("Número de variables de entrada incorrecto. Se esperaban {}, se recibieron {}.".format(len(self.X_train.columns), len(X_params_dict)))
        else:
            print("Advertencia: No se pudo verificar el número de variables de X_train.")

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
            print("Advertencia: No se pudieron obtener los nombres de salida, usando etiquetas genéricas.")
            target_keys = [f"result{i+1}" for i in range(y_pred_flat.size)]

        results = {key: value for key, value in zip(target_keys, y_pred_flat)}
        return results
