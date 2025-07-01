import os
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import joblib
from idkrom.model import idkROM

class RBFROM(idkROM.Modelo):
    """
    Radial Basis Function (RBF) model using Kernel Ridge Regression.
    """
    def __init__(self, rom_config, random_state):
        super().__init__(rom_config, random_state)

        self.alpha = rom_config['hyperparams']['alpha_rbf']
        self.kernel = rom_config['hyperparams']['kernel_rbf']
        self.gamma = rom_config['hyperparams']['gamma_rbf']
        if self.gamma == 'None':
            self.gamma = None
        self.degree = rom_config['hyperparams']['degree_rbf']  # Not used in RBF, but extracted anyway

        self.model_name = rom_config['model_name']
        self.random_state = random_state

        self.model = KernelRidge(alpha=self.alpha, kernel='rbf', gamma=self.gamma)
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
            kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            fold_val_losses = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                print(f"--- Fold {fold+1}/5 ---")
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_val = y_train.iloc[val_idx]

                model_cv = KernelRidge(alpha=self.alpha, kernel='rbf', gamma=self.gamma)
                model_cv.fit(X_fold_train, y_fold_train)

                y_fold_val_pred = model_cv.predict(X_fold_val)
                val_loss = mean_squared_error(y_fold_val, y_fold_val_pred)
                fold_val_losses.append(val_loss)

                print(f"  Fold {fold+1} Val Loss: {val_loss:.6f}")

            avg_val_loss = np.mean(fold_val_losses)
            self.val_losses.append(avg_val_loss)
            print(f"\nPromedio de la pérdida de validación en CV: {avg_val_loss:.6f}")

            # Retrain final model on full training data
            self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)

            y_train_pred = self.model.predict(X_train)
            mse_train = mean_squared_error(y_train, y_train_pred)
            self.train_losses.append(mse_train)
            print(f"Training MSE: {mse_train}")

            y_val_pred = self.model.predict(X_val)
            mse_val = mean_squared_error(y_val, y_val_pred)
            self.val_losses.append(mse_val)
            print(f"Validation MSE: {mse_val}")

        output_folder = os.path.join(os.getcwd(), 'results', self.model_name)
        os.makedirs(output_folder, exist_ok=True)
        model_path = os.path.join(output_folder, 'rbf_model.pkl')
        with open(model_path, 'wb') as f:
            joblib.dump(self, f)
        print(f"Model saved at: {model_path}")

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained yet!")
        return self.model.predict(X_test)

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
