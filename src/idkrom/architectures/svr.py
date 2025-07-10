import os
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import KFold
import joblib
from idkrom.model import idkROM

class SVRROM(idkROM.Modelo):

    def __init__(self, rom_config, random_state):
        super().__init__(rom_config, random_state)
        
        # Hiperparámetros desde el YAML
        self.C = rom_config['hyperparams'].get('C', 1.0)
        self.epsilon = rom_config['hyperparams'].get('epsilon', 0.1)
        self.kernel = rom_config['hyperparams'].get('kernel_svr', 'rbf')
        
        self.random_state = random_state
        self.model_name = rom_config['model_name']
        self.output_folder = rom_config['output_folder']

        # Se entrena un modelo por salida
        self.models = dict()

        # variables para reporte
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

    def train(self, X_train, y_train, X_val=None, y_val=None, validation_mode='cross'):
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
                X_train_fold = X_train.iloc[train_idx]
                y_train_fold = y_train.iloc[train_idx]
                X_val_fold = X_train.iloc[val_idx]
                y_val_fold = y_train.iloc[val_idx]

                # Entrenamos un modelo por variable de salida
                self.models = {}
                for col in y_train.columns:
                    svr = SVR(C=self.C, epsilon=self.epsilon, kernel=self.kernel)
                    svr.fit(X_train_fold, y_train_fold[col])
                    self.models[col] = svr

                # evaluación del fold
                val_preds = np.column_stack([
                    self.models[col].predict(X_val_fold) for col in y_train.columns
                ])
                val_loss = np.mean((val_preds - y_val_fold.values)**2)
                fold_val_losses.append(val_loss)
                print(f"  Fold {fold+1} Val Loss: {val_loss:.6f}")
            
            avg_val_loss = np.mean(fold_val_losses)
            print(f"\nPromedio de la pérdida de validación en CV: {avg_val_loss:.6f}")

        else:
            # Entrenamiento explícito
            print("Entrenando con datos de validación explícitos.")
            for col in y_train.columns:
                svr = SVR(C=self.C, epsilon=self.epsilon, kernel=self.kernel)
                svr.fit(X_train, y_train[col])
                self.models[col] = svr

    def predict(self, X_test):
        """
        Devuelve predicciones con los modelos SVR entrenados.
        """
        predictions = []
        for col in self.models:
            pred = self.models[col].predict(X_test)
            predictions.append(pred)
        y_pred = np.column_stack(predictions)
        return y_pred

    def idk_run(self, X_params_dict):
        if hasattr(self, "X_train") and hasattr(self.X_train, "columns"):
            if len(X_params_dict) != len(self.X_train.columns):
                raise ValueError("El número de variables de entrada no coincide con el número de columnas en X_train.")
        else:
            print("Advertencia: no se pudo verificar las columnas de X_train.")

        X = np.array([list(X_params_dict.values())], dtype=float)

        y_pred_scaled = self.predict(X)

        # desescalar
        output_scaler = joblib.load(os.path.join(self.output_folder, 'output_scaler.pkl'))
        y_pred_orig = output_scaler.inverse_transform(y_pred_scaled)[0]

        if hasattr(self, "y_train") and hasattr(self.y_train, "columns"):
            keys = list(self.y_train.columns)
        else:
            keys = [f"result{i+1}" for i in range(len(y_pred_orig))]

        results = {k: float(v) for k, v in zip(keys, y_pred_orig)}
        return results
