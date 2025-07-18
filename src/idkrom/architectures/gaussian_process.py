import os
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, ExpSineSquared
from sklearn.model_selection import KFold
import joblib
from idkrom.model import idkROM

class GaussianProcessROM(idkROM.Modelo):
    """
    Modelo de Proceso Gaussiano para regresión, basado en scikit-learn.

    Permite configurar diferentes kernels, realizar validación cruzada o explícita,
    y realizar predicciones y desescalado de resultados.
    """

    def __init__(self, rom_config, random_state):
        """
        Inicializa el modelo de Proceso Gaussiano con los hiperparámetros y configuración proporcionados.

        Args:
            rom_config (dict): Diccionario de configuración del modelo ROM.
            random_state (int): Semilla para reproducibilidad.
        """
        super().__init__(rom_config, random_state)
        
        # Extraer parámetros de configuración
        self.kernel = rom_config['hyperparams']['kernel_gp']
        self.cst_kernel = rom_config['hyperparams']['constant_kernel']
        self.matern_nu = rom_config['hyperparams']['matern_nu']
        self.expsine_periodicity = rom_config['hyperparams']['expsine_periodicity']
        self.alpha = rom_config['hyperparams']['alpha_gp']
        self.optimizer = rom_config['hyperparams']['optimizer_gp']
        self.random_state = random_state
   
        self.model_name = rom_config['model_name']
        self.output_folder = rom_config['output_folder']

        # Configurar el kernel
        if self.kernel == "RBF":
            self.kernel_instance = RBF(length_scale=1.0) + ConstantKernel(constant_value=self.cst_kernel)
        elif self.kernel == "Matern":
            self.kernel_instance = Matern(length_scale=1.0, nu=self.matern_nu)
        elif self.kernel == "ExpSineSquared":
            self.kernel_instance = ExpSineSquared(length_scale=1.0, periodicity=self.expsine_periodicity)
        else:
            raise ValueError(f"Kernel {self.kernel} no soportado")

        if rom_config['mode'] != 'best':
            # Crear el GaussianProcessRegressor
            self.model = GaussianProcessRegressor(kernel=self.kernel_instance, alpha=self.alpha, optimizer=self.optimizer, random_state=random_state)

        # Variables para el reporte (se llenarán durante el entrenamiento)
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

    def train(self, X_train, y_train, X_val=None, y_val=None, validation_mode='cross'):
        """
        Entrena el modelo de Proceso Gaussiano y guarda el modelo y las predicciones.

        Args:
            X_train (pd.DataFrame): Datos de entrada de entrenamiento.
            y_train (pd.DataFrame): Datos de salida de entrenamiento.
            X_val (pd.DataFrame, opcional): Datos de entrada de validación.
            y_val (pd.DataFrame, opcional): Datos de salida de validación.
            validation_mode (str): 'cross' para validación cruzada, 'single' para validación explícita.
        """
        # Almacenar datos para reporte
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        # --- Selección de Modo: Validación Explícita vs. Cross-Validation ---
        perform_cv = True
        if validation_mode == 'single' and X_val is not None and y_val is not None:
            perform_cv = False
            print("Usando conjunto de validación explícito proporcionado.")
        else:
            print("Iniciando Cross-Validation.")

        if perform_cv:
            # Validación Cruzada
            kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Usar 5 folds, ajustable
            fold_val_losses = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
                print(f"--- Fold {fold+1}/5 ---")

                # Datos del fold actual
                X_train_fold = X_train.iloc[train_idx]
                y_train_fold = y_train.iloc[train_idx]
                X_val_fold = X_train.iloc[val_idx]
                y_val_fold = y_train.iloc[val_idx]

                # Entrenar el modelo con los datos del fold
                self.model.fit(X_train_fold, y_train_fold)

                # Evaluar el modelo en el conjunto de validación del fold
                val_preds = self.model.predict(X_val_fold)
                val_loss = np.mean((val_preds - y_val_fold) ** 2)  # Calculamos el MSE
                fold_val_losses.append(val_loss)

                print(f"  Fold {fold+1} Val Loss: {val_loss:.6f}")

            avg_val_loss = np.mean(fold_val_losses)
            print(f"\nPromedio de la pérdida de validación en CV: {avg_val_loss:.6f}")

        else:
            # Validación explícita
            print("Entrenando con datos de validación explícitos.")
            self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Realiza predicciones con el modelo entrenado.

        Args:
            X_test (pd.DataFrame): Datos de entrada para predicción.

        Returns:
            np.ndarray: Predicciones del modelo.
        """
        y_pred, std_dev = self.model.predict(X_test, return_std=True)
        print(f"Standard deviation is {np.mean(std_dev):.4f}")
        return y_pred

    def idk_run(self, X_params_dict):
        """
        Ejecuta el ROM usando los parámetros de entrada para hacer una predicción
        y mapea la salida a los nombres de columnas de self.y_train. Además, verifica que
        el diccionario de entrada contenga tantas llaves como columnas tiene self.X_train y
        que el número de resultados coincida con las columnas de self.y_train.

        Args:
            X_params_dict (dict): Diccionario con variables de entrada, ejemplo:
                                {'var1': 34, 'var2': 45, ...}

        Returns:
            dict: Diccionario con los resultados de la predicción, donde las llaves
                corresponden a los encabezados del DataFrame de entrenamiento self.y_train,
                por ejemplo: {'nombre_col1': 45, 'nombre_col2': 89, ...}
        """

        # Verificar que self.X_train exista y tenga columnas
        if hasattr(self, "X_train") and hasattr(self.X_train, "columns"):
            if len(X_params_dict) != len(self.X_train.columns):
                raise ValueError("El número de variables de entrada no coincide con el número de columnas en X_train. Se esperaban {} variables, pero se recibieron {}.".format(len(self.X_train.columns), len(X_params_dict)))
        else:
            print("Advertencia: No se pudo verificar el número de variables de X_train, 'X_train' no está definido o no tiene atributo 'columns'.")

        # 1) Crear array de entrada
        X = np.array([list(X_params_dict.values())], dtype=float)  # shape (1, n_features)

        # 2) Predicción escalada
        y_pred_scaled = self.predict(X)

        # 3) Aplanar la salida si es necesario
        if y_pred_scaled.ndim > 1:
            y_pred_flat = y_pred_scaled.flatten() if y_pred_scaled.shape[0] == 1 else y_pred_scaled[0]
        else:
            y_pred_flat = y_pred_scaled

        # 4) Desescalado
        output_scaler = joblib.load(os.path.join(self.output_folder, 'output_scaler.pkl'))

        y_in = y_pred_flat.reshape(1, -1) if y_pred_flat.ndim == 1 else y_pred_flat
        y_pred_orig = output_scaler.inverse_transform(y_in)[0]

        # 5) Construir diccionario de salida
        if hasattr(self, "y_train") and hasattr(self.y_train, "columns"):
            keys = list(self.y_train.columns)
            if len(keys) != len(y_pred_orig):
                raise ValueError(f"El número de resultados predichos ({len(y_pred_orig)}) no coincide con el número de columnas en y_train ({len(keys)}).")
        else:
            print("Advertencia: No se pudo obtener las columnas de y_train, se usarán llaves genéricas.")
            keys = [f"result{i+1}" for i in range(len(y_pred_orig))]

        results = {k: float(v) for k, v in zip(keys, y_pred_orig)}
        return results