from sklearn.svm import SVR
from src.idkrom import idkROM
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.visualization.metrics import ErrorMetrics

class SVRROM(idkROM.Modelo):
    """
    A class for creating and training a Support Vector Regression (SVR) model.
    """

    def __init__(self, rom_config, random_state):
        """
        Initializes the SVR model with the given parameters.

        Args:
            rom_config (dict): Configuration dictionary for the ROM.
            random_state (int): Seed for random number generation.
        """
        super().__init__(rom_config, random_state)

        # Extraer parámetros de configuración
        self.kernel = rom_config['hyperparams']['kernel']
        self.degree = rom_config['hyperparams']['degree']
        self.gamma = rom_config['hyperparams']['gamma']
        self.tolerance = rom_config['hyperparams']['tolerance']
        self.C = rom_config['hyperparams']['C']
        self.epsilon = rom_config['hyperparams']['epsilon']

        self.random_state = random_state
        self.model_name = rom_config['model_name']
        self.model = SVR(kernel=self.kernel, degree=self.degree, gamma=self.gamma,
                          tolerance=self.tolerance, C=self.C, epsilon=self.epsilon)

        # Variables for reporting (will be filled during training)
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.train_losses = []
        self.val_losses = []
        self.rom_config = rom_config

    
    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.models = [] # Lista para almacenar los modelos SVR individuales
        self.train_losses = []
        self.val_losses = []

        for i in range(y_train.shape[1]):
            print(f"Entrenando modelo para la salida {i+1}")
            y_train_column = y_train.iloc[:, i].values.ravel()  # Seleccionar una columna y convertirla a 1D
            self.model.fit(X_train, y_train_column)
            self.models.append(self.model)

            # Calcular la pérdida de entrenamiento para esta salida
            y_train_pred = self.predict_single_output(X_train, i)
            mse_train = mean_squared_error(y_train.iloc[:, i], y_train_pred)
            self.train_losses.append(mse_train)
            print(f"Training MSE (salida {i+1}): {mse_train}")

            # Calcular la pérdida de validación para esta salida
            y_val_pred = self.predict_single_output(X_val, i)
            mse_val = mean_squared_error(y_val.iloc[:, i], y_val_pred)
            self.val_losses.append(mse_val)
            print(f"Validation MSE (salida {i+1}): {mse_val}")

        # Guardar todos los modelos
        output_folder = os.path.join(os.getcwd(), 'results', self.model_name)
        os.makedirs(output_folder, exist_ok=True)
        model_path = os.path.join(output_folder, 'svr_model.pkl')
        with open(model_path, 'wb') as f:
            joblib.dump(self.models, f)  # Guardar la lista de modelos

        print(f"Modelos guardados en: {model_path}")

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
