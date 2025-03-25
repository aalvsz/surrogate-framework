import os
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from src.idkrom import idkROM
from src.visualization.metrics import ErrorMetrics
import joblib

class GaussianProcessROM(idkROM.Modelo):

    def __init__(self, rom_config, random_state):
        super().__init__(rom_config, random_state)
        
        # Extraer parámetros de configuración
        self.kernel = rom_config['hyperparams']['kernel']
        self.noise = rom_config['hyperparams']['noise']
        self.optimizer = rom_config['hyperparams']['optimizer']
        self.random_state = random_state
   
        
        self.model_name = rom_config['model_name']
        
        # Configurar el kernel
        if self.kernel == "RBF":
            self.kernel_instance = RBF(length_scale=1.0)
        elif self.kernel == "Matern":
            self.kernel_instance = Matern(length_scale=1.0, nu=1.5)
        else:
            raise ValueError(f"Kernel {self.kernel} no soportado")

        if rom_config['mode'] != 'best':
            # Crear el GaussianProcessRegressor
            self.model = GaussianProcessRegressor(kernel=self.kernel_instance, alpha=self.noise, optimizer=self.optimizer, random_state=random_state)

        # Variables para el reporte (se llenarán durante el entrenamiento)
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None


    def train(self, X_train, y_train, X_val, y_val):
        """
        Entrena el modelo de Proceso Gaussiano y guarda el modelo y las predicciones.

        Args:
            X_train (pd.DataFrame): Datos de entrada de entrenamiento.
            y_train (pd.DataFrame): Datos de salida de entrenamiento.
            X_val (pd.DataFrame): Datos de entrada de validación.
            y_val (pd.DataFrame): Datos de salida de validación.
        """
        # Almacenar datos para reporte
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        # Entrenar el modelo con los datos de entrenamiento
        self.model.fit(X_train, y_train)

        # Guardar el modelo entrenado
        output_folder = os.path.join(os.getcwd(), 'results', self.model_name)
        os.makedirs(output_folder, exist_ok=True)
        model_path = os.path.join(output_folder, 'gp_model.pkl')
        with open(model_path, 'wb') as f:
            joblib.dump(self.model, f)

        print(f"Modelo guardado en: {model_path}")

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

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame, y_pred: np.ndarray, output_scaler=None) -> float:
        """
        Evalúa el modelo con los datos de test y guarda las predicciones.
        Si se proporciona 'output_scaler', también se calcula el MSE en la escala original.

        Args:
            X_test (pd.DataFrame): Datos de entrada del conjunto de test.
            y_test (pd.DataFrame): Datos verdaderos de salida del conjunto de test.
            y_pred (np.ndarray): Predicciones del modelo.
            output_scaler (opcional): Scaler usado para transformar los outputs durante el preprocesamiento.

        Returns:
            float: El MSE en la escala normalizada.
        """

        # Calcular el MSE en la escala en la que están (normalizada)
        mse_scaled = np.mean((y_pred - y_test) ** 2)
        mse_percentage = (mse_scaled / np.mean(np.abs(y_test))) * 100  # MSE en porcentaje
        print(f"MSE en escala normalizada: {mse_scaled:.4f}")
        print(f"MSE en porcentaje: {mse_percentage:.2f}%")

        # Calcular MAE en la escala normalizada
        mae_scaled = np.mean(np.abs(y_pred - y_test))
        mae_percentage = (mae_scaled / np.mean(np.abs(y_test))) * 100  # MAE en porcentaje
        print(f"MAE en escala normalizada: {mae_scaled:.4f}")
        print(f"MAE en porcentaje: {mae_percentage:.2f}%")

        print(f"Diferencia entre MSE y MAE = {abs(mse_percentage - mae_percentage):.2f}%")

        # Si se proporciona el scaler, se calcula también el MSE y MAE en la escala original.
        if output_scaler is not None:
            # Asegurarse de que las dimensiones sean compatibles para inverse_transform:
            y_pred_original = output_scaler.inverse_transform(y_pred.reshape(-1, y_test.shape[1]))
            y_test_original = output_scaler.inverse_transform(y_test.to_numpy())
            
            # Calcular MSE y MAE en la escala original
            mse_original = np.mean((y_pred_original - y_test_original) ** 2)
            mse_percentage_original = (mse_original / np.mean(np.abs(y_test_original))) * 100
            print(f"MSE en escala original: {mse_original:.4f}")
            print(f"MSE en porcentaje (escala original): {mse_percentage_original:.2f}%")

            mae_original = np.mean(np.abs(y_pred_original - y_test_original))
            mae_percentage_original = (mae_original / np.mean(np.abs(y_test_original))) * 100
            print(f"MAE en escala original: {mae_original:.4f}")
            print(f"MAE en porcentaje (escala original): {mae_percentage_original:.2f}%")

            print(f"Diferencia entre MSE y MAE (escala original) = {abs(mse_percentage_original - mae_percentage_original):.2f}%")

        # Create error visualization metrics
        errors = ErrorMetrics(self, self.model_name, y_test, y_pred)
        errors.create_error_graphs()

        return mse_scaled

