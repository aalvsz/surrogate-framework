import os
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, ExpSineSquared
from new_main import idkROM
from src.visualization.metrics import ErrorMetrics
import joblib

class GaussianProcessROM(idkROM.Modelo):

    def __init__(self, rom_config, random_state):
        super().__init__(rom_config, random_state)
        
        # Extraer parámetros de configuración
        self.kernel = rom_config['hyperparams']['kernel']
        self.cst_kernel = rom_config['hyperparams']['constant_kernel']
        self.matern_nu = rom_config['hyperparams']['matern_nu']
        self.expsine_periodicity = rom_config['hyperparams']['expsine_periodicity']
        self.alpha = rom_config['hyperparams']['alpha']
        self.optimizer = rom_config['hyperparams']['optimizer']
        self.random_state = random_state
   
        
        self.model_name = rom_config['model_name']
        
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
