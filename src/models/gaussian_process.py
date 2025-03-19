from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from src.idkrom import idkROM
from src.visualization.metrics import ModelReportGenerator
import numpy as np
import os
import pandas as pd
import joblib

class GaussianProcessROM(idkROM.Modelo):
    """
    A class for creating and training a Gaussian Process model.
    """

    def __init__(self, kernel: str = "RBF", noise: float = 1e-2, optimizer: str = None):
        """
        Initializes the Gaussian Process model with the given parameters.

        Args:
            kernel (str): The kernel to use in the Gaussian Process ("RBF" or "Matern").
            noise (float): The noise level in the Gaussian Process.
            optimizer (str, optional): The optimizer to use for hyperparameter tuning. Defaults to None.
        """
        super().__init__()

        # Initialize parameters
        self.kernel = kernel
        self.noise = noise
        self.optimizer = optimizer

        # Configure the kernel based on the selection
        if kernel == "RBF":
            self.kernel_instance = RBF(length_scale=1.0)  # RBF kernel with default length scale
        elif kernel == "Matern":
            self.kernel_instance = Matern(length_scale=1.0, nu=1.5)  # Matern kernel with default length scale
        else:
            raise ValueError(f"Kernel {kernel} not supported")

        # Create the GaussianProcessRegressor with the kernel and other parameters
        self.model = GaussianProcessRegressor(kernel=self.kernel_instance, alpha=noise, optimizer=optimizer)

        # Variables for reporting (will be filled during training)
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

        self.model_name = 'gaussian_process'


    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Args:
            deep (bool): If True, will return the parameters for this estimator and contained subobjects that are estimators.

        Returns:
            dict: Parameter names mapped to their values.
        """
        return {
            'kernel': self.kernel,
            'noise': self.noise,
            'optimizer': self.optimizer
        }

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Args:
            **params: Estimator parameters.

        Returns:
            self: Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        if self.kernel == "RBF":
            self.kernel_instance = RBF(length_scale=1.0)
        elif self.kernel == "Matern":
            self.kernel_instance = Matern(length_scale=1.0, nu=1.5)
        self.model = GaussianProcessRegressor(kernel=self.kernel_instance, alpha=self.noise, optimizer=self.optimizer)
        return self


    def fit(self, X_train, y_train):
        """
        Fits the gaussian process model to the training data.
        
        Args:
            X_train (pd.DataFrame): Training input data.
            y_train (pd.DataFrame): Training target data.
        """
        self.train(X_train, y_train, X_train, y_train)


    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame):
        """
        Trains the Gaussian Process model and saves the model and predictions.

        Args:
            X_train (pd.DataFrame): Training input data.
            y_train (pd.DataFrame): Training output data.
            X_val (pd.DataFrame): Validation input data.
            y_val (pd.DataFrame): Validation output data.
        """
        # Store data for reporting
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        # Fit the model to the training data
        self.model.fit(X_train, y_train)

        # Save the model
        output_folder = os.path.join(os.getcwd(), 'results', self.model_name)
        os.makedirs(output_folder, exist_ok=True)
        model_path = os.path.join(output_folder, 'gp_model.pkl')
        with open(model_path, 'wb') as f:
            joblib.dump(self.model, f)

        print(f"Model saved at: {model_path}")

    def save_predictions(self, X_test: pd.DataFrame, y_test: pd.DataFrame, predictions: pd.DataFrame):
        """
        Saves the predictions and test data to a CSV file.

        Args:
            X_test (pd.DataFrame): Test input data.
            y_test (pd.DataFrame): Test output data.
            predictions (pd.DataFrame): Model predictions.
        """
        results = pd.DataFrame(X_test, columns=[f'Input_{i}' for i in range(X_test.shape[1])])
        
        y_test = np.array(y_test)
        predictions = np.array(predictions)
        
        if y_test.ndim > 1 and y_test.shape[1] > 1:
            # Save each output column separately
            for col in range(y_test.shape[1]):
                results[f'True_Output_{col}'] = y_test[:, col]
                results[f'Predicted_Output_{col}'] = predictions[:, col]
        else:
            results['True_Output'] = y_test.squeeze()
            results['Predicted_Output'] = predictions.squeeze()

        # Save the predictions
        output_path = os.path.join(os.getcwd(), 'results', self.model_name, f'{self.model_name}_predictions.csv')
        results.to_csv(output_path, index=False)
        print(f"Predictions saved at: {output_path}")

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions with the trained model.

        Args:
            X_test (pd.DataFrame): Test input data.

        Returns:
            np.ndarray: Model predictions.
        """
        y_pred = self.model.predict(X_test)
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
        # Guardar las predicciones (esto queda igual)
        self.save_predictions(X_test, y_test, y_pred)

        # Calcular el MSE en la escala en la que están (normalizada)
        mse_scaled = np.mean((y_pred - y_test) ** 2)
        print(f"MSE en escala normalizada: {mse_scaled}")

        # Si se proporciona el scaler, se calcula también el MSE en la escala original.
        if output_scaler is not None:
            # Asegurarse de que las dimensiones sean compatibles para inverse_transform:
            y_pred_original = output_scaler.inverse_transform(y_pred.reshape(-1, y_test.shape[1]))
            y_test_original = output_scaler.inverse_transform(y_test.to_numpy())
            mse_original = np.mean((y_pred_original - y_test_original) ** 2)
            print(f"MSE en escala original: {mse_original}")

        # Generar reporte de métricas (similar a la red neuronal)
        report_generator = ModelReportGenerator(
            model=self,
            train_losses=[],  # Para Gaussian Process no tenemos un histórico de pérdidas
            val_losses=[],
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=X_test,
            y_test=y_test,
            model_name=self.model_name
        )
        report_generator.save_model_and_metrics()

        return mse_scaled
