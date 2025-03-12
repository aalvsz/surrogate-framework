from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
import numpy as np
from src.idkROM_v0 import idkROM
import os
import pandas as pd

class GaussianProcessROM(idkROM.Modelo):
    def __init__(self, kernel="RBF", noise=1e-2, optimizer=None):
        super().__init__()

        # Inicialización de parámetros
        self.kernel = kernel
        self.noise = noise
        self.optimizer = optimizer

        # Configura el kernel basado en la selección
        if kernel == "RBF":
            self.kernel_instance = RBF(length_scale=1.0)  # Kernel RBF con escala de longitud predeterminada
        elif kernel == "Matern":
            self.kernel_instance = Matern(length_scale=1.0, nu=1.5)  # Kernel Matern con escala de longitud predeterminada
        else:
            raise ValueError(f"Kernel {kernel} no soportado")

        # Crear el GaussianProcessRegressor con el kernel y otros parámetros
        self.model = GaussianProcessRegressor(kernel=self.kernel_instance, alpha=noise, optimizer=optimizer)


    def train(self, X_train, y_train):
        """Entrena el modelo de Proceso Gaussiano y guarda el modelo y las predicciones."""
        # Ajustar el modelo a los datos de entrenamiento
        self.model.fit(X_train, y_train)

        # Guardar el modelo
        idkROM.output_folder = os.path.join(os.getcwd(), 'results', f'{str(X_train.shape[0])}_samples')
        os.makedirs(idkROM.output_folder, exist_ok=True)
        model_path = os.path.join(idkROM.output_folder, 'gp_model.pkl')
        with open(model_path, 'wb') as f:
            import joblib
            joblib.dump(self.model, f)

        print(f"Modelo guardado en {model_path}")

    def save_predictions(self, X_train, y_train, predictions, model_name='gaussian_process'):
        """Guarda las predicciones y los datos de entrenamiento en un archivo CSV."""
        results = pd.DataFrame(X_train, columns=[f'Input_{i}' for i in range(X_train.shape[1])])
        
        y_train = np.array(y_train)
        predictions = np.array(predictions)
        
        if y_train.ndim > 1 and y_train.shape[1] > 1:
            # Guardar cada columna de salida por separado
            for col in range(y_train.shape[1]):
                results[f'True_Output_{col}'] = y_train[:, col]
                results[f'Predicted_Output_{col}'] = predictions[:, col]
        else:
            results['True_Output'] = y_train.squeeze()
            results['Predicted_Output'] = predictions.squeeze()

        # Guardar las predicciones
        output_path = os.path.join(idkROM.output_folder, f'{model_name}_predictions.csv')
        results.to_csv(output_path, index=False)
        print(f"Predicciones guardadas en {output_path}")

    def predict(self, X):
        """Hace predicciones con el modelo entrenado."""
        predictions = self.model.predict(X)
        return predictions

    def evaluate(self, X_test, y_test):
        """Evalúa el modelo con los datos de prueba y guarda las predicciones."""
        predictions = self.predict(X_test)
        self.save_predictions(X_test, y_test, predictions)
        return predictions

    def score(self, X_test, y_test):
        """Calcula el error cuadrático medio entre las predicciones y los datos de prueba."""
        predictions = self.evaluate(X_test, y_test)
        mse = np.mean((predictions - y_test) ** 2)
        print(f"Mean Squared Error on Test Data: {mse}")
        return mse
