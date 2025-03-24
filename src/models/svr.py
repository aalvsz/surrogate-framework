from sklearn.svm import SVR
from src.idkrom import idkROM
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.visualization.metrics import ModelReportGenerator

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
        self.kernel = rom_config['hyperparams'].get('kernel', 'rbf')
        self.C = rom_config['hyperparams'].get('C', 1.0)
        self.epsilon = rom_config['hyperparams'].get('epsilon', 0.1)
        self.random_state = random_state
        self.model_name = rom_config['model_name']
        self.model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)

        # Variables for reporting (will be filled during training)
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.train_losses = []
        self.val_losses = []
        self.rom_config = rom_config

    def calculate_bic(self, y_true, y_pred):
        """
        Calculates the Bayesian Information Criterion (BIC) for the SVR model.

        Args:
            y_true (np.ndarray or pd.DataFrame): True labels for the test data.
            y_pred (np.ndarray): Predictions made by the model on the test data.

        Returns:
            float: The BIC value.
        """
        n = len(y_true)
        if n == 0:
            return np.inf  # Return infinity if no test data

        # Convert y_true to a NumPy array if it's a DataFrame
        if isinstance(y_true, pd.DataFrame):
            y_true_np = y_true.values
        else:
            y_true_np = y_true

        # Ensure y_true_np has the same shape as y_pred for element-wise comparison
        if y_true_np.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true shape is {y_true_np.shape}, y_pred shape is {y_pred.shape}. They should be the same for BIC calculation.")

        mse = np.mean((y_pred - y_true_np) ** 2)
        sse = mse * n
        # Approximate number of parameters by the number of support vectors (sum over all models)
        num_params = sum(model.support_.shape[0] for model in self.models) if hasattr(self, 'models') else 0
        bic = n * np.log(sse) + num_params * np.log(n)
        return bic

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
            model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)
            model.fit(X_train, y_train_column)
            self.models.append(model)

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

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame, y_pred: np.ndarray, output_scaler=None) -> float:
        print("Verificación de que y_test y y_pred tengan la misma forma:")
        print("Forma de y_test:", y_test.shape)
        print("Forma de y_pred:", y_pred.shape)

        # Convert to numpy arrays for consistency
        y_test_np = y_test.to_numpy()
        y_pred_np = y_pred

        # Calculate MSE (mean over all outputs)
        mse_scaled = np.mean(mean_squared_error(y_test_np, y_pred_np, multioutput='raw_values'))
        print(f"MSE en escala normalizada: {mse_scaled:.4f}")
        mse_percentage = (mse_scaled / np.mean(np.abs(y_test_np))) * 100 if np.mean(np.abs(y_test_np)) != 0 else 0 # MSE en porcentaje
        print(f"MSE en porcentaje: {mse_percentage:.2f}%")

        # Calculate MAE (mean over all outputs)
        mae_scaled = np.mean(mean_absolute_error(y_test_np, y_pred_np, multioutput='raw_values'))
        mae_percentage = (mae_scaled / np.mean(np.abs(y_test_np))) * 100 if np.mean(np.abs(y_test_np)) != 0 else 0 # MAE en porcentaje
        print(f"MAE en escala normalizada: {mae_scaled:.4f}")
        print(f"MAE en porcentaje: {mae_percentage:.2f}%")

        print(f"Diferencia entre MSE y MAE = {abs(mse_percentage-mae_percentage):.2f}%")

        # Calculate BIC
        bic_value = self.calculate_bic(y_test, y_pred)
        print(f"Valor de BIC: {bic_value:.2f}")

        if output_scaler is not None:
            y_pred_original = output_scaler.inverse_transform(y_pred.reshape(-1, y_test.shape[1]))
            y_test_original = output_scaler.inverse_transform(y_test.to_numpy())
            mse_original = np.mean(mean_squared_error(y_test_original, y_pred_original, multioutput='raw_values'))
            print(f"MSE en escala original: {mse_original}")

            # Calcular MAE en la escala original
            mae_original = np.mean(mean_absolute_error(y_test_original, y_pred_original, multioutput='raw_values'))
            mae_original_percentage = (mae_original / np.mean(np.abs(y_test_original))) * 100 if np.mean(np.abs(y_test_original)) != 0 else 0
            print(f"MAE en escala original: {mae_original}")
            print(f"MAE en escala original (porcentaje): {mae_original_percentage:.2f}%")

        # Calcular la diferencia entre el training loss y el validation loss en porcentaje (promedio)
        if len(self.train_losses) > 0 and len(self.val_losses) > 0:
            mean_train_loss = np.mean(self.train_losses)
            mean_val_loss = np.mean(self.val_losses)
            loss_difference_percentage = ((mean_train_loss - mean_val_loss) / mean_train_loss) * 100 if mean_train_loss != 0 else 0
            print(f"Diferencia entre Training Loss y Validation Loss: {loss_difference_percentage:.2f}%")

        print(f"Este es el diccionario que se come el modelo: {self.rom_config}")

        return mse_scaled