import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from src.idkrom import idkROM
from src.visualization.metrics import ErrorMetrics
from scipy.special import comb

class PolynomialResponseSurface(idkROM.Modelo):
    def __init__(self, rom_config, random_state):
        super().__init__(rom_config, random_state)

        # Extraer parámetros de configuración
        self.degree = rom_config['hyperparams']['degree']
        self.random_state = random_state
        self.model_name = rom_config['model_name']
        self.poly = PolynomialFeatures(degree=self.degree)
        self.model = LinearRegression()

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.train_losses = []
        self.val_losses = []

    def calculate_bic(self, y_true, y_pred):
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
        # Number of parameters in polynomial regression
        num_input_features = self.X_train.shape[1] if self.X_train is not None else 0
        num_params = comb(num_input_features + self.degree, self.degree, exact=True) + 1 # +1 for the intercept
        bic = n * np.log(sse) + num_params * np.log(n)
        return bic

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        X_poly_train = self.poly.fit_transform(X_train)
        self.model.fit(X_poly_train, y_train)

        # Calculate training loss
        y_train_pred = self.predict(X_train)
        mse_train = mean_squared_error(y_train, y_train_pred)
        self.train_losses.append(mse_train)
        print(f"Training MSE: {mse_train}")

        # Calculate validation loss
        y_val_pred = self.predict(X_val)
        mse_val = mean_squared_error(y_val, y_val_pred)
        self.val_losses.append(mse_val)
        print(f"Validation MSE: {mse_val}")

        # Save the model
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

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame, y_pred: np.ndarray, eval_metrics: str, output_scaler=None) -> float:
        """
        Evalúa el modelo con los datos de test y guarda las predicciones.
        Si se proporciona 'output_scaler', también se calcula el MSE en la escala original.

        Args:
            X_test (pd.DataFrame): Datos de entrada del conjunto de test.
            y_test (pd.DataFrame): Datos verdaderos de salida del conjunto de test.
            y_pred (np.ndarray): Predicciones del modelo.
            eval_metrics (str): Métrica de evaluación a utilizar.
            output_scaler (opcional): Scaler usado para transformar los outputs durante el preprocesamiento.

        Returns:
            float: El MSE en la escala normalizada.
        """

        print("Verificación de que y_test y y_pred tengan la misma forma:")
        print("Forma de y_test:", y_test.shape)
        print("Forma de y_pred:", y_pred.shape)

        # Convertir a numpy arrays
        y_test_np = y_test.to_numpy()
        y_pred_np = np.array(y_pred)

        if eval_metrics == 'mse':
            mse_scaled = np.mean((y_pred_np - y_test_np) ** 2)
            mse_percentage = (mse_scaled / np.mean(np.abs(y_test_np))) * 100
            print(f"MSE en escala normalizada: {mse_scaled:.4f}")
            print(f"MSE en porcentaje: {mse_percentage:.2f}%")

        elif eval_metrics == 'mae':
            mae_scaled = np.mean(np.abs(y_pred_np - y_test_np))
            mae_percentage = (mae_scaled / np.mean(np.abs(y_test_np))) * 100
            print(f"MAE en escala normalizada: {mae_scaled:.4f}")
            print(f"MAE en porcentaje: {mae_percentage:.2f}%")

        elif eval_metrics == 'mape':
            epsilon = 1e-10
            mape = np.mean(np.abs((y_test_np - y_pred_np) / (y_test_np + epsilon))) * 100
            print(f"MAPE: {mape:.2f}%")

        # Calcular BIC
        bic_value = self.calculate_bic(y_test, y_pred)
        print(f"Valor de BIC: {bic_value:.2f}")

        # Calcular la diferencia entre el training loss y el validation loss en porcentaje
        if hasattr(self, 'train_losses') and hasattr(self, 'val_losses') and len(self.train_losses) > 0 and len(self.val_losses) > 0:
            last_train_loss = self.train_losses[-1]
            last_val_loss = self.val_losses[-1]
            loss_difference_percentage = ((last_train_loss - last_val_loss) / last_train_loss) * 100
            print(f"Diferencia entre Training Loss y Validation Loss: {loss_difference_percentage:.2f}%")

        print(f"Este es el diccionario que se come el modelo: {self.rom_config}")

        # Create error visualization metrics
        errors = ErrorMetrics(self, self.model_name, y_test, y_pred)
        errors.create_error_graphs()

        return mse_scaled
