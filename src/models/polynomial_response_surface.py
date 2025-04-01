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
        self.interaction_only = rom_config['hyperparams']['interaction_only'] 
        self.include_bias = rom_config['hyperparams']['include_bias'] 
        self.order = rom_config['hyperparams']['order'] 
        self.fit_intercept = rom_config['hyperparams']['fit_intercept'] 
        self.positive = rom_config['hyperparams']['positive'] 

        self.random_state = random_state
        self.model_name = rom_config['model_name']
        self.poly = PolynomialFeatures(degree=self.degree, interaction_only=self.interaction_only,
                                        include_bias=self.include_bias, order=self.order)
        self.model = LinearRegression(fit_intercept=self.fit_intercept, positive=self.positive)

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.train_losses = []
        self.val_losses = []


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
