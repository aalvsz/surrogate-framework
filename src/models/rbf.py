import os
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
import joblib
from src.idkrom import idkROM

class RBFROM(idkROM.Modelo):
    """
    A class for creating and training a Radial Basis Function (RBF) model using Kernel Ridge Regression.
    """
    def __init__(self, rom_config, random_state):
        """
        Initializes the RBF model with the given parameters.

        Args:
            rom_config (dict): Configuration dictionary containing hyperparameters.
            random_state (int): Random state for reproducibility.
        """
        super().__init__(rom_config, random_state)

        # Extract hyperparameters
        self.alpha = rom_config['hyperparams']['alpha']
        self.kernel = rom_config['hyperparams']['kernel']
        self.gamma = rom_config['hyperparams']['gamma']
        self.degree = rom_config['hyperparams']['degree']
        
        self.model_name = rom_config['model_name']
        
        # Initialize the Kernel Ridge Regression model with RBF kernel
        self.model = KernelRidge(alpha=self.alpha, kernel='rbf', gamma=self.gamma)

        # Variables for reporting
        self.train_losses = []
        self.val_losses = []
        
    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame):
        """
        Trains the RBF model using Kernel Ridge Regression.

        Args:
            X_train (pd.DataFrame): Training input data.
            y_train (pd.DataFrame): Training output data.
            X_val (pd.DataFrame): Validation input data.
            y_val (pd.DataFrame): Validation output data.
        """
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Compute training loss
        y_train_pred = self.model.predict(X_train)
        mse_train = mean_squared_error(y_train, y_train_pred)
        self.train_losses.append(mse_train)
        print(f"Training MSE: {mse_train}")

        # Compute validation loss
        y_val_pred = self.model.predict(X_val)
        mse_val = mean_squared_error(y_val, y_val_pred)
        self.val_losses.append(mse_val)
        print(f"Validation MSE: {mse_val}")

        # Save the trained model
        output_folder = os.path.join(os.getcwd(), 'results', self.model_name)
        os.makedirs(output_folder, exist_ok=True)
        model_path = os.path.join(output_folder, 'rbf_model.pkl')
        with open(model_path, 'wb') as f:
            joblib.dump(self.model, f)
        print(f"Model saved at: {model_path}")

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions with the trained RBF model.

        Args:
            X_test (pd.DataFrame): Test input data.

        Returns:
            np.ndarray: Model predictions.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet!")

        return self.model.predict(X_test)
