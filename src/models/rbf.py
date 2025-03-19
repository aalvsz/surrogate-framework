import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import rbf_kernel
import joblib
from src.idkrom import idkROM
from src.visualization.metrics import ModelReportGenerator

class RBFROM(idkROM.Modelo):
    """
    A class for creating and training a Radial Basis Function (RBF) model.
    """

    def __init__(self, gamma: float = 1.0):
        """
        Initializes the RBF model with the given parameters.

        Args:
            gamma (float): The gamma parameter for the RBF kernel.
        """
        super().__init__()

        self.gamma = gamma
        self.model_name = 'rbf'
        self.model = None

        # Variables for reporting (will be filled during training)
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Args:
            deep (bool): If True, will return the parameters for this estimator and contained subobjects that are estimators.

        Returns:
            dict: Parameter names mapped to their values.
        """
        return {
            'gamma': self.gamma
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
        return self

    def fit(self, X_train, y_train):
        """
        Fits the RBF model to the training data.
        
        Args:
            X_train (pd.DataFrame): Training input data.
            y_train (pd.DataFrame): Training target data.
        """
        self.train(X_train, y_train, X_train, y_train)


    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame):
        """
        Trains the RBF model.

        Args:
            X_train (pd.DataFrame): Training input data.
            y_train (pd.DataFrame): Training output data.
            X_val (pd.DataFrame): Validation input data.
            y_val (pd.DataFrame): Validation output data.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        # Use RBF kernel to calculate distances between training points
        self.model = rbf_kernel(X_train, X_train, gamma=self.gamma)

        # For simplicity, we use a least squares regression to get the weights of the model
        # using the RBF kernel matrix
        self.weights = np.linalg.pinv(self.model) @ y_train

        # Save the model
        output_folder = os.path.join(os.getcwd(), 'results', self.model_name)
        os.makedirs(output_folder, exist_ok=True)
        model_path = os.path.join(output_folder, 'rbf_model.pkl')
        with open(model_path, 'wb') as f:
            joblib.dump(self, f)

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

        # Compute the RBF kernel between the test set and training set
        rbf_test = rbf_kernel(X_test, self.X_train, gamma=self.gamma)

        # Predict using the RBF model weights
        return rbf_test @ self.weights

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame, y_pred: np.ndarray, output_scaler=None) -> float:
        """
        Evaluates the model with the test data and saves the predictions.

        Args:
            X_test (pd.DataFrame): Test input data.
            y_test (pd.DataFrame): True output data for the test set.
            y_pred (np.ndarray): Model predictions.
            output_scaler (optional): Scaler used to transform the outputs during preprocessing.

        Returns:
            float: The MSE in the scaled form.
        """
        # Save predictions
        self.save_predictions(X_test, y_test, y_pred)

        # Calculate MSE
        mse_scaled = mean_squared_error(y_test, y_pred)
        print(f"MSE in normalized scale: {mse_scaled}")

        # If output_scaler is provided, calculate the MSE in the original scale
        if output_scaler is not None:
            y_pred_original = output_scaler.inverse_transform(y_pred)
            y_test_original = output_scaler.inverse_transform(y_test.to_numpy())
            mse_original = mean_squared_error(y_test_original, y_pred_original)
            print(f"MSE in original scale: {mse_original}")

        # Generate the report for metrics
        report_generator = ModelReportGenerator(
            model=self,
            train_losses=[],  # For RBF, no training loss history
            val_losses=[],
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=X_test,
            y_test=y_test,
            model_name=self.model_name
        )
        report_generator.save_model_and_metrics()

        return mse_scaled

    def save_predictions(self, X_test: pd.DataFrame, y_test: pd.DataFrame, predictions: np.ndarray):
        """
        Saves the predictions and test data to a CSV file.

        Args:
            X_test (pd.DataFrame): Test input data.
            y_test (pd.DataFrame): Test output data.
            predictions (np.ndarray): Model predictions, can be 1D or 2D array.
        """
        # Create DataFrame with input features
        results = pd.DataFrame(X_test, columns=[f'Input_{i}' for i in range(X_test.shape[1])])
        
        # Convert predictions to numpy array if it's not already
        predictions = np.array(predictions)
        
        # Handle multiple outputs
        if isinstance(y_test, pd.DataFrame) and y_test.shape[1] > 1:
            for col in range(y_test.shape[1]):
                results[f'True_Output_{col}'] = y_test.iloc[:, col]
                if predictions.ndim > 1:
                    results[f'Predicted_Output_{col}'] = predictions[:, col]
                else:
                    results[f'Predicted_Output_{col}'] = predictions
        else:
            # Single output
            results['True_Output'] = y_test.squeeze()
            results['Predicted_Output'] = predictions.squeeze()

        # Save the predictions
        output_path = os.path.join(os.getcwd(), 'results', self.model_name, f'{self.model_name}_predictions.csv')
        results.to_csv(output_path, index=False)
        print(f"Predictions saved at: {output_path}")
