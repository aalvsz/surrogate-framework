from sklearn.svm import SVR
from src.idkrom import idkROM
import os
import joblib
import pandas as pd
import numpy as np
from src.visualization.metrics import ModelReportGenerator

class SVRROM(idkROM.Modelo):
    """
    A class for creating and training a Support Vector Regression (SVR) model.
    """

    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1):
        """
        Initializes the SVR model with the given parameters.

        Args:
            kernel (str): The kernel type to be used in the algorithm.
            C (float): Regularization parameter.
            epsilon (float): Epsilon in the epsilon-SVR model.
        """
        super().__init__()
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)

        # Variables for reporting (will be filled during training)
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

        self.model_name = 'svr'

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
            'C': self.C,
            'epsilon': self.epsilon
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
        self.model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)
        return self

    def fit(self, X_train, y_train):
        """
        Fits the SVR model to the training data.
        
        Args:
            X_train (pd.DataFrame): Training input data.
            y_train (pd.DataFrame): Training target data.
        """
        self.train(X_train, y_train, X_train, y_train)

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame):
        """
        Trains the SVR model and saves the model and predictions.

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

        # Reshape y_train if necessary
        if y_train.ndim == 2 and y_train.shape[1] == 1:
            y_train = y_train.ravel()

        # Fit the model to the training data
        self.model.fit(X_train, y_train)

        # Save the model
        output_folder = os.path.join(os.getcwd(), 'results', self.model_name)
        os.makedirs(output_folder, exist_ok=True)
        model_path = os.path.join(output_folder, 'svr_model.pkl')
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
        Makes predictions with the trained SVR model.

        Args:
            X_test (pd.DataFrame): Test input data.

        Returns:
            np.ndarray: Model predictions.
        """
        y_pred = self.model.predict(X_test)
        return y_pred

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame, y_pred: np.ndarray, output_scaler=None) -> float:
        """
        Evaluates the model with the test data and saves the predictions.

        Args:
            X_test (pd.DataFrame): Test input data.
            y_test (pd.DataFrame): Test output data.
            y_pred (np.ndarray): Model predictions.
            output_scaler (optional): Scaler used to transform the outputs during preprocessing.

        Returns:
            float: The MSE in the scaled form.
        """
        # Save the predictions
        self.save_predictions(X_test, y_test, y_pred)

        # Calculate the MSE in the current scale
        mse_scaled = np.mean((y_pred - y_test) ** 2)
        print(f"MSE in scaled form: {mse_scaled}")

        # If the scaler is provided, also calculate the MSE in the original scale
        if output_scaler is not None:
            y_pred_original = output_scaler.inverse_transform(y_pred.reshape(-1, y_test.shape[1]))
            y_test_original = output_scaler.inverse_transform(y_test.to_numpy())
            mse_original = np.mean((y_pred_original - y_test_original) ** 2)
            print(f"MSE in original scale: {mse_original}")

        # Generate the report for metrics
        report_generator = ModelReportGenerator(
            model=self,
            train_losses=[],  # No training loss history for SVR
            val_losses=[],
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=X_test,
            y_test=y_test,
            model_name=self.model_name
        )
        report_generator.save_model_and_metrics()

        return mse_scaled