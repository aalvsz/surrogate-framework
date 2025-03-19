import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
from src.idkrom import idkROM
from src.visualization.metrics import ModelReportGenerator

class PolynomialResponseSurface(idkROM.Modelo):
    def __init__(self, degree: int = 2):
        super().__init__()
        self.degree = degree
        self.model_name = 'response_surface'
        self.poly = PolynomialFeatures(degree=self.degree)
        self.model = LinearRegression()

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

    def get_params(self, deep=True):
        return {'degree': self.degree}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self.poly = PolynomialFeatures(degree=self.degree)
        return self

    def fit(self, X_train, y_train):
        self.train(X_train, y_train, X_train, y_train)

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        X_poly = self.poly.fit_transform(X_train)
        self.model.fit(X_poly, y_train)

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

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame, y_pred: np.ndarray, output_scaler=None) -> float:
        self.save_predictions(X_test, y_test, y_pred)

        mse_scaled = mean_squared_error(y_test, y_pred)
        print(f"MSE in normalized scale: {mse_scaled}")

        if output_scaler is not None:
            y_pred_original = output_scaler.inverse_transform(y_pred)
            y_test_original = output_scaler.inverse_transform(y_test.to_numpy())
            mse_original = mean_squared_error(y_test_original, y_pred_original)
            print(f"MSE in original scale: {mse_original}")

        report_generator = ModelReportGenerator(
            model=self,
            train_losses=[],
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
        results = pd.DataFrame(X_test, columns=[f'Input_{i}' for i in range(X_test.shape[1])])
        predictions = np.array(predictions)

        if isinstance(y_test, pd.DataFrame) and y_test.shape[1] > 1:
            for col in range(y_test.shape[1]):
                results[f'True_Output_{col}'] = y_test.iloc[:, col]
                if predictions.ndim > 1:
                    results[f'Predicted_Output_{col}'] = predictions[:, col]
                else:
                    results[f'Predicted_Output_{col}'] = predictions
        else:
            results['True_Output'] = y_test.squeeze()
            results['Predicted_Output'] = predictions.squeeze()

        output_path = os.path.join(os.getcwd(), 'results', self.model_name, f'{self.model_name}_predictions.csv')
        results.to_csv(output_path, index=False)
        print(f"Predictions saved at: {output_path}")
