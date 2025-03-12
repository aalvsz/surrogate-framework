import os
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.interpolate import RBFInterpolator
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd

class idkROM:
    def __init__(self):
        self.scaler = StandardScaler()

    def preprocessing(self, data):
        """Aplica normalización y detección de outliers"""

        # Display the DataFrame
        print(data.shape)
        print(data.dtypes)

        # Separar inputs y outputs
        self.inputs = data.iloc[:, :7]  # Primeras 7 columnas como inputs
        self.outputs = data.iloc[:, 7:]  # Resto como outputs

        n_samples = data.shape[0]

        #creamos la carpeta de salida
        self.output_folder = os.path.join(os.getcwd(), 'results', f'{str(n_samples)}_samples')
        os.makedirs(self.output_folder, exist_ok=True)

        #sample selection
        data = data.iloc[:n_samples, :]

        # Data profiling
        report = ProfileReport(data, title='Report', minimal=True)

        # exportar reporte a html
        report.to_file(os.path.join(self.output_folder, 'train_profiling_report.html'))
        
        
        def boxplot(data):
            # Create a box plot for each variable
            fig, axes = plt.subplots(nrows=(len(data.columns) + 3) // 4, ncols=4, figsize=(20, 5 * ((len(data.columns) + 3) // 4)))

            for i, column in enumerate(data.columns):
                row, col = divmod(i, 4)
                axes[row, col].boxplot(data[column])
                # axes[row, col].set_title(f'Box plot of {column}')
                axes[row, col].set_xlabel(column)
                axes[row, col].set_ylabel('Value')

            # Hide any empty subplots
            for j in range(i + 1, len(axes.flatten())):
                fig.delaxes(axes.flatten()[j])

            plt.savefig(os.path.join(self.output_folder, "preprocessed_data.png"))
            plt.close(fig)

        #print('Box plot for input & output variables')
        #boxplot(data)

        # Drop outliers
        df_filtered = data[(data['Q_total_int'] <= 0.00025) &
                   (data['h_mean_int'] <= 0.0002) &
                   (data['h_mean_ext'] <= 0.00015)]

        print(df_filtered.shape)

        #print('Box plot for input & output variables')
        #boxplot(df_filtered)

        # Normalize variables
        scaler = MinMaxScaler()
        df_normalized = pd.DataFrame(scaler.fit_transform(df_filtered), columns=df_filtered.columns)

        joblib.dump(scaler, os.path.join(self.output_folder, 'scaler.pkl'))

        boxplot(df_normalized)

        df_normalized.describe()

        X_normalized = df_normalized[self.inputs.columns]
        y_normalized = df_normalized[self.outputs.columns]

        # Guardar los datos preprocesados en un CSV
        preprocessed_file_path = os.path.join(self.output_folder, 'preprocessed_data.csv')
        df_normalized.to_csv(preprocessed_file_path, index=False, sep=";", decimal=".")
        print(f'Datos preprocesados guardados en {preprocessed_file_path}')

        return X_normalized, y_normalized


    def get_output_folder(self):
            if self.output_folder:
                return self.output_folder
            else:
                raise ValueError("El output_folder aún no ha sido definido. Asegúrate de ejecutar preprocessing() primero.")
            
    """Clase base abstracta
    Sirve de plantilla para las subclases de cada método ROM, donde se implementan las funciones train y evaluate dedicadas."""
    class Modelo:
        def __init__(self):
            self.model = None

        def train(self, X_train, y_train):
            """Método a sobrescribir en cada subclase"""
            raise NotImplementedError

        def evaluate(self, X_test, y_test):
            """Método a sobrescribir en cada subclase"""
            raise NotImplementedError
        
        def score(self, X_test, y_test):
            """Calcula el error cuadrático medio usando las predicciones devueltas por evaluate"""
            y_pred = self.evaluate(X_test, y_test)
            return mean_squared_error(y_test, y_pred)


    class NeuralNetworkROM(Modelo):  # Hereda de la clase base idkROM
        def __init__(self, input_dim=2, hidden_layers=2, neurons_per_layer=50, output_dim=1, learning_rate=1e-3, activation_function='Tanh'):
            """Inicializa la red neuronal con la arquitectura dada."""
            super().__init__()  # Llama al constructor de la clase base Modelo

            # Establecer la función de activación
            self.activation_function = activation_function

            # Crear la red neuronal usando una función separada
            self.nn = self.crear_red_neuronal(input_dim, hidden_layers, neurons_per_layer, output_dim)

            # Parámetros de entrenamiento
            self.learning_rate = learning_rate
            self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=self.learning_rate)

        def crear_red_neuronal(self, input_size, hidden_layers, neurons_per_layer, output_size):
            """Crea una red neuronal feedforward."""
            layers = []
            
            # Capa de entrada
            layers.append(torch.nn.Linear(input_size, neurons_per_layer))
            layers.append(self.get_activation_function())  # Usar la función de activación seleccionada
            
            # Capas ocultas
            for _ in range(hidden_layers - 1):
                layers.append(torch.nn.Linear(neurons_per_layer, neurons_per_layer))
                layers.append(self.get_activation_function())  # Usar la función de activación seleccionada
            
            # Capa de salida
            layers.append(torch.nn.Linear(neurons_per_layer, output_size))
            
            return torch.nn.Sequential(*layers)

        def get_activation_function(self):
            """Devuelve la función de activación correspondiente."""
            if self.activation_function == 'Tanh':
                return torch.nn.Tanh()
            elif self.activation_function == 'ReLU':
                return torch.nn.ReLU()
            elif self.activation_function == 'Sigmoid':
                return torch.nn.Sigmoid()
            else:
                raise ValueError(f"Función de activación desconocida: {self.activation_function}")


        def train(self, X_train, y_train, num_epochs=1000, batch_size=32):
            """Entrena la red neuronal."""

            #conversion de los datos a tensores
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

            loss_fn = torch.nn.MSELoss()  # Función de pérdida: error cuadrático medio

            for epoch in range(num_epochs):
                # Entrenamiento en lotes
                for i in range(0, len(X_train_tensor), batch_size):
                    # Obtener un lote de datos
                    X_batch = X_train_tensor[i:i+batch_size]
                    y_batch = y_train_tensor[i:i+batch_size]

                    # Forward pass (los datos pasan por las neuronas y se les aplican
                    #  las transformaciones lineales (pesos+bias) y la funcion de activacion)
                    predictions = self.nn(X_batch)

                    # Calcular la pérdida usando MSE (diferencia entre salida predicha y salida real)
                    loss = loss_fn(predictions, y_batch)

                    # Backward pass y optimización
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Mostrar el progreso cada 100 épocas
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item()}")

            # Save the model
            model_path = os.path.join(self.get_output_folder(), 'nn_model.pth')
            torch.save(nn, model_path)

        def evaluate(self, X_test, y_test):
            """Evalúa el modelo con los datos de prueba y devuelve las predicciones como array de NumPy."""
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            self.nn.eval()  # Modo evaluación

            with torch.no_grad():
                predictions = self.nn(X_test_tensor)
                # Elimina la dimensión extra: si predictions es de forma [N, 1] pasa a [N]
                predictions = predictions.squeeze()
                # Convertir a NumPy para usar con sklearn
                predictions_np = predictions.cpu().numpy()

                # (Opcional) Calcular el MSE usando torch para imprimirlo
                y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
                mse = torch.nn.functional.mse_loss(predictions, y_test_tensor)
                print(f"Mean Squared Error on Test Data: {mse.item()}")

            self.nn.train()  # Volver a modo entrenamiento
            return predictions_np


    """Gaussian Process"""
    class GaussianProcessROM(Modelo):
        def __init__(self):
            super().__init__()
            self.model = GaussianProcessRegressor()

        def train(self, X_train, y_train):
            self.model.fit(X_train, y_train)

        def evaluate(self, X_test, y_test):
            return self.model.predict(X_test)

    """RBF"""
    class RBFROM(Modelo):
        def __init__(self):
            super().__init__()
            self.model = None

        def train(self, X_train, y_train):
            self.model = RBFInterpolator(X_train, y_train)

        def evaluate(self, X_test, y_test):
            return self.model(X_test)

    """Response Surface"""
    class ResponseSurfaceROM(Modelo):
        """Modelo basado en ajuste polinómico (Response Surface)."""
        def __init__(self, degree=2):
            super().__init__()
            self.degree = degree
            self.model = None
            self.poly = PolynomialFeatures(degree=self.degree)

        def train(self, X_train, y_train):
            X_poly = self.poly.fit_transform(X_train)
            self.model = LinearRegression()
            self.model.fit(X_poly, y_train)

        def evaluate(self, X_test, y_test):
            X_poly = self.poly.transform(X_test)
            return self.model.predict(X_poly)


    class SVRROM(Modelo):
        """Modelo basado en Support Vector Regression (SVR)."""
        def __init__(self, kernel='rbf', C=1.0, epsilon=0.1):
            super().__init__()
            self.model = SVR(kernel=kernel, C=C, epsilon=epsilon)

        def train(self, X_train, y_train):
            self.model.fit(X_train, y_train)

        def evaluate(self, X_test, y_test):
            return self.model.predict(X_test)


# Ejemplo de uso
if __name__ == "__main__":
    # Generar datos de prueba
    X_train = np.random.rand(100, 2)
    y_train = np.sin(X_train[:, 0]) + np.cos(X_train[:, 1])

    X_test = np.random.rand(20, 2)
    y_test = np.sin(X_test[:, 0]) + np.cos(X_test[:, 1])

    # Gaussian Process ROM
    gp_model = idkROM.GaussianProcessROM()
    gp_model.train(X_train, y_train)
    print("GP Model MSE:", gp_model.score(X_test, y_test))

    # Neural Network ROM
    nn_model = idkROM.NeuralNetworkROM(input_dim=2)
    nn_model.train(X_train, y_train, num_epochs=200)
    print("NN Model MSE:", nn_model.score(X_test, y_test))

    # Response Surface ROM
    rs_model = idkROM.ResponseSurfaceROM(degree=2)
    rs_model.train(X_train, y_train)
    print("Response Surface Model MSE:", rs_model.score(X_test, y_test))

    # SVR ROM
    svr_model = idkROM.SVRROM(kernel='rbf', C=1.0, epsilon=0.1)
    svr_model.train(X_train, y_train)
    print("SVR Model MSE:", svr_model.score(X_test, y_test))
