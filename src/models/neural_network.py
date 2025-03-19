import os
import torch
import pandas as pd
import numpy as np
from src.idkrom import idkROM
from src.visualization.metrics import ModelReportGenerator
from sklearn.model_selection import train_test_split

class NeuralNetworkROM(idkROM.Modelo):
    """
    A class that implements a neural network model for ROM (Reduced Order Model).
    
    Args:
        input_dim (int): Dimension of input features. Default is 2.
        output_dim (int): Dimension of output features. Default is 1.
        hidden_layers (int): Number of hidden layers. Default is 2.
        neurons_per_layer (int): Number of neurons in each hidden layer. Default is 50.
        learning_rate (float): Learning rate for optimizer. Default is 1e-3.
        activation_function (str): Activation function to use ('Tanh', 'ReLU', or 'Sigmoid'). Default is 'Tanh'.
        optimizer (str): Optimizer to use ('Adam', 'SGD', or 'RMSprop'). Default is 'Adam'.
        num_epochs (int): Number of training epochs. Default is 1000.
    """
    def __init__(self, input_dim=7, output_dim=18, hidden_layers=2, neurons_per_layer=20, 
                 learning_rate=1e-3, activation_function='Tanh', optimizer='Adam', num_epochs=1000):
        super().__init__()
        
        # Guardar parámetros
        # self.hyperparams = hyperparams (diccionario)
        # self.hyperparams['input_dim'] = input_dim
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.optimizer_name = optimizer  # Store the optimizer name
        self.num_epochs = num_epochs

        # Crear la red neuronal
        self.nn = self.crear_red_neuronal(input_dim, hidden_layers, neurons_per_layer, output_dim)
        # Crear el optimizador
        self.optimizer = self.get_optimizer()

        # Variables para el reporte (se llenarán durante el entrenamiento)
        self.train_losses = []
        self.val_losses = []
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_val = None
        self.y_val = None

        self.model_name = 'neural_network'

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Args:
            deep (bool): If True, will return the parameters for this estimator and contained subobjects that are estimators.

        Returns:
            dict: Parameter names mapped to their values.
        """
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_layers': self.hidden_layers,
            'neurons_per_layer': self.neurons_per_layer,
            'learning_rate': self.learning_rate,
            'activation_function': self.activation_function,
            'optimizer': self.optimizer_name,
            'num_epochs': self.num_epochs
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
        self.nn = self.crear_red_neuronal(self.input_dim, self.hidden_layers, self.neurons_per_layer, self.output_dim)
        self.optimizer = self.get_optimizer()  # Update the optimizer
        return self

    def fit(self, X_train, y_train):
        """
        Fits the neural network model to the training data.
        
        Args:
            X_train (pd.DataFrame): Training input data.
            y_train (pd.DataFrame): Training target data.
        """
        self.train(X_train, y_train, X_train, y_train)

    def get_optimizer(self):
        """
        Returns the appropriate optimizer based on user selection.
        
        Returns:
            torch.optim: The selected optimizer instance with the configured learning rate.
        
        Raises:
            ValueError: If the specified optimizer is not supported.
        """
        if self.optimizer_name == 'Adam':
            return torch.optim.Adam(self.nn.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'SGD':
            return torch.optim.SGD(self.nn.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'RMSprop':
            return torch.optim.RMSprop(self.nn.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Optimizer {self.optimizer_name} not supported")


    def crear_red_neuronal(self, input_size, hidden_layers, neurons_per_layer, output_size):
        """
        Creates a feedforward neural network with dropout layers to reduce overfitting.
        
        Args:
            input_size (int): Number of input features.
            hidden_layers (int): Number of hidden layers.
            neurons_per_layer (int): Number of neurons in each hidden layer.
            output_size (int): Number of output features.
        
        Returns:
            torch.nn.Sequential: The constructed neural network model.
        """
        layers = []
        # Capa de entrada
        layers.append(torch.nn.Linear(input_size, neurons_per_layer))
        layers.append(self.get_activation_function())
        layers.append(torch.nn.Dropout(0.2))  # Dropout con p=0.2
        # Capas ocultas
        for _ in range(hidden_layers - 1):
            layers.append(torch.nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(self.get_activation_function())
            layers.append(torch.nn.Dropout(0.2))  # Dropout en cada capa oculta
        # Capa de salida
        layers.append(torch.nn.Linear(neurons_per_layer, output_size))
        return torch.nn.Sequential(*layers)



    def get_activation_function(self):
        """
        Returns the specified activation function.
        
        Returns:
            torch.nn.Module: The selected activation function.
        
        Raises:
            ValueError: If the specified activation function is not supported.
        """
        if self.activation_function == 'Tanh':
            return torch.nn.Tanh()
        elif self.activation_function == 'ReLU':
            return torch.nn.ReLU()
        elif self.activation_function == 'Sigmoid':
            return torch.nn.Sigmoid()
        else:
            raise ValueError(f"Función de activación desconocida: {self.activation_function}")


    def train(self, X_train, y_train, X_val, y_val, batch_size=32):
        """
        Trains the neural network and saves training and validation losses.
        
        Args:
            X_train (pd.DataFrame): Training input data.
            y_train (pd.DataFrame): Training target data.
            X_val (pd.DataFrame): Validation input data.
            y_val (pd.DataFrame): Validation target data.
            batch_size (int): Size of mini-batches for training. Default is 32.
        """

        # Almacenar datos para el reporte
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val

        # Convertir datos a tensores
        X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val.to_numpy(), dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32)

        loss_function = torch.nn.MSELoss()

        for epoch in range(self.num_epochs):
            self.nn.train()
            total_loss = 0.0
            total_examples = 0
            for i in range(0, len(X_train_tensor), batch_size):
                X_batch = X_train_tensor[i:i+batch_size]
                y_batch = y_train_tensor[i:i+batch_size]
                batch_size_actual = X_batch.size(0)

                # Forward pass
                predictions = self.nn(X_batch)
                loss = loss_function(predictions, y_batch)
                # Acumular pérdida ponderada
                total_loss += loss.item() * batch_size_actual
                total_examples += batch_size_actual

                # Backward pass y optimización
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


            # Guardamos la pérdida del último batch como representativa de la época
            """Training loss: Se calcula durante el proceso de entrenamiento usando batches de datos,
            se actualizan los pesos despues de cada batch y mide como el modelo se comporta mientra aprende
            se pone el modelo en modo self.nn.train() y se activarían dropout y actualización de gradientes."""
            epoch_loss = total_loss / total_examples
            self.train_losses.append(epoch_loss)

            # Cada 100 épocas, calcular la pérdida de validación y mostrar información
            """Validation loss: Se calcula al final de cada 100 épocas,
            se pone el modelo en modo self.nn.eval() y se desactivan el dropout y la normalización por lotes
            se calcula sobre TODOS los datos, no sobre batches, sin actualizar los pesos."""
            if epoch % 100 == 0:
                self.nn.eval()
                with torch.no_grad():
                    predictions_val = self.nn(X_val_tensor)
                    val_loss = loss_function(predictions_val, y_val_tensor)
                self.val_losses.append(val_loss.item())
                print(f"Epoch {epoch}/{self.num_epochs}, Training Loss: {epoch_loss:.6f}, Validation Loss: {val_loss.item():.6f}")

        # Guardar el modelo entrenado
        output_folder = os.path.join(os.getcwd(), 'results', self.model_name)
        os.makedirs(output_folder, exist_ok=True)
        model_path = os.path.join(output_folder, 'neural_network_model.pth')
        torch.save(self.nn.state_dict(), model_path)
        print(f"Modelo guardado en: {model_path}")


    def save_predictions(self, X_train, y_train, predictions):
        """
        Saves the predictions and training data to a CSV file.
        
        Args:
            X_train (pd.DataFrame): Input features.
            y_train (pd.DataFrame): True target values.
            predictions (np.ndarray): Model predictions.
        """
        import numpy as np

        results = pd.DataFrame(X_train, columns=[f'Input_{i}' for i in range(X_train.shape[1])])
        y_train = np.array(y_train)
        predictions = np.array(predictions)

        if y_train.ndim > 1 and y_train.shape[1] > 1:
            for col in range(y_train.shape[1]):
                results[f'True_Output_{col}'] = y_train[:, col]
                results[f'Predicted_Output_{col}'] = predictions[:, col]
        else:
            results['True_Output'] = y_train.squeeze()
            results['Predicted_Output'] = predictions.squeeze()

        output_path = os.path.join(os.getcwd(), 'results', self.model_name, f'{self.model_name}_predictions.csv')
        results.to_csv(output_path, index=False)
        print(f"Predicciones guardadas en: {output_path}")


    def predict(self, X_test):
        """
        Makes predictions using the trained model on the test input subset.
        
        Args:
            X_test (pd.DataFrame): Test input data.
        
        Returns:
            np.ndarray: Model predictions.
        """
        X_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
        self.nn.eval()
        with torch.no_grad():
            y_pred = self.nn(X_tensor)
        # Verificar las dimensiones y aplicar squeeze de manera segura:
        if y_pred.ndim > 1:
            # Si la segunda dimensión es 1, se elimina; en caso contrario, se mantiene la estructura.
            if y_pred.shape[1] == 1:
                y_pred = y_pred.squeeze(1)
        y_pred = y_pred.cpu().numpy()
        self.nn.train()
        return y_pred



    def evaluate(self, X_test:list, y_test:list, y_pred:list, output_scaler=None):
        """
        Evaluates the model using the test data and saves the predictions to a CSV file.

        Args:
            X_test (array-like): Test features.
            y_test (array-like): True labels for the test data.
            y_pred (array-like): Predictions made by the model on the test data.
            output_scaler (sklearn.preprocessing.StandardScaler, optional): Scaler used for output normalization.
        
        Returns:
            int: Returns 0 if evaluation completes successfully.
        """
        """Evalúa el modelo con los datos de prueba y guarda las predicciones en CSV."""


        print("Verificación de que y_test y y_pred tengan la misma forma:")
        print("Forma de y_test:", y_test.shape)
        print("Forma de y_pred:", y_pred.shape)

        # Convertir a tensores en escala normalizada
        y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32)
        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)
        
        # Calcular MSE en la escala actual
        mse_scaled = torch.nn.functional.mse_loss(y_pred_tensor, y_test_tensor).item()
        print(f"MSE en escala normalizada: {mse_scaled}")
        
        # Si se proporciona el scaler, se deshace la normalización para comparar en la escala original.
        if output_scaler is not None:
            # Asegurarse de que las dimensiones sean compatibles para inverse_transform
            y_pred_original = output_scaler.inverse_transform(y_pred.reshape(-1, y_test.shape[1]))
            y_test_original = output_scaler.inverse_transform(y_test.to_numpy())
            mse_original = torch.nn.functional.mse_loss(torch.tensor(y_pred_original, dtype=torch.float32),
                                                        torch.tensor(y_test_original, dtype=torch.float32)).item()
            print(f"MSE en escala original: {mse_original}")


        # Usar self (el modelo actual) y los datos almacenados en la instancia para el reporte
        self.save_predictions(X_test, y_test, y_pred)

        report_generator = ModelReportGenerator(self, self.train_losses, self.val_losses,
                                                self.X_train, self.y_train, X_test, y_test,
                                                self.model_name)
        report_generator.save_model_and_metrics()

        return 0
