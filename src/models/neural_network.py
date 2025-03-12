import os
import torch
import pandas as pd
from src.idkROM_v0 import idkROM
#from src.gui.visualizations import create_comparison_html

class NeuralNetworkROM(idkROM.Modelo):
    
      # Hereda de la clase base idkROM
    def __init__(self, input_dim=2, hidden_layers=2, neurons_per_layer=50, output_dim=1,
                 learning_rate=1e-3, activation_function='Tanh', optimizer_type='Adam', num_epochs=1000):
        super().__init__()

        # Aseguramos que los parámetros se pasen correctamente
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.optimizer_type = optimizer_type  # Aseguramos que optimizer_type se guarde aquí
        self.num_epochs = num_epochs

        # Crear la red neuronal
        self.nn = self.crear_red_neuronal(input_dim, hidden_layers, neurons_per_layer, output_dim)

        # Crear el optimizador
        self.optimizer = self.get_optimizer()


    def get_optimizer(self):
        """Devuelve el optimizador correspondiente según la elección del usuario."""
        if self.optimizer_type == 'Adam':
            return torch.optim.Adam(self.nn.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == 'SGD':
            return torch.optim.SGD(self.nn.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == 'RMSprop':
            return torch.optim.RMSprop(self.nn.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Optimizer {self.optimizer_type} not supported")
    
    
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
        """Entrena la red neuronal y evalúa cada 100 épocas."""
        # Convertir los datos a tensores
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

        loss_function = torch.nn.MSELoss()

        for epoch in range(num_epochs):
            for i in range(0, len(X_train_tensor), batch_size):
                # Obtener un lote de datos
                X_batch = X_train_tensor[i:i+batch_size]
                y_batch = y_train_tensor[i:i+batch_size]

                # Forward pass
                predictions = self.nn(X_batch)
                loss = loss_function(predictions, y_batch)

                # Backward pass y optimización
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Cada 100 épocas, evalúa el modelo
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item()}")

                # Establecer el modo evaluación y calcular la pérdida sobre todo el conjunto
                self.nn.eval()
                with torch.no_grad():
                    predictions_val = self.nn(X_train_tensor)
                    val_loss = loss_function(predictions_val, y_train_tensor)
                    print(f"Evaluación en datos de entrenamiento en la época {epoch}: Loss: {val_loss.item()}")
                self.nn.train()

        # Guardar el modelo
        #creamos la carpeta de salida
        idkROM.output_folder = os.path.join(os.getcwd(), 'results', f'{str(X_train.shape[0])}_samples')
        os.makedirs(idkROM.output_folder, exist_ok=True)
        model_path = os.path.join(idkROM.output_folder, 'nn_model.pth')
        torch.save(self.nn.state_dict(), model_path)


    def save_predictions(self, X_train, y_train, predictions, model_name='neural_network'):
        """Guarda las predicciones y los datos de entrenamiento en un archivo CSV."""
        import pandas as pd
        import numpy as np

        # Crear DataFrame con las entradas
        results = pd.DataFrame(X_train, columns=[f'Input_{i}' for i in range(X_train.shape[1])])
        
        # Verificar si y_train es multidimensional
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
        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.nn.eval()
        with torch.no_grad():
            predictions = self.nn(X_tensor).squeeze().cpu().numpy()
        self.nn.train()
        return predictions


    def evaluate(self, X_test, y_test):
        """Evalúa el modelo con los datos de prueba y devuelve las predicciones."""
        predictions = self.predict(X_test)
        # Guardamos las predicciones y datos de entrenamiento
        self.save_predictions(X_test, y_test, predictions)
        return predictions
    

    def score(self, X_test, y_test):
        """Calcula el error cuadrático medio entre las predicciones y los datos de prueba."""
        predictions = self.evaluate(X_test, y_test)
        mse = torch.nn.functional.mse_loss(torch.tensor(predictions, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)).item()
        print(f"Mean Squared Error on Test Data: {mse}")
        return mse
