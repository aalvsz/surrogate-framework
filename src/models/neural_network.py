import os
import torch
import pandas as pd
import numpy as np
import random
from src.idkrom import idkROM
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import StepLR

class NeuralNetworkROM(idkROM.Modelo):

    def __init__(self, rom_config, random_state):
        super().__init__(rom_config, random_state)

        self.input_dim = rom_config['input_dim']
        self.output_dim = rom_config['output_dim']
        self.hidden_layers = rom_config['hyperparams']['n_layers']
        self.neurons_per_layer = rom_config['hyperparams']['n_neurons']
        self.learning_rate = rom_config['hyperparams']['learning_rate']
        self.activation_function = rom_config['hyperparams']['activation']
        self.optimizer_name = rom_config['hyperparams']['optimizer']
        self.num_epochs = rom_config['hyperparams']['epochs']
        self.random_state = random_state


        self.model_name = rom_config['model_name']

        if rom_config['mode'] != 'best':
            # Crear la red neuronal
            self.nn = self.crear_red_neuronal(self.input_dim, self.hidden_layers, self.neurons_per_layer, self.output_dim)
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

    def search_best_hyperparams(self, X_train, y_train, X_val, y_val, iterations=10, batch_size=32, cv_folds=5):
        """
        Realiza una búsqueda aleatoria (randomized search) sobre combinaciones de hiperparámetros.
        Esta función se activa si los hiperparámetros vienen como listas en lugar de valores únicos.
        
        Args:
            X_train (pd.DataFrame): Datos de entrada de entrenamiento.
            y_train (pd.DataFrame): Datos objetivo de entrenamiento.
            X_val (pd.DataFrame): Datos de validación para evaluar la combinación.
            y_val (pd.DataFrame): Datos objetivo de validación.
            iterations (int): Número de iteraciones en la búsqueda.
            batch_size (int): Tamaño de lote para entrenamiento.
            cv_folds (int): Número de folds para validación cruzada.
            random_state (int, optional): Semilla para reproducibilidad.
        
        Returns:
            dict: Diccionario con la mejor combinación de hiperparámetros.
        """
        best_score = float('inf')
        best_params = None

        # Verificar que al menos uno de los hiperparámetros sea una lista
        if not (isinstance(self.hidden_layers, list) or isinstance(self.neurons_per_layer, list) or 
                isinstance(self.learning_rate, list) or isinstance(self.activation_function, list) or 
                isinstance(self.optimizer_name, list) or isinstance(self.num_epochs, list)):
            print("No se encontraron hiperparámetros en formato lista. search_best_hyperparams no se ejecutará.")
            return None
        
        # Guardar las listas originales
        orig_hidden_layers = self.hidden_layers if isinstance(self.hidden_layers, list) else [self.hidden_layers]
        orig_neurons = self.neurons_per_layer if isinstance(self.neurons_per_layer, list) else [self.neurons_per_layer]
        orig_lr = self.learning_rate if isinstance(self.learning_rate, list) else [self.learning_rate]
        orig_activation = self.activation_function if isinstance(self.activation_function, list) else [self.activation_function]
        orig_optimizer = self.optimizer_name if isinstance(self.optimizer_name, list) else [self.optimizer_name]
        orig_epochs = self.num_epochs if isinstance(self.num_epochs, list) else [self.num_epochs]

        for it in range(iterations):
            current_hidden_layers = random.choice(orig_hidden_layers)
            current_neurons = random.choice(orig_neurons)
            current_lr = random.choice(orig_lr)
            current_activation = random.choice(orig_activation)
            current_optimizer = random.choice(orig_optimizer)
            current_epochs = random.choice(orig_epochs)

            print(f"\nIteración {it+1}/{iterations} con parámetros:")
            print(f"  Hidden layers: {current_hidden_layers}, Neurons per layer: {current_neurons}")
            print(f"  Learning rate: {current_lr}, Activation: {current_activation}")
            print(f"  Optimizer: {current_optimizer}, Epochs: {current_epochs}")

            # Actualizar los atributos del objeto con la combinación actual
            self.hidden_layers = current_hidden_layers
            self.neurons_per_layer = current_neurons
            self.learning_rate = current_lr
            self.activation_function = current_activation
            self.optimizer_name = current_optimizer
            self.num_epochs = current_epochs

            # Reconstruir el modelo y el optimizador con los nuevos hiperparámetros
            self.nn = self.crear_red_neuronal(self.input_dim, self.hidden_layers, self.neurons_per_layer, self.output_dim)
            self.optimizer = self.get_optimizer()

            # Entrenar el modelo con la función train (la cual incluye validación cruzada)
            self.train(X_train, y_train, X_val, y_val, batch_size=batch_size, cv=False, cv_folds=cv_folds, random_state=self.random_state)
            # Suponemos que al finalizar self.val_losses[-1] contiene la pérdida de validación final
            current_score = self.val_losses[-1]
            print(f"  Pérdida de validación obtenida: {current_score:.6f}")

            # Si se obtiene una menor pérdida, actualizar la mejor combinación
            if current_score < best_score:
                best_score = current_score
                best_params = {
                    'hidden_layers': current_hidden_layers,
                    'neurons_per_layer': current_neurons,
                    'learning_rate': current_lr,
                    'activation_function': current_activation,
                    'optimizer_name': current_optimizer,
                    'num_epochs': current_epochs
                }
                print("  --> ¡Nueva mejor combinación encontrada!")

        print("\nBúsqueda completada. Mejor combinación:")
        print(best_params)

        # Actualizar los atributos del modelo con la mejor combinación
        if best_params is not None:
            self.hidden_layers = best_params['hidden_layers']
            self.neurons_per_layer = best_params['neurons_per_layer']
            self.learning_rate = best_params['learning_rate']
            self.activation_function = best_params['activation_function']
            self.optimizer_name = best_params['optimizer_name']
            self.num_epochs = best_params['num_epochs']
            # Reconstruir el modelo final con los mejores hiperparámetros
            self.nn = self.crear_red_neuronal(self.input_dim, self.hidden_layers, self.neurons_per_layer, self.output_dim)
            self.optimizer = self.get_optimizer()
        return best_params


    def train(self, X_train, y_train, X_val, y_val, batch_size=32, cv=True, cv_folds=5, patience=1000):
        """
        Trains the neural network using k-fold cross-validation and saves training and validation losses.

        Args:
            X_train (pd.DataFrame): Training input data.
            y_train (pd.DataFrame): Training target data.
            X_val (pd.DataFrame): Validation input data (used for final validation after CV).
            y_val (pd.DataFrame): Validation target data (used for final validation after CV).
            batch_size (int): Size of mini-batches for training. Default is 32.
            cv_folds (int): Number of folds for cross-validation. Default is 5.
            patience (int): Number of epochs to wait for improvement in validation loss before stopping. Default is 10.
        """

        # Almacenar datos para el reporte
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val

        X_train_np = X_train.to_numpy()
        y_train_np = y_train.to_numpy()

        if cv:
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            fold_train_losses = []
            fold_val_losses = []

            print(f"Iniciando Cross-Validation con {cv_folds} folds.")

            for fold, (train_index, val_index) in enumerate(kf.split(X_train_np, y_train_np)):
                print(f"Fold {fold+1}/{cv_folds}")
                X_train_fold = pd.DataFrame(X_train_np[train_index], columns=X_train.columns)
                y_train_fold = pd.DataFrame(y_train_np[train_index], columns=y_train.columns)
                X_val_fold = pd.DataFrame(X_train_np[val_index], columns=X_train.columns)
                y_val_fold = pd.DataFrame(y_train_np[val_index], columns=y_train.columns)

                # Convertir datos a tensores para el fold
                X_train_tensor = torch.tensor(X_train_fold.to_numpy(), dtype=torch.float32)
                y_train_tensor = torch.tensor(y_train_fold.to_numpy(), dtype=torch.float32)
                X_val_tensor = torch.tensor(X_val_fold.to_numpy(), dtype=torch.float32)
                y_val_tensor = torch.tensor(y_val_fold.to_numpy(), dtype=torch.float32)

                # Reiniciar el modelo y el optimizador para cada fold
                model_fold = self.crear_red_neuronal(self.input_dim, self.hidden_layers, self.neurons_per_layer, self.output_dim)
                optimizer_fold = self.get_optimizer()
                loss_function = torch.nn.MSELoss()
                scheduler_fold = StepLR(optimizer_fold, step_size=500, gamma=0.1) # Initialize scheduler for the fold

                train_losses_fold = []
                val_losses_fold = []

                for epoch in range(self.num_epochs):
                    model_fold.train()
                    total_loss = 0.0
                    total_examples = 0
                    for i in range(0, len(X_train_tensor), batch_size):
                        X_batch = X_train_tensor[i:i+batch_size]
                        y_batch = y_train_tensor[i:i+batch_size]
                        batch_size_actual = X_batch.size(0)

                        # Forward pass
                        predictions = model_fold(X_batch)
                        loss = loss_function(predictions, y_batch)
                        # Acumular pérdida ponderada
                        total_loss += loss.item() * batch_size_actual
                        total_examples += batch_size_actual

                        # Backward pass y optimización
                        optimizer_fold.zero_grad()
                        loss.backward()
                        optimizer_fold.step()

                    # Guardamos la pérdida del último batch como representativa de la época
                    epoch_loss = total_loss / total_examples
                    train_losses_fold.append(epoch_loss)

                    # Calcular la pérdida de validación
                    model_fold.eval()
                    with torch.no_grad():
                        predictions_val = model_fold(X_val_tensor)
                        val_loss = loss_function(predictions_val, y_val_tensor)
                    val_losses_fold.append(val_loss.item())

                    if epoch % 100 == 0:
                        print(f"  Fold {fold+1}, Epoch {epoch}/{self.num_epochs}, Training Loss: {epoch_loss:.6f}, Validation Loss: {val_loss.item():.6f}")

                    scheduler_fold.step() # Step the scheduler at the end of each epoch

                fold_train_losses.append(train_losses_fold)
                fold_val_losses.append(val_losses_fold)
                print(f"  Fold {fold+1} finalizado. Validation Loss: {val_losses_fold[-1]:.6f}")

            # Calcular el promedio de las pérdidas de validación por fold
            avg_val_losses = np.mean([fold_val_losses[i][-1] for i in range(cv_folds)])
            print(f"\nPromedio de la pérdida de validación en Cross-Validation: {avg_val_losses:.6f}")

        print("\nEntrenando el modelo final con el conjunto de entrenamiento completo hasta convergencia (o máximo de épocas).")
        # Entrenar el modelo final con todos los datos de entrenamiento hasta convergencia
        self.nn = self.crear_red_neuronal(self.input_dim, self.hidden_layers, self.neurons_per_layer, self.output_dim)
        self.optimizer = self.get_optimizer()
        loss_function = torch.nn.MSELoss()
        scheduler = StepLR(self.optimizer, step_size=500, gamma=0.1) # Initialize scheduler for the final training
        self.train_losses = []
        self.val_losses = []
        X_train_tensor_full = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
        y_train_tensor_full = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
        X_val_tensor_final = torch.tensor(X_val.to_numpy(), dtype=torch.float32)
        y_val_tensor_final = torch.tensor(y_val.to_numpy(), dtype=torch.float32)

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(self.num_epochs): # Keep the loop with the maximum number of epochs
            self.nn.train()
            total_loss = 0.0
            total_examples = 0
            for i in range(0, len(X_train_tensor_full), batch_size):
                X_batch = X_train_tensor_full[i:i+batch_size]
                y_batch = y_train_tensor_full[i:i+batch_size]
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

            epoch_loss = total_loss / total_examples
            self.train_losses.append(epoch_loss)

            # Validación final
            self.nn.eval()
            with torch.no_grad():
                predictions_val = self.nn(X_val_tensor_final)
                val_loss = loss_function(predictions_val, y_val_tensor_final)
            self.val_losses.append(val_loss.item())

            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{self.num_epochs}, Training Loss: {epoch_loss:.6f}, Validation Loss: {val_loss.item():.6f}")

            scheduler.step() # Step the scheduler at the end of each epoch

            # Implement Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = self.nn.state_dict() # Save the state of the best model
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
                    self.nn.load_state_dict(best_model_state) # Load the state of the best model
                    break

        # Guardar el modelo entrenado (will now be the best model if early stopping occurred)
        output_folder = os.path.join(os.getcwd(), 'results', self.model_name)
        os.makedirs(output_folder, exist_ok=True)
        model_path = os.path.join(output_folder, 'neural_network_model.pth')
        torch.save(self.nn.state_dict(), model_path)
        print(f"Modelo guardado en: {model_path}")


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


    def calculate_bic(self, y_true, y_pred):
        """
        Calculates the Bayesian Information Criterion (BIC).

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
        num_params = sum(p.numel() for p in self.nn.parameters() if p.requires_grad)
        bic = n * np.log(sse) + num_params * np.log(n)
        return bic

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
        print("Verificación de que y_test y y_pred tengan la misma forma:")
        print("Forma de y_test:", y_test.shape)
        print("Forma de y_pred:", y_pred.shape)

        # Convertir a numpy arrays
        y_test_np = y_test.to_numpy()
        y_pred_np = np.array(y_pred)

        # Calcular MSE en la escala normalizada
        mse_scaled = np.mean((y_pred_np - y_test_np)**2)
        mse_percentage = (mse_scaled / np.mean(np.abs(y_test_np))) * 100  # MSE en porcentaje
        print(f"MSE en escala normalizada: {mse_scaled:.4f}")
        print(f"MSE en porcentaje: {mse_percentage:.2f}%")

        # Calcular MAE normalizado
        mae_scaled = np.mean(np.abs(y_pred_np - y_test_np))
        mae_percentage = (mae_scaled / np.mean(np.abs(y_test_np))) * 100  # MAE en porcentaje
        print(f"MAE en escala normalizada: {mae_scaled:.4f}")
        print(f"MAE en porcentaje: {mae_percentage:.2f}%")

        # Calcular Mean Absolute Percentage Error (MAPE)
        # Añadir una pequeña constante para evitar la división por cero
        epsilon = 1e-10
        mape = np.mean(np.abs((y_test_np - y_pred_np) / (y_test_np + epsilon))) * 100
        print(f"MAPE: {mape:.2f}%")

        print(f"Diferencia entre MSE y MAE = {abs(mse_percentage-mae_percentage):.2f}%")

        # Calcular BIC
        bic_value = self.calculate_bic(y_test, y_pred)
        print(f"Valor de BIC: {bic_value:.2f}")

        # Si se proporciona el scaler, deshacer la normalización para comparar en la escala original.
        """if output_scaler is not None:
            # Asegurarse de que las dimensiones sean compatibles para inverse_transform
            y_pred_original = output_scaler.inverse_transform(y_pred.reshape(-1, y_test.shape[1]))
            y_test_original = output_scaler.inverse_transform(y_test.to_numpy())
            mse_original = torch.nn.functional.mse_loss(torch.tensor(y_pred_original, dtype=torch.float32),
                                                                     torch.tensor(y_test_original, dtype=torch.float32)).item()
            print(f"MSE en escala original: {mse_original}")

            # Calcular MAE en la escala original
            mae_original = np.mean(np.abs(y_pred_original - y_test_original))
            mae_original_percentage = (mae_original / np.mean(np.abs(y_test_original))) * 100
            print(f"MAE en escala original: {mae_original}")
            print(f"MAE en escala original (porcentaje): {mae_original_percentage:.2f}%")"""

        # Calcular la diferencia entre el training loss y el validation loss en porcentaje
        if len(self.train_losses) > 0 and len(self.val_losses) > 0:
            last_train_loss = self.train_losses[-1]
            last_val_loss = self.val_losses[-1]
            loss_difference_percentage = ((last_train_loss - last_val_loss) / last_train_loss) * 100
            print(f"Diferencia entre Training Loss y Validation Loss: {loss_difference_percentage:.2f}%")

        print(f"Este es el diccionario que se come el modelo: {self.rom_config}")
        

        import matplotlib.pyplot as plt
        """
        Generates and displays visualization plots based on the model, test data, and predictions.

        Args:
            model: The trained model object (either SVRROM or NeuralNetworkROM).
            X_test (pd.DataFrame or np.ndarray): Test input data.
            y_test (pd.DataFrame or np.ndarray): True labels for the test data.
            y_pred (np.ndarray): Predictions made by the model on the test data.
        """
        plt.figure(figsize=(15, 12))

        # 1. Predicciones vs. Valores Reales
        plt.subplot(2, 3, 1)
        if isinstance(y_test, pd.DataFrame):
            y_test_np = y_test.values
        else:
            y_test_np = y_test
        plt.scatter(y_test_np.flatten(), y_pred.flatten())
        plt.xlabel("Valores Reales")
        plt.ylabel("Predicciones")
        plt.title("Predicciones vs. Valores Reales")
        plt.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'k--', lw=2) # Línea de referencia
        plt.grid(True)

        # 2. Distribución de Errores
        plt.subplot(2, 3, 2)
        errors = y_test_np.flatten() - y_pred.flatten()
        plt.hist(errors, bins=50, edgecolor='black')
        plt.xlabel("Error")
        plt.ylabel("Frecuencia")
        plt.title("Distribución de Errores")
        plt.grid(True)

        # 3. Errores vs. Predicciones
        plt.subplot(2, 3, 5)
        plt.scatter(y_pred.flatten(), errors)
        plt.xlabel("Predicciones")
        plt.ylabel("Error")
        plt.title("Errores vs. Predicciones")
        plt.axhline(y=0, color='r', linestyle='--') # Línea de referencia en cero
        plt.grid(True)

        # 4. Training and Validation Loss Curves (solo para NeuralNetworkROM)
        if hasattr(self, 'train_losses') and hasattr(self, 'val_losses'):
            plt.subplot(2, 3, 4)
            epochs = range(1, len(self.train_losses) + 1)
            plt.plot(epochs, self.train_losses, label='Pérdida de Entrenamiento')
            plt.plot(epochs, self.val_losses, label='Pérdida de Validación')
            plt.xlabel("Épocas")
            plt.ylabel("Pérdida")
            plt.title("Curvas de Pérdida de Entrenamiento y Validación")
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        return 0
