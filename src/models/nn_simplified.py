import os
import torch
import numpy as np
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import StepLR
from new_main import idkROM

class NeuralNetworkROM(idkROM.Modelo):
    """
    Modelo de Red Neuronal para optimización con PyTorch.
    """

    def __init__(self, rom_config, random_state):
        super().__init__(rom_config, random_state)

        # Extraer parámetros básicos y de hiperparámetros
        self.input_dim = rom_config['input_dim']
        self.output_dim = rom_config['output_dim']
        self.hidden_layers = rom_config['hyperparams']['n_layers']
        self.neurons_per_layer = rom_config['hyperparams']['n_neurons']
        self.learning_rate = rom_config['hyperparams']['learning_rate']
        self.activation_function = rom_config['hyperparams']['activation']
        self.dropout = rom_config['hyperparams']['dropout_rate']
        self.optimizer_name = rom_config['hyperparams']['optimizer']
        self.num_epochs = rom_config['hyperparams']['epochs']
        self.random_state = random_state
        self.model_name = rom_config['model_name']

        # Variables para reportes y almacenamiento de datos
        self.train_losses = []
        self.val_losses = []
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

        # Agregar esto en el constructor
        self.nn = self._crear_red_neuronal(self.input_dim, self.hidden_layers,
                                        self.neurons_per_layer, self.output_dim)
        self.optimizer = self._get_optimizer()


    def _get_optimizer(self):
        """
        Devuelve el optimizador para el modelo actual.
        """
        optimizers = {
            'Adam': torch.optim.Adam(self.nn.parameters(), lr=self.learning_rate),
            'SGD': torch.optim.SGD(self.nn.parameters(), lr=self.learning_rate),
            'RMSprop': torch.optim.RMSprop(self.nn.parameters(), lr=self.learning_rate),
        }
        
        try:
            return optimizers[self.optimizer_name]
        except KeyError:
            raise ValueError(f"Optimizer {self.optimizer_name} not supported")


    def _get_activation_function(self):
        """
        Devuelve la función de activación especificada.
        """
        activations = {
            'Tanh': torch.nn.Tanh(),
            'ReLU': torch.nn.ReLU(),
            'Sigmoid': torch.nn.Sigmoid(),
        }
        
        try:
            return activations[self.activation_function]
        except KeyError:
            raise ValueError(f"Función de activación desconocida: {self.activation_function}")


    def _crear_red_neuronal(self, input_size, hidden_layers, neurons_per_layer, output_size):
        """
        Construye una red neuronal feedforward con capas ocultas y dropout.
        """
        layers = []
        # Capa de entrada
        layers.append(torch.nn.Linear(input_size, neurons_per_layer))
        layers.append(self._get_activation_function())
        layers.append(torch.nn.Dropout(self.dropout))
        # Capas ocultas
        for _ in range(hidden_layers - 1):
            layers.append(torch.nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(self._get_activation_function())
            layers.append(torch.nn.Dropout(self.dropout))
        # Capa de salida
        layers.append(torch.nn.Linear(neurons_per_layer, output_size))
        return torch.nn.Sequential(*layers)


    def _train_epoch(self, model, optimizer, X_tensor, y_tensor, batch_size):
        """
        Ejecuta un ciclo de entrenamiento (una época) sobre los datos en mini-batches.
        """
        model.train()
        total_loss = 0.0
        total_examples = 0
        loss_function = torch.nn.MSELoss()
        for i in range(0, len(X_tensor), batch_size):
            X_batch = X_tensor[i:i + batch_size]
            y_batch = y_tensor[i:i + batch_size]
            batch_actual = X_batch.size(0)

            predictions = model(X_batch)
            loss = loss_function(predictions, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_actual
            total_examples += batch_actual

        return total_loss / total_examples


    def train(self, X_train, y_train, X_val, y_val, batch_size=32, cv=True, cv_folds=5, patience=1000):
        """
        Entrena la red neuronal utilizando validación cruzada (si se activa)
        y luego entrena el modelo final con todos los datos de entrenamiento.

        Args:
            X_train (pd.DataFrame): Datos de entrada para entrenamiento.
            y_train (pd.DataFrame): Datos objetivo de entrenamiento.
            X_val (pd.DataFrame): Datos de entrada de validación.
            y_val (pd.DataFrame): Datos objetivo de validación.
            batch_size (int): Tamaño de lote para entrenamiento.
            cv (bool): Si se activa la validación cruzada.
            cv_folds (int): Número de folds para CV.
            patience (int): Épocas de espera para Early Stopping.
        """
        # Guardar datos para reportes
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val

        X_train_np = X_train.to_numpy()
        y_train_np = y_train.to_numpy()

        # Validación cruzada
        if cv:
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            fold_train_losses = []
            fold_val_losses = []
            print(f"Iniciando Cross-Validation con {cv_folds} folds.")

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_np, y_train_np)):
                print(f"Fold {fold+1}/{cv_folds}")
                # Preparar datos para este fold
                X_train_fold = torch.tensor(X_train_np[train_idx], dtype=torch.float32)
                y_train_fold = torch.tensor(y_train_np[train_idx], dtype=torch.float32)
                X_val_fold = torch.tensor(X_train_np[val_idx], dtype=torch.float32)
                y_val_fold = torch.tensor(y_train_np[val_idx], dtype=torch.float32)

                # Reiniciar modelo y optimizador para el fold
                model_fold = self._crear_red_neuronal(self.input_dim, self.hidden_layers,
                                                       self.neurons_per_layer, self.output_dim)
                optimizer_fold = self._get_optimizer()
                scheduler_fold = StepLR(optimizer_fold, step_size=500, gamma=0.1)
                loss_function = torch.nn.MSELoss()

                train_losses_fold = []
                val_losses_fold = []

                for epoch in range(self.num_epochs):
                    epoch_loss = self._train_epoch(model_fold, optimizer_fold, X_train_fold, y_train_fold, batch_size)
                    train_losses_fold.append(epoch_loss)

                    model_fold.eval()
                    with torch.no_grad():
                        val_loss = loss_function(model_fold(X_val_fold), y_val_fold).item()
                    val_losses_fold.append(val_loss)

                    if epoch % 100 == 0:
                        print(f"  Fold {fold+1}, Epoch {epoch}/{self.num_epochs}, Training Loss: {epoch_loss:.6f}, Validation Loss: {val_loss:.6f}")
                    scheduler_fold.step()

                fold_train_losses.append(train_losses_fold)
                fold_val_losses.append(val_losses_fold)
                print(f"  Fold {fold+1} finalizado. Última Validation Loss: {val_losses_fold[-1]:.6f}")

            avg_val_loss = np.mean([fold_val_losses[i][-1] for i in range(cv_folds)])
            print(f"\nPromedio de la pérdida de validación en CV: {avg_val_loss:.6f}")

        print("\nEntrenando el modelo final con todo el conjunto de entrenamiento.")
        # Entrenamiento final con todos los datos
        self.nn = self._crear_red_neuronal(self.input_dim, self.hidden_layers,
                                           self.neurons_per_layer, self.output_dim)
        self.optimizer = self._get_optimizer()
        scheduler = StepLR(self.optimizer, step_size=500, gamma=0.1)
        loss_function = torch.nn.MSELoss()
        self.train_losses = []
        self.val_losses = []

        X_train_tensor_full = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
        y_train_tensor_full = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
        X_val_tensor_final = torch.tensor(X_val.to_numpy(), dtype=torch.float32)
        y_val_tensor_final = torch.tensor(y_val.to_numpy(), dtype=torch.float32)

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(self.num_epochs):
            epoch_loss = self._train_epoch(self.nn, self.optimizer, X_train_tensor_full, y_train_tensor_full, batch_size)
            self.train_losses.append(epoch_loss)

            self.nn.eval()
            with torch.no_grad():
                val_loss = loss_function(self.nn(X_val_tensor_final), y_val_tensor_final).item()
            self.val_losses.append(val_loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{self.num_epochs}, Training Loss: {epoch_loss:.6f}, Validation Loss: {val_loss:.6f}")
            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = self.nn.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                    self.nn.load_state_dict(best_model_state)
                    break

        # Guardar el modelo final
        output_folder = os.path.join(os.getcwd(), 'results', self.model_name)
        os.makedirs(output_folder, exist_ok=True)
        model_path = os.path.join(output_folder, 'neural_network_model.pth')
        torch.save(self.nn.state_dict(), model_path)
        print(f"Modelo guardado en: {model_path}")


    def predict(self, X_test):
        """
        Realiza predicciones usando el modelo entrenado.

        Args:
            X_test (pd.DataFrame): Datos de entrada de prueba.

        Returns:
            np.ndarray: Predicciones del modelo.
        """
        X_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
        self.nn.eval()
        with torch.no_grad():
            y_pred = self.nn(X_tensor)
        # Si la salida tiene dimensión extra y es de tamaño 1, se quita la dimensión
        if y_pred.ndim > 1 and y_pred.shape[1] == 1:
                y_pred = y_pred.squeeze(1)
        y_pred = y_pred.cpu().numpy()
        self.nn.train()  # Volver al modo entrenamiento
        return y_pred
    

    def idk_run(model, X_train, y_train, X_val, y_val, X_test, batch_size=32, cv=True, cv_folds=5, patience=1000):
        """
        Función que entrena un modelo de red neuronal y realiza predicciones.

        Args:
            model (NeuralNetworkROM): Instancia del modelo a entrenar.
            X_train (pd.DataFrame): Datos de entrada para entrenamiento.
            y_train (pd.DataFrame): Datos objetivo de entrenamiento.
            X_val (pd.DataFrame): Datos de entrada de validación.
            y_val (pd.DataFrame): Datos objetivo de validación.
            X_test (pd.DataFrame): Datos de entrada de prueba.
            y_test (pd.DataFrame): Datos objetivo de prueba (para comparación, si es necesario).
            batch_size (int): Tamaño de lote para entrenamiento.
            cv (bool): Si se activa la validación cruzada.
            cv_folds (int): Número de folds para CV.
            patience (int): Épocas de espera para Early Stopping.

        Returns:
            np.ndarray: Predicciones del modelo en X_test.
        """
        # Entrenar el modelo
        model.train(X_train, y_train, X_val, y_val, batch_size=batch_size, cv=cv, cv_folds=cv_folds, patience=patience)
        
        # Realizar predicciones en el conjunto de prueba
        y_pred = model.predict(X_test)
        
        return y_pred

