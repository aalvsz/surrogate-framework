import os
import torch
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.optim as optim
from idkrom.model import idkROM

import torch
import torch.nn as nn

class FeedforwardNet(nn.Module):
    """
    Red neuronal feedforward configurable para regresión.

    Args:
        input_dim (int): Dimensión de entrada.
        output_dim (int): Dimensión de salida.
        n_layers (int): Número de capas ocultas.
        n_neurons (int): Neuronas por capa oculta.
        activation (str): Función de activación para capas ocultas.
        dropout_rate (float): Tasa de dropout.
        output_activation (str, opcional): Activación de la capa de salida.
    """
    def __init__(self, input_dim, output_dim, n_layers, n_neurons, activation, dropout_rate, output_activation=None):
        """
        Inicializa la red neuronal feedforward.

        Args:
            input_dim (int): Dimensión de entrada.
            output_dim (int): Dimensión de salida.
            n_layers (int): Número de capas ocultas.
            n_neurons (int): Neuronas por capa oculta.
            activation (str): Función de activación para capas ocultas.
            dropout_rate (float): Tasa de dropout.
            output_activation (str, opcional): Activación de la capa de salida.
        """
        super().__init__()

        activations = {
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'leakyrelu': nn.LeakyReLU,
        }

        layers = []
        current_dim = input_dim

        # Capas ocultas
        for i in range(n_layers):
            layers.append(nn.Linear(current_dim, n_neurons))
            layers.append(activations[activation.lower()]())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = n_neurons

        # Capa de salida
        layers.append(nn.Linear(current_dim, output_dim))

        # Activación de salida (opcional)
        if output_activation and output_activation.lower() in activations:
            layers.append(activations[output_activation.lower()]())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Propaga la entrada a través de la red.

        Args:
            x (torch.Tensor): Tensor de entrada.

        Returns:
            torch.Tensor: Salida de la red.
        """
        return self.net(x)



class NeuralNetworkROM(idkROM.Modelo):
    """
    Modelo de Red Neuronal Feedforward para regresión usando PyTorch.
    Implementa entrenamiento con Validación Cruzada opcional y Early Stopping.
    """

    def __init__(self, rom_config, random_state):
        """
        Inicializa el modelo de Red Neuronal.

        Args:
            rom_config (dict): Diccionario de configuración que contiene:
                - input_dim (int): Dimensión de entrada.
                - output_dim (int): Dimensión de salida.
                - hyperparams (dict): Diccionario con hiperparámetros:
                    - n_layers (int): Número de capas ocultas.
                    - n_neurons (int): Neuronas por capa oculta.
                    - learning_rate (float): Tasa de aprendizaje inicial.
                    - lr_step (int): Épocas entre reducciones de LR.
                    - lr_decrease_rate (float): Factor de reducción de LR (gamma).
                    - activation (str): Nombre de la función de activación ('ReLU', 'Tanh', etc.).
                    - dropout_rate (float): Tasa de dropout.
                    - optimizer (str): Nombre del optimizador ('Adam', 'SGD', etc.).
                    - epochs (int): Número máximo de épocas.
                    - batch_size (int, opcional): Tamaño del lote (default: 32).
                    - cv_folds (int, opcional): Número de folds para CV (default: 5).
                    - patience (int, opcional): Paciencia para Early Stopping (default: 50).
                    - convergence_threshold (float, opcional): Umbral mejora mínima (default: 1e-4).
                - model_name (str): Nombre para guardar el modelo.
            random_state (int): Semilla para reproducibilidad.
        """
        super().__init__(rom_config, random_state) # Llama al __init__ de la clase padre si es necesario

        # --- Parámetros Requeridos ---
        try:
            self.input_dim = int(rom_config['input_dim'])
            self.output_dim = int(rom_config['output_dim'])
            self.output_folder = rom_config['output_folder']
            hyperparams = rom_config['hyperparams'] # Alias para hiperparámetros

            self.hidden_layers = int(hyperparams['n_layers'])
            self.neurons_per_layer = int(hyperparams['n_neurons'])
            self.learning_rate = float(hyperparams['learning_rate'])
            self.lr_step = int(hyperparams['lr_step'])
            self.lr_decrease_rate = float(hyperparams['lr_decrease_rate'])
            self.activation_function_name = hyperparams['activation']
            self.output_layer_activation = hyperparams['output activation']
            self.dropout = float(hyperparams['dropout_rate'])
            self.optimizer_name = hyperparams['optimizer']
            self.num_epochs = int(hyperparams['epochs'])
            self.batch_size = int(hyperparams['batch_size'])
            self.patience = int(hyperparams['patience'])
            self.cv_folds = int(hyperparams['cv_folds'])
            self.convergence_threshold = float(hyperparams['convergence_threshold'])
            
            self.model_name = rom_config['model_name']
        except KeyError as e:
            raise KeyError(f"Falta el parámetro requerido en rom_config: {e}")
        except ValueError as e:
             raise ValueError(f"Error convirtiendo parámetro a tipo numérico: {e}")

        self.hyperparams = hyperparams

        self.random_state = random_state
        torch.manual_seed(self.random_state) # Seed para PyTorch
        np.random.seed(self.random_state)   # Seed para NumPy

        # --- Estado Interno del Modelo ---
        self.nn = None          # El modelo PyTorch (Sequential)
        self.optimizer = None   # El optimizador PyTorch
        self.train_losses = []  # Historial de pérdidas de entrenamiento (final)
        self.val_losses = []    # Historial de pérdidas de validación (final)

        # --- Datos (se asignarán en train) ---
        self.X_train = None
        self.y_train = None


    def _get_optimizer(self, model):
        """
        Crea un optimizador para los parámetros del modelo proporcionado.

        Args:
            model (torch.nn.Module): Modelo de PyTorch.

        Returns:
            torch.optim.Optimizer: Optimizador configurado.
        """
        if model is None:
            raise ValueError("Se intentó crear un optimizador sin un modelo válido.")

        lr = self.learning_rate # Usa el LR almacenado en el objeto

        if self.optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif self.optimizer_name.lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr)
        # Añade otros optimizadores si es necesario (e.g., AdamW, RMSprop)
        # elif self.optimizer_name.lower() == 'adamw':
        #     optimizer = optim.AdamW(model.parameters(), lr=lr)
        else:
            print(f"Warning: Optimizer '{self.optimizer_name}' not recognized. Using Adam.")
            optimizer = optim.Adam(model.parameters(), lr=lr)
        return optimizer


    def _get_activation_function(self, output_layer=False):
        """
        Devuelve la instancia de la función de activación especificada.

        Args:
            output_layer (bool): Si True, devuelve la activación de la capa de salida.

        Returns:
            torch.nn.Module: Instancia de la función de activación.
        """
        activations = {
            'tanh': torch.nn.Tanh(),
            'relu': torch.nn.ReLU(),
            'sigmoid': torch.nn.Sigmoid(),
            'leakyrelu': torch.nn.LeakyReLU(),
        }
        if not output_layer:
            try:
                # Busca ignorando mayúsculas/minúsculas
                return activations[self.activation_function_name.lower()]
            except KeyError:
                raise ValueError(f"Función de activación desconocida: {self.activation_function_name}")
        else:
            try:
                # Busca ignorando mayúsculas/minúsculas
                return activations[self.output_layer_activation.lower()]
            except KeyError:
                raise ValueError(f"Función de activación desconocida: {self.output_layer_activation}")


    def _crear_red_neuronal(self):
        """
        Crea una instancia de la red neuronal feedforward con los parámetros actuales.

        Returns:
            FeedforwardNet: Red neuronal configurada.
        """
        return FeedforwardNet(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            n_layers=self.hidden_layers,
            n_neurons=self.neurons_per_layer,
            activation=self.activation_function_name,
            dropout_rate=self.dropout,
            output_activation=self.output_layer_activation
        )


    def _train_epoch(self, model, optimizer, X_tensor, y_tensor):
        """
        Ejecuta un ciclo de entrenamiento (una época) sobre los datos en mini-batches.

        Args:
            model (torch.nn.Module): Modelo a entrenar.
            optimizer (torch.optim.Optimizer): Optimizador.
            X_tensor (torch.Tensor): Datos de entrada.
            y_tensor (torch.Tensor): Datos objetivo.

        Returns:
            float: Pérdida promedio de la época.
        """
        model.train() # Asegura modo entrenamiento (activa dropout, etc.)
        total_loss = 0.0
        total_examples = 0
        loss_function = torch.nn.MSELoss() # Pérdida para regresión

        # Crear índices y barajar para cada época
        permutation = torch.randperm(X_tensor.size(0))
        X_tensor_shuffled = X_tensor[permutation]
        y_tensor_shuffled = y_tensor[permutation]

        for i in range(0, len(X_tensor), self.batch_size):
            optimizer.zero_grad() # Limpia gradientes del batch anterior

            X_batch = X_tensor_shuffled[i:i + self.batch_size]
            y_batch = y_tensor_shuffled[i:i + self.batch_size]
            batch_actual = X_batch.size(0)

            if batch_actual == 0: continue # Evita batches vacíos

            predictions = model(X_batch)

            # Asegura compatibilidad de formas para la loss
            loss = loss_function(predictions, y_batch.view_as(predictions))

            loss.backward()  # Calcula gradientes
            optimizer.step() # Actualiza pesos

            total_loss += loss.item() * batch_actual
            total_examples += batch_actual

        # Evita división por cero
        if total_examples == 0: return 0.0
        return total_loss / total_examples


    def train(self, X_train, y_train, X_val=None, y_val=None, validation_mode='cross', save_interval=10):
        """
        Entrena la red neuronal.

        Usa Validación Cruzada si X_val/y_val no se proporcionan.
        Utiliza los hiperparámetros definidos en la instancia (self.*).

        Args:
            X_train (pd.DataFrame o np.ndarray): Datos de entrada de entrenamiento.
            y_train (pd.DataFrame o np.ndarray): Datos objetivo de entrenamiento.
            X_val (pd.DataFrame o np.ndarray, opcional): Datos de entrada de validación.
            y_val (pd.DataFrame o np.ndarray, opcional): Datos objetivo de validación.
            validation_mode (str): 'cross' para CV, 'single' para validación explícita.
            save_interval (int): Frecuencia para guardar el modelo (no implementado).
        """
        self.X_train = X_train # Guarda referencia si la necesitas después
        self.y_train = y_train

        # --- Preparación de Datos ---
        # Convertir a NumPy si son DataFrames de Pandas
        if isinstance(X_train, pd.DataFrame): X_train_np = X_train.to_numpy()
        else: X_train_np = X_train
        if isinstance(y_train, pd.DataFrame): y_train_np = y_train.to_numpy()
        else: y_train_np = y_train

        # Asegurar que 'y' sea 2D (N, output_dim), especialmente para MSELoss
        if len(y_train_np.shape) == 1:
            y_train_np = y_train_np.reshape(-1, 1)
        # Verifica que la segunda dimensión coincida con output_dim
        if y_train_np.shape[1] != self.output_dim:
             # Intenta remodelar si solo hay una columna y output_dim es 1
            if y_train_np.shape[1] == 1 and self.output_dim > 1:
                 print(f"Warning: y_train tiene {y_train_np.shape[1]} columnas pero output_dim es {self.output_dim}. Verifica los datos.")
            elif self.output_dim == 1 and y_train_np.shape[1] != 1:
                 print(f"Warning: y_train tiene {y_train_np.shape[1]} columnas pero output_dim es 1. Verifica los datos.")
            # No lanzar error, pero advertir
            # raise ValueError(f"La segunda dimensión de y_train ({y_train_np.shape[1]}) no coincide con output_dim ({self.output_dim})")


        print(f"Target variance (train): {np.var(y_train_np):.6f}")

        # --- Selección de Modo: Validación Explícita vs. Cross-Validation ---
        perform_cv = True
        if validation_mode == 'single':
            perform_cv = False
            print("Usando conjunto de validación explícito proporcionado.")
            # Preparar datos de validación explícitos
            if isinstance(X_val, pd.DataFrame): X_val_np = X_val.to_numpy()
            else: X_val_np = X_val
            if isinstance(y_val, pd.DataFrame): y_val_np = y_val.to_numpy()
            else: y_val_np = y_val

            if len(y_val_np.shape) == 1:
                y_val_np = y_val_np.reshape(-1, 1)
            if y_val_np.shape[1] != self.output_dim:
                 print(f"Warning: y_val tiene {y_val_np.shape[1]} columnas pero output_dim es {self.output_dim}. Verifica los datos.")
                # raise ValueError(f"La segunda dimensión de y_val ({y_val_np.shape[1]}) no coincide con output_dim ({self.output_dim})")

            # Convertir a tensores una sola vez
            X_train_tensor_full = torch.tensor(X_train_np, dtype=torch.float32)
            y_train_tensor_full = torch.tensor(y_train_np, dtype=torch.float32)
            X_val_tensor_full = torch.tensor(X_val_np, dtype=torch.float32)
            y_val_tensor_full = torch.tensor(y_val_np, dtype=torch.float32)
        else:
            print(f"Iniciando Cross-Validation con {self.cv_folds} folds.")
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            fold_val_losses = []

            # --- Bucle de Cross-Validation ---
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_np, y_train_np)):
                print(f"--- Fold {fold+1}/{self.cv_folds} ---")

                # Datos del fold actual
                X_train_fold_np = X_train_np[train_idx]
                y_train_fold_np = y_train_np[train_idx]
                X_val_fold_np = X_train_np[val_idx]
                y_val_fold_np = y_train_np[val_idx]

                # Convertir a tensores para este fold
                X_train_fold = torch.tensor(X_train_fold_np, dtype=torch.float32)
                y_train_fold = torch.tensor(y_train_fold_np, dtype=torch.float32)
                X_val_fold = torch.tensor(X_val_fold_np, dtype=torch.float32)
                y_val_fold = torch.tensor(y_val_fold_np, dtype=torch.float32)

                # Crear modelo, optimizador y scheduler PARA ESTE FOLD
                model_fold = self._crear_red_neuronal() # Usa parámetros de self
                optimizer_fold = self._get_optimizer(model_fold)
                scheduler_fold = StepLR(optimizer_fold, step_size=self.lr_step, gamma=self.lr_decrease_rate)
                
                loss_function = torch.nn.MSELoss()

                best_fold_val_loss = float('inf')
                epochs_no_improve = 0

                # Bucle de entrenamiento para el fold
                for epoch in range(self.num_epochs):
                    train_loss_epoch = self._train_epoch(model_fold, optimizer_fold, X_train_fold, y_train_fold)

                    # Evaluación en validación del fold
                    model_fold.eval()
                    val_loss_epoch = 0.0
                    with torch.no_grad():
                        val_preds = model_fold(X_val_fold)
                        val_loss_epoch = loss_function(val_preds, y_val_fold.view_as(val_preds)).item()

                    print_freq = 100 # Frecuencia de impresión
                    if epoch % print_freq == 0 or epoch == self.num_epochs - 1:
                        current_lr = optimizer_fold.param_groups[0]['lr']
                        print(f"  Fold {fold+1}, Epoch {epoch}/{self.num_epochs}, Train Loss: {train_loss_epoch:.6f}, Val Loss: {val_loss_epoch:.6f}, LR: {current_lr:.2e}")

                    # Check Early Stopping
                    if val_loss_epoch < best_fold_val_loss - self.convergence_threshold:
                        best_fold_val_loss = val_loss_epoch
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= self.patience:
                            print(f"  Early stopping en Fold {fold+1} en época {epoch+1}.")
                            break

                    scheduler_fold.step() # Llama al scheduler al final de cada época

                fold_val_losses.append(best_fold_val_loss)
                print(f"  Fold {fold+1} finalizado. Mejor Val Loss: {best_fold_val_loss:.6f}")

            avg_val_loss = np.mean(fold_val_losses)
            print(f"\nPromedio de la pérdida de validación en CV: {avg_val_loss:.6f}")

        # --- Entrenamiento Final (siempre se hace, con todos los datos de train) ---
        print("\n--- Iniciando Entrenamiento Final con todos los datos de train ---")

        # Crear el modelo final y almacenar en self.nn
        self.nn = self._crear_red_neuronal()
        self.optimizer = self._get_optimizer(self.nn) # Optimizador para el modelo final
        scheduler = StepLR(self.optimizer, step_size=self.lr_step, gamma=self.lr_decrease_rate)
        loss_function = torch.nn.MSELoss()

        # Si se hizo CV, necesitamos los tensores completos ahora
        if perform_cv:
            X_train_tensor_full = torch.tensor(X_train_np, dtype=torch.float32)
            y_train_tensor_full = torch.tensor(y_train_np, dtype=torch.float32)
            # No hay conjunto de validación explícito en este caso para el ent. final
            X_val_tensor_full = None
            y_val_tensor_full = None

        self.train_losses = [] # Reinicia historial para el entrenamiento final
        self.val_losses = []
        best_final_val_loss = float('inf')
        epochs_no_improve_final = 0
        last_epoch = self.num_epochs # Guarda la última época alcanzada

        # --- Bucle de Entrenamiento Final ---
        for epoch in range(self.num_epochs):
            current_lr_before_step = self.optimizer.param_groups[0]['lr'] # LR antes del paso del scheduler

            # Paso de entrenamiento
            train_loss = self._train_epoch(self.nn, self.optimizer, X_train_tensor_full, y_train_tensor_full)
            self.train_losses.append(train_loss)

            # Paso de validación (si hay datos de validación explícitos)
            val_loss = None
            if X_val_tensor_full is not None and y_val_tensor_full is not None:
                self.nn.eval()
                with torch.no_grad():
                    final_preds = self.nn(X_val_tensor_full)
                    val_loss = loss_function(final_preds, y_val_tensor_full.view_as(final_preds)).item()
                    self.val_losses.append(val_loss)

                # Imprimir estado
                print_freq = 100
                if epoch % print_freq == 0 or epoch == self.num_epochs - 1:
                    print(f"Epoch {epoch}/{self.num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr_before_step:.2e}")

                # Check Early Stopping (solo si hay validación)
                if val_loss < best_final_val_loss - self.convergence_threshold:
                    best_final_val_loss = val_loss
                    epochs_no_improve_final = 0
                    # --- Opcional: Guardar el MEJOR modelo basado en validación ---
                    # output_folder = os.path.join(os.getcwd(), 'results', self.model_name)
                    # os.makedirs(output_folder, exist_ok=True)
                    # best_model_path = os.path.join(output_folder, f'best_{self.model_name}_model.pth')
                    # torch.save(self.nn.state_dict(), best_model_path)
                    # print(f"  * Mejor modelo guardado en época {epoch+1} con Val Loss: {best_final_val_loss:.6f}")
                    # --- Fin Opcional ---
                else:
                    epochs_no_improve_final += 1
                    if epochs_no_improve_final >= self.patience:
                        print(f"Early stopping final en época {epoch+1}.")
                        last_epoch = epoch + 1
                        break
            else:
                # Si no hay validación, solo imprime train loss
                 print_freq = 100
                 if epoch % print_freq == 0 or epoch == self.num_epochs - 1:
                    print(f"Epoch {epoch}/{self.num_epochs}, Train Loss: {train_loss:.6f}, LR: {current_lr_before_step:.2e}")

            # Paso del scheduler (después de la época)
            scheduler.step()

            # Imprimir si el LR cambió
            new_lr = self.optimizer.param_groups[0]['lr']
            if abs(new_lr - current_lr_before_step) > 1e-9: # Tolerancia numérica
                print(f"  Learning rate actualizado a {new_lr:.2e} en epoch {epoch+1}")

        final_lr = self.optimizer.param_groups[0]['lr']
        print(f"El entrenamiento finalizó después de {last_epoch} épocas.")
        print(f"Learning rate final: {final_lr:.2e}.")
        if not perform_cv and best_final_val_loss != float('inf'):
            print(f"Mejor loss de validación alcanzado: {best_final_val_loss:.6f}")

        
        var_y = y_train.var(axis=0)  # Series con varianza por columna
        print("Varianza por columna:\n", var_y)
        

    def predict(self, X_test) -> np.ndarray:
        """
        Realiza predicciones usando el modelo final entrenado (self.nn).

        Args:
            X_test (pd.DataFrame o np.ndarray): Datos de entrada para predicción.

        Returns:
            np.ndarray: Predicciones del modelo.
        """
        if self.nn is None:
            raise RuntimeError("El modelo no ha sido entrenado todavía. Llama a train() primero.")

        # Asegurar que el modelo esté en modo evaluación
        self.nn.eval()

        # Convertir a NumPy y luego a Tensor
        if isinstance(X_test, pd.DataFrame): X_test_np = X_test.to_numpy()
        else: X_test_np = X_test
        X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)

        # Realizar predicción sin calcular gradientes
        with torch.no_grad():
            predictions_tensor = self.nn(X_test_tensor)

        # Convertir a NumPy array para devolver
        y_pred_np = predictions_tensor.cpu().numpy()

        # Quitar dimensión extra si output_dim es 1 y el resultado es (N, 1)
        if self.output_dim == 1 and y_pred_np.ndim == 2 and y_pred_np.shape[1] == 1:
           y_pred_np = y_pred_np.squeeze(axis=1) # Devuelve (N,)

        # Nota: No volver a self.nn.train() aquí. La predicción no debe cambiar el estado del modelo.

        return y_pred_np


    def idk_run(self, X_params_dict):
        """
        Ejecuta el ROM usando los parámetros de entrada para hacer una predicción
        y mapea la salida a los nombres de columnas de self.y_train. Además, verifica que
        el diccionario de entrada contenga tantas llaves como columnas tiene self.X_train y
        que el número de resultados coincida con las columnas de self.y_train.

        Args:
            X_params_dict (dict): Diccionario con variables de entrada, ejemplo:
                                {'var1': 34, 'var2': 45, ...}

        Returns:
            dict: Diccionario con los resultados de la predicción, donde las llaves
                corresponden a los encabezados del DataFrame de entrenamiento self.y_train,
                por ejemplo: {'nombre_col1': 45, 'nombre_col2': 89, ...}
        """

        # Verificar que self.X_train exista y tenga columnas
        if hasattr(self, "X_train") and hasattr(self.X_train, "columns"):
            if len(X_params_dict) != len(self.X_train.columns):
                raise ValueError("El número de variables de entrada no coincide con el número de columnas en X_train. Se esperaban {} variables, pero se recibieron {}.".format(len(self.X_train.columns), len(X_params_dict)))
        else:
            print("Advertencia: No se pudo verificar el número de variables de X_train, 'X_train' no está definido o no tiene atributo 'columns'.")
        

        # 1) Crea el array de entrada
        X = np.array([[X_params_dict[col] for col in self.X_train.columns]], dtype=float)


        # Separar: primeras 15 variables (numéricas) y la última (N_trans)
        X_numeric = X[:, :-1]   # (1, 15)
        X_discrete = X[:, -1:]  # (1, 1)

        # 1bis) ESCALAR INPUTS CON EL SCALER DEL ENTRENAMIENTO
        input_scaler = joblib.load(os.path.join(self.output_folder, 'input_scaler.pkl'))
        X_scaled_numeric = input_scaler.transform(X_numeric)  # Aquí sí escalás la entrada
        X_scaled_full = np.concatenate([X_scaled_numeric, X_discrete], axis=1)  # (1, 16)

        # 2) Predice (salida escalada)
        y_pred_scaled = self.predict(X_scaled_full)  # ndarray shape (1, n_outputs) o (n_outputs,)
        
        # Verificar que el número de resultados coincide con el número de columnas de y_train
        if hasattr(self, "y_train") and hasattr(self.y_train, "columns"):
            expected_n_results = len(self.y_train.columns)
            if y_pred_scaled.size != expected_n_results:
                raise ValueError("El número de resultados predichos ({}) no coincide con el número de columnas en y_train ({}).".format(y_pred_scaled.size, expected_n_results))
            # Obtener los nombres de las columnas a utilizar como llaves
            target_keys = list(self.y_train.columns)
        else:
            print("Advertencia: No se pudo obtener las columnas de y_train, se usarán llaves genéricas.")
            target_keys = [f"result{i+1}" for i in range(y_pred_scaled.size)]

        # Aplanar la salida para trabajar con un array 1D si es necesario
        if y_pred_scaled.ndim > 1:
            # Si y_pred es 2D y tiene una sola fila, se aplana
            y_pred_flat = y_pred_scaled.flatten() if y_pred_scaled.shape[0] == 1 else y_pred_scaled[0]
        else:
            y_pred_flat = y_pred_scaled


        # 3) Desescala si tienes scaler
        # Cargar el diccionario de scalers
        output_scalers: dict = joblib.load(os.path.join(self.output_folder, 'output_scaler.pkl'))

        # Asegura que sea array 1D
        if y_pred_scaled.ndim > 1:
            y_pred_flat = y_pred_scaled.flatten()
        else:
            y_pred_flat = y_pred_scaled

        # Obtener nombres de columnas de salida
        keys = list(self.y_train.columns) if hasattr(self, 'y_train') else [f"out{i}" for i in range(len(y_pred_flat))]

        # Aplicar scaler individual por columna
        y_pred_orig = []
        for i, col in enumerate(keys):
            scaler = output_scalers[col]
            val = scaler.inverse_transform([[y_pred_flat[i]]])[0][0]
            y_pred_orig.append(val)

        # Construir el diccionario de resultados
        results = {k: float(v) for k, v in zip(keys, y_pred_orig)}

        # 4) Mapear a dict con nombres de columnas
        # Construir el diccionario de resultados utilizando los nombres de columnas como llaves

        keys = list(self.y_train.columns) if hasattr(self, 'y_train') else [f"out{i}" for i in range(len(y_pred_orig))]
        results = {k: float(v) for k, v in zip(keys, y_pred_orig)}

        return results