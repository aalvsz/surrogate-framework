import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

class ErrorMetrics:
    def __init__(self, model, rom_config, y_test, y_pred, train_losses=None, val_losses=None):
        self.model = model
        self.rom_config = rom_config
        self.y_test = y_test
        self.y_pred = y_pred
        self.train_losses = train_losses
        self.val_losses = val_losses

# Calcular BIC
    def calculate_bic(self, y_true, y_pred):
        """
        Calculates the Bayesian Information Criterion (BIC).

        Args:
            y_true (np.ndarray or pd.DataFrame): True labels for the test data.
            y_pred (np.ndarray): Predictions made by the model on the test data.
            rom_config (dict): Configuration dictionary for the model.

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

        # Calcular MSE y SSE
        mse = np.mean((y_pred - y_true_np) ** 2)
        sse = mse * n
        
        # Calcular el número de parámetros del modelo
        if self.rom_config['model_name'].lower() == 'neural_network':
            # Obtener dimensiones de la red neuronal del rom_config
            input_dim = self.rom_config['input_dim']
            output_dim = self.rom_config['output_dim']
            
            if 'hyperparams' in self.rom_config:
                hidden_layers = self.rom_config['hyperparams'].get('n_layers', 0)
                neurons_per_layer = self.rom_config['hyperparams'].get('n_neurons', 0)
                
                # Calcular número total de parámetros (pesos + sesgos)
                if hidden_layers > 0:
                    # Primera capa: input_dim * neurons_per_layer + neurons_per_layer (sesgos)
                    num_params = (input_dim + 1) * neurons_per_layer
                    
                    # Capas ocultas intermedias (si hay más de una capa oculta)
                    if hidden_layers > 1:
                        num_params += (hidden_layers - 1) * (neurons_per_layer + 1) * neurons_per_layer
                    
                    # Última capa: neurons_per_layer * output_dim + output_dim (sesgos)
                    num_params += (neurons_per_layer + 1) * output_dim
                else:
                    # Si no hay capas ocultas, conexión directa entre entrada y salida
                    num_params = (input_dim + 1) * output_dim
            else:
                # Estimación por defecto si no hay información de hiperparámetros
                num_params = input_dim * output_dim + output_dim
                
        elif self.rom_config['model_name'].lower() == 'gaussian_process':
            # Cálculo más preciso para procesos gaussianos
            if hasattr(self, 'model') and hasattr(self.model, 'kernel_'):
                # Obtener número de parámetros del kernel
                kernel_params = 0
                
                # Analizar tipo de kernel para contar parámetros correctamente
                from sklearn.gaussian_process.kernels import RBF, Matern
                kernel = self.model.kernel_
                
                # Para kernels RBF
                if isinstance(kernel, RBF) or "RBF" in str(kernel):
                    if hasattr(kernel, 'length_scale'):
                        if kernel.length_scale_bounds == 'fixed':
                            kernel_params += 0  # Escalas fijas no son parámetros
                        elif np.isscalar(kernel.length_scale):
                            kernel_params += 1  # Una escala global
                        else:
                            kernel_params += len(kernel.length_scale)  # Una escala por dimensión
                    else:
                        # Estimación para RBF sin atributos accesibles
                        input_dim = self.rom_config.get('input_dim', 1)
                        kernel_params += input_dim
                
                # Para kernels Matern
                elif isinstance(kernel, Matern) or "Matern" in str(kernel):
                    if hasattr(kernel, 'length_scale'):
                        if kernel.length_scale_bounds == 'fixed':
                            kernel_params += 0
                        elif np.isscalar(kernel.length_scale):
                            kernel_params += 1
                        else:
                            kernel_params += len(kernel.length_scale)
                    else:
                        input_dim = self.rom_config.get('input_dim', 1)
                        kernel_params += input_dim
                    
                    # Añadir parámetro nu si no está fijo
                    if hasattr(kernel, 'nu_bounds') and kernel.nu_bounds != 'fixed':
                        kernel_params += 1
                
                # Para otros kernels o kernels compuestos
                else:
                    # Contar parámetros a través de theta
                    if hasattr(kernel, 'theta'):
                        kernel_params = len(kernel.theta)
                    else:
                        # Estimación basada en la dimensión de entrada
                        input_dim = self.rom_config.get('input_dim', 1)
                        kernel_params = input_dim
                
                # Añadir parámetro de ruido (alpha/noise)
                if hasattr(self.model, 'alpha_bounds') and self.model.alpha_bounds == 'fixed':
                    noise_params = 0
                else:
                    noise_params = 1
                
                # Total de parámetros
                num_params = kernel_params + noise_params
            else:
                # Estimación cuando no podemos acceder al modelo entrenado
                input_dim = self.rom_config.get('input_dim', 1)
                # Típicamente, longitud de escala por dimensión + varianza del kernel + ruido
                num_params = input_dim + 2
            
        elif self.rom_config['model_name'].lower() == 'response_surface':
            # Para superficies de respuesta polinómicas
            degree = self.rom_config['hyperparams'].get('degree', 2) if 'hyperparams' in self.rom_config else 2
            # Número de términos en un polinomio completo de grado 'degree'
            # con 'input_dim' variables
            from math import comb
            input_dim = self.rom_config['input_dim']
            num_params = sum(comb(input_dim + i, i) for i in range(degree + 1))
            
        elif self.rom_config['model_name'].lower() in ['rbf', 'svr']:
            # Para RBF y SVR, una estimación simple
            num_params = self.rom_config['input_dim'] * 2 + 5  # Aproximación
            
        else:
            # Valor por defecto para otros modelos
            num_params = 10
            
        # Calcular BIC: n*ln(SSE/n) + k*ln(n)
        # Donde k es el número de parámetros
        bic = n * np.log(sse/n) + num_params * np.log(n)
        print(f"Número de parámetros estimados: {num_params}")
        print(f"Valor de BIC: {bic:.2f}")
        return bic
    

    def create_error_graphs(self):
        plt.figure(figsize=(15, 10))  # Adjust figure size for a 2x2 grid

        # 1. Predicciones vs. Valores Reales (Arriba, Izquierda)
        plt.subplot(2, 2, 1)
        if isinstance(self.y_test, pd.DataFrame):
            y_test_np = self.y_test.values
        else:
            y_test_np = self.y_test
        plt.scatter(y_test_np.flatten(), self.y_pred.flatten())
        plt.xlabel("Valores Reales")
        plt.ylabel("Predicciones")
        plt.title("Predicciones vs. Valores Reales")
        plt.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'k--', lw=2) # Línea de referencia
        plt.grid(True)

        # 2. Distribución de Errores (Arriba, Derecha)
        plt.subplot(2, 2, 2)
        errors = y_test_np.flatten() - self.y_pred.flatten()
        plt.hist(errors, bins=50, edgecolor='black')
        plt.xlabel("Error")
        plt.ylabel("Frecuencia")
        plt.title("Distribución de Errores")
        plt.grid(True)

        # 3. Errores vs. Predicciones (Abajo, Izquierda)
        plt.subplot(2, 2, 3)
        plt.scatter(self.y_pred.flatten(), errors)
        plt.xlabel("Predicciones")
        plt.ylabel("Error")
        plt.title("Errores vs. Predicciones")
        plt.axhline(y=0, color='r', linestyle='--') # Línea de referencia en cero
        plt.grid(True)

        # 4. Training and Validation Loss Curves (Abajo, Derecha - solo para NeuralNetworkROM)
        if hasattr(self.model, 'train_losses') and hasattr(self.model, 'val_losses'):
            plt.subplot(2, 2, 4)
            self.epoch = range(1, len(self.model.train_losses) + 1)
            plt.plot(self.epoch, self.model.train_losses, label='Pérdida de Entrenamiento')
            plt.plot(self.epoch, self.model.val_losses, label='Pérdida de Validación')
            plt.xlabel("Épocas")
            plt.ylabel("Pérdida")
            plt.title("Curvas de Pérdida de Entrenamiento y Validación")
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), 'images', f"error_metrics_{self.rom_config['model_name']}.png"))
        plt.close()
        return 0