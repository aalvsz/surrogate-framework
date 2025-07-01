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
        if hasattr(self.model.model, 'train_losses') and hasattr(self.model.model, 'val_losses'):
            plt.subplot(2, 2, 4)
            self.epoch = range(1, len(self.model.model.train_losses) + 1)
            plt.plot(self.epoch, self.model.model.train_losses, label='Pérdida de Entrenamiento')
            plt.plot(self.epoch, self.model.model.val_losses, label='Pérdida de Validación')
            plt.xlabel("Épocas")
            plt.ylabel("Pérdida")
            plt.title("Curvas de Pérdida de Entrenamiento y Validación")
            plt.legend()
            plt.grid(True)


        plt.tight_layout()
        plt.savefig(os.path.join(self.rom_config['output_folder'], f"error_metrics_{self.rom_config['model_name']}.png"))
        plt.close()

        from matplotlib.ticker import ScalarFormatter

        # Calcula los errores
        errors = self.y_pred - self.y_test.values
        eps = 1e-10
        error_rel_percent = (self.y_pred - self.y_test.values) / (self.y_test.values + eps) * 100

        num_outputs = self.y_test.shape[1]
        cols = self.y_test.columns

        # Generar una figura por cada variable
        for i in range(num_outputs):
            fig, ax = plt.subplots(figsize=(6, 5))

            # Gráfico: Error absoluto vs. Predicción
            ax.scatter(self.y_pred[:, i], errors[:, i], alpha=0.5)
            ax.axhline(0, color='red', linestyle='--')
            ax.set_xlabel(f'Predicciones - {cols[i]}')
            ax.set_ylabel('Error (Pred - Real)')
            ax.set_title(f'Error vs. Predicción para {cols[i]}')
            ax.grid(True)

            # Formato científico para ejes
            formatter_x = ScalarFormatter(useMathText=True)
            formatter_x.set_scientific(True)
            formatter_x.set_powerlimits((-2, 4))  # Activa notación científica fuera de este rango
            ax.xaxis.set_major_formatter(formatter_x)

            formatter_y = ScalarFormatter(useMathText=True)
            formatter_y.set_scientific(True)
            formatter_y.set_powerlimits((-2, 4))  # Igual para el eje Y
            ax.yaxis.set_major_formatter(formatter_y)

            fig.tight_layout()
            fig.savefig(os.path.join(self.rom_config['output_folder'], f"error_{cols[i]}.png"))
            plt.close(fig)

        return 0