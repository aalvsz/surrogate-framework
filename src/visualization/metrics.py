import matplotlib.pyplot as plt
import pandas as pd

class ErrorMetrics:
    def __init__(self, model, model_name, y_test, y_pred, train_losses=None, val_losses=None):
        self.model = model
        self.model_name = model_name
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
        plt.savefig(f"images/error_metrics_{self.model_name}")
        plt.show()
        return 0