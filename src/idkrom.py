import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
from abc import abstractmethod

class idkROM:

    def __init__(self):
        self.scaler = StandardScaler()

    def preprocessing(self, data):
        """Aplica normalización y detección de outliers"""

        # Separar inputs y outputs
        self.inputs = data.iloc[:, :7]  # Primeras 7 columnas como inputs
        self.outputs = data.iloc[:, 7:]  # Resto como outputs

        n_samples = data.shape[0]

        #sample selection
        data = data.iloc[:n_samples, :]

        # Data profiling
        report = ProfileReport(data, title='Report', minimal=True)

        # exportar reporte a html
        output_folder = os.path.join(os.getcwd(), 'results')
        os.makedirs(os.path.dirname(output_folder), exist_ok=True)
        report.to_file(os.path.join(output_folder,'train_profiling_report.html'))
        
        
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

            plt.savefig(os.path.join(output_folder, "preprocessed_data.png"))
            plt.close(fig)

        #print('Box plot for input & output variables')
        #boxplot(data)

        # Drop outliers
        df_filtered = data[(data['Q_total_int'] <= 0.00025) &
                   (data['h_mean_int'] <= 0.0002) &
                   (data['h_mean_ext'] <= 0.00015)]

        # Normalize variables
        scaler = MinMaxScaler()
        df_normalized = pd.DataFrame(scaler.fit_transform(df_filtered), columns=df_filtered.columns)

        joblib.dump(scaler, os.path.join(output_folder, 'scaler.pkl'))

        boxplot(df_normalized)

        df_normalized.describe()

        X_normalized = df_normalized[self.inputs.columns]
        y_normalized = df_normalized[self.outputs.columns]

        # Guardar los datos preprocesados en un CSV
        preprocessed_file_path = os.path.join(output_folder, 'preprocessed_data.csv')
        df_normalized.to_csv(preprocessed_file_path, index=False, sep=";", decimal=".")
        print(f'Datos preprocesados guardados en {preprocessed_file_path}')

        return X_normalized, y_normalized
            
    """Clase base abstracta
    Sirve de plantilla para las subclases de cada método ROM, donde se implementan las funciones train y evaluate dedicadas."""
    class Modelo:
        def __init__(self):
            self.model = None

        @abstractmethod
        def train(self, X_train, y_train):
            """Método a sobrescribir en cada subclase"""
            raise NotImplementedError

        @abstractmethod
        def evaluate(self, X_test, y_test):
            """Método a sobrescribir en cada subclase"""
            raise NotImplementedError
        
        @abstractmethod
        def score(self, X_test, y_test):
            """Método a sobrescribir en cada subclase"""
            raise NotImplementedError
