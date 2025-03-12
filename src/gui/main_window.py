from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QDialog, QFormLayout, QSpinBox, QFileDialog
from PyQt6.QtCore import Qt
from src.idkROM_v0 import idkROM  # Importar la clase base idkROM
import numpy as np
import pandas as pd

class ROMApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROM Tool")
        self.setGeometry(100, 100, 600, 400)

        # Inicializar datos
        self.data = None
        self.inputs = None
        self.outputs = None

        # Crear los componentes de la GUI
        self.create_widgets()

    def create_widgets(self):
        layout = QVBoxLayout(self)

        # Etiqueta y cuadro de entrada para los datos (x, y)
        self.label = QLabel("Importar conjunto de datos (x, y)", self)
        layout.addWidget(self.label)

        # Combo para elegir si se cargan datos crudos o preprocesados
        self.data_source_combobox = QComboBox(self)
        self.data_source_combobox.addItems(["Datos crudos", "Datos preprocesados"])
        layout.addWidget(QLabel("Tipo de datos:"))
        layout.addWidget(self.data_source_combobox)

        self.upload_button = QPushButton("Cargar datos", self)
        self.upload_button.clicked.connect(self.load_data)
        layout.addWidget(self.upload_button)

        self.model_label = QLabel("Selecciona el modelo ROM", self)
        layout.addWidget(self.model_label)

        # Lista desplegable para seleccionar el modelo ROM
        self.model_options = ["Neural Network", "RBF", "Response Surface", "Gaussian Process", "SVR"]
        self.model_combobox = QComboBox(self)
        self.model_combobox.addItems(self.model_options)
        layout.addWidget(self.model_combobox)

        self.train_button = QPushButton("Entrenar modelo", self)
        self.train_button.clicked.connect(self.train_model)
        layout.addWidget(self.train_button)

        self.result_label = QLabel("Resultado:", self)
        layout.addWidget(self.result_label)

    def load_data(self):
        # Abrir el cuadro de diálogo para seleccionar un archivo CSV
        file_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar archivo CSV", "")
        if file_path:
            try:
                # Leer el CSV (ajusta el delimiter y decimal según tu archivo)
                df = pd.read_csv(file_path, delimiter=",", decimal=".")
                # Mostrar estado de carga
                self.result_label.setText("Datos cargados correctamente.")
                # Dependiendo de la opción seleccionada, se preprocesan o se usan tal cual
                data_source = self.data_source_combobox.currentText()
                if data_source == "Datos crudos":
                    # Preprocesar: se ejecuta el método preprocessing que normaliza, elimina outliers, etc.
                    rom = idkROM()
                    self.inputs, self.outputs = rom.preprocessing(df)
                else:  # Datos preprocesados
                    # Se asume que el CSV ya contiene los datos normalizados, con las primeras 7 columnas como inputs
                    # y el resto como outputs
                    self.inputs = df.iloc[:, :7]
                    self.outputs = df.iloc[:, 7:]
                self.data = df  # Guarda el DataFrame (puedes ajustar según necesites)
                self.result_label.setText("Datos procesados correctamente.")
            except Exception as e:
                self.result_label.setText(f"Error al cargar los datos: {str(e)}")

    def train_model(self):
        if self.data is None:
            self.result_label.setText("Error: No se han cargado datos.")
            return

        # Obtener el modelo seleccionado
        model_name = self.model_combobox.currentText()
        
        # Obtener los datos (convertir a valores numpy)
        X_train = self.inputs.values
        y_train = self.outputs.values

        # Si se selecciona "Neural Network", mostrar el cuadro de diálogo para configurar parámetros
        if model_name == "Neural Network":
            dialog = NeuralNetworkConfigDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                hidden_layers = dialog.hidden_layers_spinbox.value()
                neurons_per_layer = dialog.neurons_per_layer_spinbox.value()
                activation_function = dialog.activation_function_combobox.currentText()
                # Importante: se debe especificar la dimensión de salida según el número de columnas de y_train
                model = idkROM.NeuralNetworkROM(input_dim=X_train.shape[1],
                                                output_dim=y_train.shape[1],
                                                hidden_layers=hidden_layers,
                                                neurons_per_layer=neurons_per_layer,
                                                activation_function=activation_function)
        elif model_name == "RBF":
            model = idkROM.RBFROM()
        elif model_name == "Response Surface":
            model = idkROM.ResponseSurfaceROM(degree=2)
        elif model_name == "Gaussian Process":
            model = idkROM.GaussianProcessROM()
        elif model_name == "SVR":
            model = idkROM.SVRROM()

        # Entrenar el modelo
        model.train(X_train, y_train)

        # Evaluar el modelo
        mse = model.score(X_train, y_train)
        self.result_label.setText(f"Modelo entrenado, MSE: {mse:.4f}")

class NeuralNetworkConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Configuración de Red Neuronal")
        layout = QFormLayout(self)

        # Configurar número de capas ocultas
        self.hidden_layers_spinbox = QSpinBox(self)
        self.hidden_layers_spinbox.setMinimum(1)
        self.hidden_layers_spinbox.setMaximum(10)
        self.hidden_layers_spinbox.setValue(2)  # Valor por defecto
        layout.addRow("Número de capas ocultas:", self.hidden_layers_spinbox)

        # Configurar neuronas por capa
        self.neurons_per_layer_spinbox = QSpinBox(self)
        self.neurons_per_layer_spinbox.setMinimum(1)
        self.neurons_per_layer_spinbox.setMaximum(200)
        self.neurons_per_layer_spinbox.setValue(50)  # Valor por defecto
        layout.addRow("Neurona por capa:", self.neurons_per_layer_spinbox)

        # Configurar función de activación
        self.activation_function_combobox = QComboBox(self)
        self.activation_function_combobox.addItems(["Tanh", "ReLU", "Sigmoid"])
        layout.addRow("Función de activación:", self.activation_function_combobox)

        # Botones de aceptar y cancelar
        self.accept_button = QPushButton("Aceptar", self)
        self.accept_button.clicked.connect(self.accept)
        self.reject_button = QPushButton("Cancelar", self)
        self.reject_button.clicked.connect(self.reject)
        layout.addRow(self.accept_button, self.reject_button)
