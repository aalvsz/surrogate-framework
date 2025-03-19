import pandas as pd
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QDialog, QFormLayout, QSpinBox, QFileDialog, QDoubleSpinBox
from src.idkrom import idkROM  # Importar la clase base idkROM
from src.pre import Pre
from src.models.neural_network import NeuralNetworkROM
from src.models.gaussian_process import GaussianProcessROM


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
                # Dependiendo de la opción seleccionada, se preprocesan o se usan tal cual
                data_source = self.data_source_combobox.currentText()

                if data_source == "Datos crudos":
                    # Preprocesar: se ejecuta el método preprocessing que normaliza, elimina outliers, etc.
                    df = pd.read_csv(file_path, delimiter=",", decimal=".")
                    rom = idkROM()
                    self.inputs, self.outputs = Pre.preprocessing(df)

                else:  # Datos preprocesados
                    # Se asume que el CSV ya contiene los datos normalizados, con las primeras 7 columnas como inputs
                    # y el resto como outputs
                    df = pd.read_csv(file_path, delimiter=";", decimal=".")
                    self.inputs = df.iloc[:, :7]
                    self.outputs = df.iloc[:, 7:]

                self.result_label.setText("Datos cargados correctamente.")

                self.data = df  # Guarda el DataFrame (puedes ajustar según necesites)
                self.result_label.setText("Datos procesados correctamente.")
            except Exception as e:
                self.result_label.setText(f"Error al cargar los datos: {str(e)}")
                print(f"Error al cargar los datos: {str(e)}")


    def train_model(self):
        if self.data is None:
            self.result_label.setText("Error: No se han cargado datos.")
            return

        model_name = self.model_combobox.currentText()
        X_train = self.inputs.values
        y_train = self.outputs.values

        model = None  # Asegurarnos de que el modelo esté vacío inicialmente

        # Crear el cuadro de diálogo según el modelo seleccionado
        if model_name == "Neural Network":
            dialog = NeuralNetworkConfigDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                hidden_layers = dialog.hidden_layers_spinbox.value()
                neurons_per_layer = dialog.neurons_per_layer_spinbox.value()
                activation_function = dialog.activation_function_combobox.currentText()
                learning_rate = dialog.learning_rate_spinbox.value()
                num_epochs = dialog.num_epochs_spinbox.value()
                optimizer = dialog.optimizer_combobox.currentText()

                model = NeuralNetworkROM(input_dim=X_train.shape[1],
                                        output_dim=y_train.shape[1],
                                        hidden_layers=hidden_layers,
                                        neurons_per_layer=neurons_per_layer,
                                        activation_function=activation_function,
                                        learning_rate=learning_rate,
                                        optimizer=optimizer,
                                        num_epochs=num_epochs)
            
            # Entrenar el modelo
            if model is not None:
                model.train(X_train, y_train)
        
        elif model_name == "RBF":
            model = idkROM.RBFROM()

        elif model_name == "Response Surface":
            model = idkROM.ResponseSurfaceROM(degree=2)

        elif model_name == "Gaussian Process":
            dialog = GaussianProcessConfigDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                kernel, noise, optimizer = dialog.get_config()
                model = GaussianProcessROM(kernel=kernel, noise=noise, optimizer=optimizer)
            if model is not None:
                    model.train(X_train, y_train)

        elif model_name == "SVR":
            model = idkROM.SVRROM()

        # Verificar si el usuario canceló la selección del modelo
        if model is None:
            self.result_label.setText("Operación cancelada.")
            return

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
        self.hidden_layers_spinbox.setMaximum(50)
        self.hidden_layers_spinbox.setValue(2)
        layout.addRow("Número de capas ocultas:", self.hidden_layers_spinbox)

        # Configurar neuronas por capa
        self.neurons_per_layer_spinbox = QSpinBox(self)
        self.neurons_per_layer_spinbox.setMinimum(1)
        self.neurons_per_layer_spinbox.setMaximum(200)
        self.neurons_per_layer_spinbox.setValue(50)
        layout.addRow("Neurona por capa:", self.neurons_per_layer_spinbox)

        # Configurar función de activación
        self.activation_function_combobox = QComboBox(self)
        self.activation_function_combobox.addItems(["Tanh", "ReLU", "Sigmoid"])
        layout.addRow("Función de activación:", self.activation_function_combobox)

        # Configurar Learning Rate con más precisión
        self.learning_rate_spinbox = QDoubleSpinBox(self)
        self.learning_rate_spinbox.setRange(1e-6, 1)  # Ajusta el rango según sea necesario
        self.learning_rate_spinbox.setDecimals(5)  # Establece la cantidad de decimales a 5
        self.learning_rate_spinbox.setSingleStep(1e-5)  # Ajusta el paso entre valores
        self.learning_rate_spinbox.setValue(1e-3)  # Establecer el valor por defecto a 1e-3
        layout.addRow("Learning Rate:", self.learning_rate_spinbox)

        # Configurar número de épocas
        self.num_epochs_spinbox = QSpinBox(self)
        self.num_epochs_spinbox.setMinimum(100)
        self.num_epochs_spinbox.setMaximum(100000)
        self.num_epochs_spinbox.setValue(1000)
        layout.addRow("Número de épocas:", self.num_epochs_spinbox)

        # Configurar optimizador
        self.optimizer_combobox = QComboBox(self)
        self.optimizer_combobox.addItems(["Adam", "SGD", "RMSprop"])
        layout.addRow("Seleccionar optimizador:", self.optimizer_combobox)

        # Botones de aceptar y cancelar
        self.accept_button = QPushButton("Aceptar", self)
        self.accept_button.clicked.connect(self.accept)
        self.reject_button = QPushButton("Cancelar", self)
        self.reject_button.clicked.connect(self.reject)
        layout.addRow(self.accept_button, self.reject_button)


class GaussianProcessConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Configuración del Proceso Gaussiano")
        layout = QFormLayout(self)

        # Configurar el kernel
        self.kernel_combobox = QComboBox(self)
        self.kernel_combobox.addItems(["RBF", "Matern"])
        layout.addRow("Seleccionar Kernel:", self.kernel_combobox)

        # Configurar Noise (nivel de ruido)
        self.noise_spinbox = QDoubleSpinBox(self)
        self.noise_spinbox.setRange(1e-6, 1)
        self.noise_spinbox.setDecimals(5)
        self.noise_spinbox.setValue(1e-2)
        layout.addRow("Nivel de Ruido (alpha):", self.noise_spinbox)

        # Configurar optimizador (en lugar de Sí/No, usar valores correctos)
        self.optimizer_combobox = QComboBox(self)
        self.optimizer_combobox.addItems(["fmin_l_bfgs_b", "None"])
        layout.addRow("¿Optimizar kernel?", self.optimizer_combobox)

        # Botones de aceptar y cancelar
        self.accept_button = QPushButton("Aceptar", self)
        self.accept_button.clicked.connect(self.accept)
        self.reject_button = QPushButton("Cancelar", self)
        self.reject_button.clicked.connect(self.reject)
        layout.addRow(self.accept_button, self.reject_button)

    def get_config(self):
        """Retorna la configuración seleccionada del cuadro de diálogo."""
        kernel = self.kernel_combobox.currentText()
        noise = self.noise_spinbox.value()
        optimizer = self.optimizer_combobox.currentText()
        # Convertir "None" a None (valor nulo de Python)
        if optimizer == "None":
            optimizer = None
        return kernel, noise, optimizer
