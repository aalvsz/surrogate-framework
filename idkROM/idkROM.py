import numpy as np
from abc import ABC, abstractmethod
import os
from idkROM.loader.import_data import DataLoader
from idkROM.pre.preprocessing import Pre
from idkROM.visualization.metrics import ErrorMetrics



class idkROM(ABC):

    def __init__(self, random_state=None, config_yml_path=None):
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_yml_path = os.path.join(base_path, "config.yml")
        print(f"Este es el path del yaml {config_yml_path}.")
        self.config_yml_path = config_yml_path
        self.random_state = random_state
        self.config_yml_path = config_yml_path
        self.config_dict = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.eval_metrics = None
        self.model = None
        self.input_scaler = None
        self.output_scaler = None

    """Clase base abstracta
    Sirve de plantilla para las subclases de cada método ROM, donde se implementan las funciones train y evaluate dedicadas."""
    class Modelo(ABC):
        def __init__(self, rom_config, random_state):
            self.rom_config = rom_config
            self.random_state = random_state

        @abstractmethod
        def train(self, X_train, y_train, X_val, y_val):
            """Método a sobrescribir en cada subclase"""
            raise NotImplementedError
        
        @abstractmethod
        def predict(self, X_test):
            """Método a sobrescribir en cada subclase"""
            raise NotImplementedError
        
        @abstractmethod
        def idk_run(self, X_train, y_train, X_val, y_val, X_test):
            """Método a sobrescribir en cada subclase"""
            raise NotImplementedError

        
    def load(self, loader, config_yml):

        config_dict = loader.read_yml(os.path.join(os.getcwd(), config_yml))

        inputs_file_path = config_dict['data inputs']
        outputs_file_path = config_dict['data outputs']
        data_source = config_dict['read mode']
        validation_mode = config_dict['validation mode']
        imputer = config_dict['imputer']
        scaler = config_dict['scaler']
        filter_method = config_dict['filter method']
        test_set_size = config_dict['test size']
        validation_set_size = config_dict['validation size']
        model_name = config_dict['model type']
        hyperparams = config_dict['hyperparams']
        mode = config_dict['mode']
        eval_metrics = config_dict['eval metrics']
        default_params = loader.default_params

        if data_source == "raw":
            ##### CARGAR LOS DATOS
            inputs_df, outputs_df = loader.load_data(input_path=inputs_file_path, output_path=outputs_file_path, data_source="raw")
            print(f"Datos cargados.")

            ##### PREPROCESAMIENTO
            preprocessor = Pre()
            data_after_split, self.input_scaler, self.output_scaler = preprocessor.pre_process_data(inputs_df, outputs_df, test_set_size, validation_set_size, imputer, scaler, filter_method, self.random_state)

        else: 
            # Cargamos los datos preprocesados
            X_train, y_train, X_val, y_val, X_test, y_test = loader.load_data(output_path=outputs_file_path, data_source="pre")
            data_after_split = [X_train, y_train, X_test, y_test, X_val, y_val]
            print(f"Datos preprocesados cargados.")

        rom_config = {
        'validation_mode': validation_mode,    
        'input_dim': data_after_split[0].shape[1],
        'output_dim': data_after_split[1].shape[1],
        'hyperparams': hyperparams if mode in ['best', 'manual'] else default_params[model_name],
        'mode': mode,
        'model_name': model_name,
        'eval_metrics': eval_metrics
        }

        print(f"\nConfiguracion del ROM: {rom_config}")
        return rom_config, data_after_split


    def create_model(self, rom_config, data_after_split):
        """
        Instancia y entrena el modelo ROM utilizando la fábrica de modelos.
        """
        model_name = rom_config['model_name']
        validation_mode = rom_config['validation_mode']

        if model_name == "neural_network":
            from idkROM.models.nn_simplified import NeuralNetworkROM
            model = NeuralNetworkROM(rom_config, self.random_state)
        elif model_name == "gaussian_process":
            from idkROM.models.gaussian_process import GaussianProcessROM
            model = GaussianProcessROM(rom_config, self.random_state)
        elif model_name == "rbf":
            from idkROM.models.rbf import RBFROM
            model = RBFROM(rom_config, self.random_state)
        elif model_name == "response_surface":
            from idkROM.models.polynomial_response_surface import PolynomialResponseSurface
            model = PolynomialResponseSurface(rom_config, self.random_state)
        elif model_name == "svr":
            from idkROM.models.svr import SVRROM
            model = SVRROM(rom_config, self.random_state)
        else:
            raise ValueError(f"Modelo '{model_name}' no reconocido.")

        model.train(data_after_split[0], data_after_split[1], data_after_split[4], data_after_split[5], validation_mode) # X_train, y_train, X_val, y_val
        y_pred = model.predict(data_after_split[2]) # X_test

        return y_pred, model
    

    def evaluate(self, y_test:list, y_pred:list, rom_config:dict, output_scaler=None):
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

        if rom_config['eval_metrics'] == 'mse':
            # Calcular MSE en la escala normalizada
            mse_scaled = np.mean((y_pred_np - y_test_np)**2)
            mse_percentage = (mse_scaled / np.mean(np.abs(y_test_np))) * 100  # MSE en porcentaje
            print(f"MSE en escala normalizada: {mse_scaled:.4f}")
            print(f"MSE en porcentaje: {mse_percentage:.2f}%")
            metric = mse_scaled

        elif rom_config['eval_metrics'] == 'mae':
            # Calcular MAE normalizado
            mae_scaled = np.mean(np.abs(y_pred_np - y_test_np))
            mae_percentage = (mae_scaled / np.mean(np.abs(y_test_np))) * 100  # MAE en porcentaje
            print(f"MAE en escala normalizada: {mae_scaled:.4f}")
            print(f"MAE en porcentaje: {mae_percentage:.2f}%")
            metric = mae_scaled

        elif rom_config['eval_metrics'] == 'mape':
            # Calcular Mean Absolute Percentage Error (MAPE)
            # Añadir una pequeña constante para evitar la división por cero
            epsilon = 1e-10
            mape = np.mean(np.abs((y_test_np - y_pred_np) / (y_test_np + epsilon))) * 100
            print(f"MAPE: {mape:.2f}%")
            metric = mape

        # Create error visualization metrics
        errors = ErrorMetrics(self, rom_config, y_test, y_pred)
        errors.calculate_bic(y_test, y_pred)
        errors.create_error_graphs()

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

        # print(f"Este es el diccionario que se come el modelo: {rom_config}")
        metric = {'metric': metric}
        return metric


    def run(self, config_yml): # runear un modelo para obtener un simple predict
        loader = DataLoader()
        rom_config, data_after_split = self.load(loader, config_yml)
        y_pred, self.model = self.create_model(rom_config, data_after_split)
        metric = self.evaluate(data_after_split[3], y_pred, rom_config) # usamos la funcion evaluate de aqui, no la del modelo
        return metric


    def idk_run(self, dict_hyperparams): # runear un modelo con parametros arbitrarios
        loader = DataLoader()
        loader.actualizar_yaml(self.config_yml_path, dict_hyperparams) # actualizar el config.yml con los parametros de dict_hyperparams
        metric = self.run(self.config_yml_path)
        return metric