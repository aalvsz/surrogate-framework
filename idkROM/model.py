import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import os
import json
from idkrom.loader.import_data import DataLoader
from idkrom.pre.preprocessing import Pre
from idkrom.utils.metrics import ErrorMetrics
from idkrom.utils.save_model import save_rom_instance



class idkROM(ABC):

    def __init__(self, random_state=None, config_yml_path=None, data_dict=None):
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        config_yml_path = os.path.join(base_path, "config.yml")
        print(f"Este es el path del yaml {config_yml_path}.")

        self.config_yml_path = config_yml_path
        self.random_state = random_state
        self.config_yml_path = config_yml_path
        self.data_dict = data_dict

        self.config_dict = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.eval_metrics = None
        self.model = None
        self.input_scaler = None
        self.output_scaler = None
        self.output_folder = None

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

        
    def load(self, loader, config_yml=None):

        
        config_dict = loader.read_yml(os.path.join(os.getcwd(), config_yml))

        inputs_file_path = config_dict['data inputs']
        outputs_file_path = config_dict['data outputs']
        data_source = config_dict['read mode']
        preprocessed_data_path = config_dict['preprocessed data path']
        validation_mode = config_dict['validation mode']
        imputer = config_dict['imputer']
        scaler = config_dict['scaler']
        scaler_path = config_dict['scaler path']
        filter_method = config_dict['filter method']
        test_set_size = config_dict['test size']
        validation_set_size = config_dict['validation size']
        model_name = config_dict['model type']
        hyperparams = config_dict['hyperparams']
        discrete_inputs = config_dict['discrete inputs']
        mode = config_dict['mode']
        eval_metrics = config_dict['eval metrics']
        default_params = loader.default_params

        preprocessor = Pre(model_name)
        self.output_folder = preprocessor.output_folder

        if data_source == "raw":
            ##### CARGAR LOS DATOS
            inputs_df, outputs_df = loader.load_data(inputs_path=inputs_file_path, outputs_path=outputs_file_path, data_source="raw")
            print(f"Datos cargados.")

            ##### PREPROCESAMIENTO
            data_after_split, self.input_scaler, self.output_scaler = preprocessor.pre_process_data(
                            inputs_df, outputs_df, discrete_inputs,
                            test_set_size, validation_set_size,
                            imputer, scaler, filter_method, self.random_state
                        )
        else: 
            # Cargamos los datos preprocesados
            X_train, y_train, X_val, y_val, X_test, y_test = loader.load_data(inputs_path=preprocessed_data_path, data_source="pre")
            data_after_split = [X_train, y_train, X_test, y_test, X_val, y_val]
            print(f"Datos preprocesados cargados.")

        rom_config = {
        'validation_mode': validation_mode,    
        'input_dim': data_after_split[0].shape[1],
        'output_dim': data_after_split[1].shape[1],
        'discrete_inputs': discrete_inputs,
        'hyperparams': hyperparams if mode in ['best', 'manual'] else default_params[model_name],
        'mode': mode,
        'model_name': model_name,
        'output_folder': self.output_folder,
        'eval_metrics': eval_metrics
        }

        print(f"\nConfiguracion del ROM: {rom_config}")
        return rom_config, data_after_split


    def train_and_predict(self, rom_config, data_after_split, only_train=False):
        """
        Instancia y entrena el modelo ROM utilizando la fábrica de modelos.
        """
        model_name = rom_config['model_name']
        validation_mode = rom_config['validation_mode']

        if model_name == "neural_network":
            from idkrom.architectures.nn_simplified import NeuralNetworkROM
            model = NeuralNetworkROM(rom_config, self.random_state)
        elif model_name == "gaussian_process":
            from idkrom.architectures.gaussian_process import GaussianProcessROM
            model = GaussianProcessROM(rom_config, self.random_state)
        elif model_name == "rbf":
            from idkrom.architectures.rbf import RBFROM
            model = RBFROM(rom_config, self.random_state)
        elif model_name == "response_surface":
            from idkrom.architectures.polynomial_response_surface import PolynomialResponseSurface
            model = PolynomialResponseSurface(rom_config, self.random_state)
        elif model_name == "svr":
            from idkrom.architectures.svr import SVRROM
            model = SVRROM(rom_config, self.random_state)
        else:
            raise ValueError(f"Modelo '{model_name}' no reconocido.")

        model.train(data_after_split[0], data_after_split[1], data_after_split[4], data_after_split[5], validation_mode) # X_train, y_train, X_val, y_val
        save_rom_instance(model, rom_config)
        y_pred = model.predict(data_after_split[2]) # X_test

        return y_pred, model
    


    def evaluate(self, y_test_df, y_pred_arr, rom_config: dict):
        """
        Evaluates the model and saves metrics to a JSON file.
        """
        # Convert inputs to numpy arrays
        y_test_np = y_test_df.to_numpy()
        y_pred_np = np.array(y_pred_arr)

        results = {}

        # ========================
        # Metrics on scaled data
        # ========================
        if rom_config['eval_metrics'] == 'mse':
            mse_scaled = float(np.mean((y_pred_np - y_test_np)**2))
            results['MSE_scaled'] = mse_scaled

            # Per-column scaled MSE
            mse_per_col_scaled = (np.mean((y_pred_np - y_test_np)**2, axis=0)).tolist()
            results['MSE_scaled_per_col'] = dict(zip(y_test_df.columns, mse_per_col_scaled))

        elif rom_config['eval_metrics'] == 'mae':
            mae_scaled = float(np.mean(np.abs(y_pred_np - y_test_np)))
            results['MAE_scaled'] = mae_scaled

        elif rom_config['eval_metrics'] == 'mape':
            eps = 1e-10
            mape_scaled = float(np.mean(np.abs((y_test_np - y_pred_np)/(y_test_np + eps))) * 100)
            results['MAPE_scaled'] = mape_scaled
        else:
            raise ValueError(f"Unknown metric {rom_config['eval_metrics']}")

        # ========================
        # Metrics on original data
        # ========================
        y_test_orig = self.output_scaler.inverse_transform(y_test_np)
        y_pred_orig = self.output_scaler.inverse_transform(y_pred_np)

        if rom_config['eval_metrics'] == 'mse':
            mse_orig = float(np.mean((y_pred_orig - y_test_orig)**2))
            results['MSE_original'] = mse_orig

            mse_per_col_orig = (np.mean((y_pred_orig - y_test_orig)**2, axis=0)).tolist()
            results['MSE_original_per_col'] = dict(zip(y_test_df.columns, mse_per_col_orig))

        elif rom_config['eval_metrics'] == 'mae':
            mae_orig = float(np.mean(np.abs(y_pred_orig - y_test_orig)))
            results['MAE_original'] = mae_orig

        elif rom_config['eval_metrics'] == 'mape':
            eps = 1e-10
            mape_orig = float(np.mean(np.abs((y_test_orig - y_pred_orig)/(y_test_orig + eps))) * 100)
            results['MAPE_original'] = mape_orig

        # ========================
        # Error graphs & BIC
        # ========================
        errors = ErrorMetrics(self, rom_config, pd.DataFrame(y_test_np, columns=y_test_df.columns), y_pred_np)
        bic_val = errors.calculate_bic(pd.DataFrame(y_test_np, columns=y_test_df.columns), y_pred_np)
        results['BIC'] = float(bic_val)

        try:
            errors.create_error_graphs()
        except Exception as e:
            print(f"Error creating error graphs: {e}")

        # ========================
        # Save results to JSON
        # ========================
        output_path = os.path.join(self.output_folder, "metrics_results.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)

        print(f"Métricas guardadas en {output_path}")

        return {'metric': results.get('MSE_scaled', None) if rom_config['eval_metrics']=='mse' else results.get('MAE_scaled', None)}



    def run(self, config_yml): # runear un modelo para obtener un simple predict
        loader = DataLoader()
        rom_config, data_after_split = self.load(loader, config_yml)
        y_pred, self.model = self.train_and_predict(rom_config, data_after_split)
        metric = self.evaluate(data_after_split[3], y_pred, rom_config) # usamos la funcion evaluate de aqui, no la del modelo

        return metric


    def idk_run(self, dict_hyperparams=None): # runear un modelo con parametros arbitrarios

        if dict_hyperparams is not None:
            loader = DataLoader()
            loader.actualizar_yaml(self.config_yml_path, dict_hyperparams) # actualizar el config.yml con los parametros de dict_hyperparams
        metric = self.run(self.config_yml_path)

        return metric
    
    
    def rom_training_pipeline(self, data):
        # importar datos
        # preprocesamiento datos
        # crear modelo con hiperparametros
        # entrenar modelo
        # guardar modelo
        loader = DataLoader()
    
        loader.actualizar_yaml(self.config_yml_path, data)
        rom_config, data_after_split = self.load(loader, self.config_yml_path)
        model_path = self.create_model(rom_config, data_after_split, only_train=True)
        return model_path