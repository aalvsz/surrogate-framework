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
    """
    Clase base para modelos ROM (Reduced Order Model) en el framework idkROM.

    Proporciona métodos para cargar datos, preprocesar, entrenar, predecir y evaluar modelos.
    Sirve como plantilla para subclases específicas de cada tipo de modelo ROM.
    """

    def __init__(self, random_state=None, config_yml_path=None, data_dict=None):
        """
        Inicializa la clase base idkROM.

        Args:
            random_state (int, optional): Semilla para reproducibilidad.
            config_yml_path (str, optional): Ruta al archivo de configuración YAML.
            data_dict (dict, optional): Diccionario de configuración de datos.
        """
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_yml_path = config_yml_path if config_yml_path is not None else os.path.join(base_path, "config.yml")
        print(f"Este es el path del yaml {self.config_yml_path}.")
        self.random_state = random_state
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

    class Modelo(ABC):
        """
        Clase abstracta interna para definir la interfaz de los modelos ROM concretos.
        """

        def __init__(self, rom_config, random_state):
            """
            Inicializa el modelo base.

            Args:
                rom_config (dict): Configuración del modelo ROM.
                random_state (int): Semilla para reproducibilidad.
            """
            self.rom_config = rom_config
            self.random_state = random_state

        @abstractmethod
        def train(self, X_train, y_train, X_val, y_val):
            """
            Método abstracto para entrenar el modelo.

            Args:
                X_train: Datos de entrenamiento.
                y_train: Etiquetas de entrenamiento.
                X_val: Datos de validación.
                y_val: Etiquetas de validación.
            """
            raise NotImplementedError

        @abstractmethod
        def predict(self, X_test):
            """
            Método abstracto para predecir con el modelo.

            Args:
                X_test: Datos de prueba.
            """
            raise NotImplementedError

        @abstractmethod
        def idk_run(self, X_train, y_train, X_val, y_val, X_test):
            """
            Método abstracto para ejecutar el modelo de forma personalizada.

            Args:
                X_train: Datos de entrenamiento.
                y_train: Etiquetas de entrenamiento.
                X_val: Datos de validación.
                y_val: Etiquetas de validación.
                X_test: Datos de prueba.
            """
            raise NotImplementedError

    def load(self, loader, config_yml=None):
        """
        Carga y preprocesa los datos según la configuración YAML o diccionario.

        Args:
            loader (DataLoader): Objeto para cargar datos.
            config_yml (str, optional): Ruta al archivo YAML de configuración.

        Returns:
            tuple: (rom_config, data_after_split)
                rom_config (dict): Configuración del modelo ROM.
                data_after_split (list): Datos procesados y divididos.
        """
        config_dict = loader.read_yml(self.config_yml_path)
        inputs_file_path = config_dict['data inputs']
        outputs_file_path = config_dict['data outputs']
        output_folder = config_dict['output folder']
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

        self.output_folder = output_folder

        if data_source == "raw":
            # 1. Cargar datos
            inputs_df, outputs_df = loader.load_data(
                inputs_path=inputs_file_path,
                outputs_path=outputs_file_path,
                data_source="raw"
            )
            print(f"Datos cargados.")
            
            output_scalers = {
                col: 'minmax' if col.startswith('TCP') else scaler
                for col in outputs_df.columns
            }

            preprocessor = Pre(model_name, self.output_folder)

            # 2. Preprocesar con escalers individuales si hay
            data_after_split, self.input_scaler, self.output_scaler, self.output_folder = preprocessor.pre_process_data(
                inputs_df, outputs_df, discrete_inputs,
                test_set_size, validation_set_size,
                imputer, scaler, filter_method,
                self.random_state,
                output_scalers=output_scalers
            )

        else:
            # 3. Si viene de preprocesado
            X_train, y_train, X_val, y_val, X_test, y_test = loader.load_data(
                inputs_path=preprocessed_data_path,
                data_source="pre"
            )
            data_after_split = [X_train, y_train, X_test, y_test, X_val, y_val]
            print(f"Datos preprocesados cargados.")

            # Si cargas datos preprocesados, asegúrate de cargar el dict de scalers también:
            scaler_path = os.path.join(self.output_folder, "output_scaler.pkl")
            import joblib
            self.output_scaler = joblib.load(scaler_path)

        # Crear config ROM
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
        Instancia y entrena el modelo ROM utilizando la configuración proporcionada.

        Args:
            rom_config (dict): Configuración del modelo ROM.
            data_after_split (list): Datos procesados y divididos.
            only_train (bool): Si es True, solo entrena sin predecir.

        Returns:
            tuple: (y_pred, model)
                y_pred: Predicciones del modelo sobre el conjunto de prueba.
                model: Instancia del modelo entrenado.
        """
        model_name = rom_config['model_name']
        validation_mode = rom_config['validation_mode']

        if model_name == "neural_network":
            from idkrom.architectures.neural_network import NeuralNetworkROM
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
        Evalúa el modelo y guarda métricas y errores en archivos.

        Args:
            y_test_df (pd.DataFrame): Valores verdaderos del conjunto de prueba.
            y_pred_arr (np.ndarray): Predicciones del modelo.
            rom_config (dict): Configuración del modelo ROM.

        Returns:
            dict: Diccionario con la métrica principal calculada.
        """
        # Convert to numpy
        y_test_np = y_test_df.to_numpy()
        y_pred_np = np.array(y_pred_arr)

        results = {}

        # ========================
        # Metrics on scaled data
        # ========================
        if rom_config['eval_metrics'] == 'mse':
            mse_scaled = float(np.mean((y_pred_np - y_test_np)**2))
            results['MSE_scaled'] = mse_scaled
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
        y_test_orig = []
        y_pred_orig = []

        for i, col in enumerate(y_test_df.columns):
            scaler = self.output_scaler[col]
            y_test_orig.append(scaler.inverse_transform(y_test_np[:, i].reshape(-1, 1)).flatten())
            y_pred_orig.append(scaler.inverse_transform(y_pred_np[:, i].reshape(-1, 1)).flatten())

        y_test_orig = np.vstack(y_test_orig).T
        y_pred_orig = np.vstack(y_pred_orig).T

        # Guardar CSV con predicciones
        pred_df = pd.DataFrame(y_pred_orig, columns=y_test_df.columns)
        pred_df.to_csv(os.path.join(self.output_folder, "predicciones_test.csv"), index=False)

        # Guardar CSV con valores esperados
        true_df = pd.DataFrame(y_test_orig, columns=y_test_df.columns)
        true_df.to_csv(os.path.join(self.output_folder, "valores_esperados_test.csv"), index=False)

        print(f"Predicciones guardadas en {os.path.join(self.output_folder, 'predicciones_test.csv')}")
        print(f"Valores esperados guardados en {os.path.join(self.output_folder, 'valores_esperados_test.csv')}")

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
        # Save detailed errors to TXT (with means)
        # ========================
        def format_num(num, threshold_low=1e-4, threshold_high=1e4, decimals=6):
            """
            Formatea num en notación científica si es mayor que threshold_high o menor que threshold_low.

            Args:
                num (float): Número a formatear.
                threshold_low (float): Umbral inferior para notación científica.
                threshold_high (float): Umbral superior para notación científica.
                decimals (int): Decimales a mostrar.

            Returns:
                str: Número formateado.
            """
            if abs(num) != 0 and (abs(num) < threshold_low or abs(num) > threshold_high):
                return f"{num:.{decimals}e}"
            return f"{num:.{decimals}f}"

        txt_path = os.path.join(self.output_folder, "predictions_errors.txt")

        with open(txt_path, "w", encoding="utf-8") as f:
            mean_error_line = "Promedio: "
            mean_parts = []

            # Calcular promedios
            for j, col in enumerate(y_test_df.columns):
                errors = y_pred_orig[:, j] - y_test_orig[:, j]
                errors_pct = errors / (y_test_orig[:, j] + 1e-10) * 100

                mean_error = np.mean(np.abs(errors))
                mean_error_pct = np.mean(np.abs(errors_pct))

                mean_parts.append(
                    f"{col} -> error={format_num(mean_error)}, error%={format_num(mean_error_pct)}%"
                )

            mean_error_line += " | ".join(mean_parts)
            f.write(mean_error_line + "\n\n")

            # Por fila
            for i in range(len(y_test_orig)):
                line = f"fila {i}: "
                parts = []
                for j, col in enumerate(y_test_df.columns):
                    true_val = y_test_orig[i, j]
                    pred_val = y_pred_orig[i, j]
                    error = pred_val - true_val
                    error_pct = (error / (true_val + 1e-10)) * 100

                    parts.append(
                        f"{col} -> esperado={format_num(true_val)}, "
                        f"predicho={format_num(pred_val)}, "
                        f"error={format_num(error)}, "
                        f"error%={format_num(error_pct)}%"
                    )
                line += " | ".join(parts)
                f.write(line + "\n")

        print(f"Errores individuales guardados en {txt_path}")

        # ========================
        # Error graphs 
        # ========================
        errors = ErrorMetrics(self, rom_config, y_test=pd.DataFrame(y_test_orig, columns=y_test_df.columns),
                               y_pred=y_pred_orig,
                                 y_test_scaled=y_test_np,
                                   y_pred_scaled=y_pred_np)
        try:
            errors.create_error_graphs()
        except Exception as e:
            print(f"Error creating error graphs: {e}")

        # ========================
        # Save metrics JSON
        # ========================
        output_path = os.path.join(self.output_folder, "metrics_results.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)

        print(f"Métricas guardadas en {output_path}")

        return {'metric': results.get('MSE_scaled', None) if rom_config['eval_metrics']=='mse' else results.get('MAE_scaled', None)}

    def run(self, config_yml):
        """
        Ejecuta el pipeline completo: carga, entrenamiento, predicción y evaluación.

        Args:
            config_yml (str): Ruta al archivo YAML de configuración.

        Returns:
            dict: Métrica principal obtenida tras la evaluación.
        """
        loader = DataLoader()
        rom_config, data_after_split = self.load(loader, config_yml)
        y_pred, self.model = self.train_and_predict(rom_config, data_after_split)
        metric = self.evaluate(data_after_split[3], y_pred, rom_config) # usamos la funcion evaluate de aqui, no la del modelo

        return metric

    def idk_run(self, dict_hyperparams=None):
        """
        Ejecuta el pipeline completo con hiperparámetros arbitrarios.

        Args:
            dict_hyperparams (dict, optional): Diccionario de hiperparámetros para actualizar el YAML.

        Returns:
            dict: Métrica principal obtenida tras la evaluación.
        """
        if dict_hyperparams is not None:
            loader = DataLoader()
            loader.actualizar_yaml(self.config_yml_path, dict_hyperparams) # actualizar el config.yml con los parametros de dict_hyperparams
        metric = self.run(self.config_yml_path)

        return metric