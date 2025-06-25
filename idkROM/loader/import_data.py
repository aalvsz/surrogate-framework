import os
import pandas as pd
import yaml
from ruamel.yaml import YAML

class DataLoader:
    """
    Clase encargada de importar los datos desde un archivo CSV.
    
    Permite cargar datos en dos formatos:
      - 'raw': datos crudos que serán preprocesados (se aplica normalización, eliminación de outliers, etc.).
      - 'processed': datos ya preprocesados.
    """
    def __init__(self):
        self.results_path = None
        self.default_params = None
        self.config = None
        pass

    def read_yml(self, config_path: str):
        """
        Lee el archivo YAML de configuración y devuelve las variables asociadas.
        """
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Extraer configuración general
        data_input = config['data']['input']
        data_output = config['data']['output']

        # Preprocesamiento
        preprocess = config['preprocess']
        read_mode = preprocess.get('read mode', 'raw')
        preprocessed_data_path = preprocess.get('processed data path', None)
        validation_mode = preprocess.get('validation mode', 'cross')
        imputer = preprocess.get('imputer', 'simple')
        scaler = preprocess.get('scaler', 'minmax')
        scaler_path = os.path.join(os.getcwd(), preprocess.get('scaler path', None))
        filter_method = preprocess.get('filter method', 'isolation forest')
        validation = preprocess.get('validation', 'single')
        test_size = preprocess.get('test size', 0.15)
        validation_size = preprocess.get('validation size', 0)

        # Modelo
        model_config = config['model']
        model_type = model_config['type']
        discrete_inputs = model_config['discrete inputs']
        hyper_params_config = model_config['hyperparams']
        mode = hyper_params_config['mode']

        # Evaluacion
        evaluate = config['evaluate']
        metrics = evaluate.get('metrics', 'mse')
        plot = evaluate.get('plot', 'True')
        save = evaluate.get('save', 'True')


        # Definir hiperparámetros por defecto
        self.default_params = {
            'neural_network': {
                'n_layers': 2,
                'n_neurons': 10,
                'activation': 'Tanh',
                'optimizer': 'Adam',
                'learning_rate': 0.001,
                'lr_step': 500,
                'lr_decrease_rate': 0.1,
                #'loss': 'mse',
                'epochs': 1000,
                'batch_size': 32,
                'patience': 50,
                'cv_folds': 5,
                'convergence_threshold': 1e-5
            },
            'gaussian_process': {
                'kernel': 'RBF',
                'noise': 1e-3
            },
            'rbf': {
                'gamma': 1.0
            },
            'response_surface': {
                'degree': 2
            },
            'svr': {
                'kernel': 'rbf',
                'C': 1.0,
                'epsilon': 0.1
            }
        }

        # Función para convertir una cadena con opciones separadas por '|' en una lista
        def parse_options(value):
            if isinstance(value, str):
                if '|' in value:
                    options = value.split('|')
                    # Intentar convertir los valores en enteros o flotantes si es posible
                    return [float(option.strip()) if '.' in option else int(option.strip()) if option.strip().isdigit() else option.strip() for option in options]
                elif value.replace('.', '', 1).isdigit():  # Si es un número (entero o flotante)
                    return float(value) if '.' in value else int(value)
            return value

        # Crear diccionario de hiperparámetros
        if mode == 'manual':
            hyper_params = hyper_params_config['params']
        elif mode == 'best':
            hyper_params = hyper_params_config['params']
            # Include n_iter and cv only if mode is 'best'
            hyper_params['n_iter'] = hyper_params_config['params'].get('n_iter', 10)
            hyper_params['cv'] = hyper_params_config['params'].get('cv', 5)
        else:  # Default
            hyper_params = self.default_params.get(model_type.lower(), {})

        # Convertir las cadenas de listas en listas reales o números individuales
        for key, value in hyper_params.items():
            hyper_params[key] = parse_options(value)

        return {
            'data inputs': data_input,
            'data outputs': data_output,
            'read mode': read_mode,
            'preprocessed data path': preprocessed_data_path,
            'validation mode': validation_mode,
            'imputer': imputer,
            'scaler': scaler,
            'scaler path': scaler_path,
            'filter method': filter_method,
            'validation': validation,
            'test size': test_size,
            'validation size': validation_size,
            'model type': model_type,
            'mode': mode,
            'discrete inputs': discrete_inputs,
            'hyperparams': hyper_params,
            'eval metrics': metrics,
            'plot': plot,
            'save': save
        }


    def load_data(self, inputs_path=None, outputs_path=None, data_source="raw"):
        

        if data_source.lower() == "raw":
            # Nos quedamos con el directorio (sin el nombre del archivo)
            self.results_path = os.path.dirname(os.path.dirname(inputs_path))
            self.results_path = os.path.join(self.results_path, "results")
            
            # Leer datos usando coma como separador decimal
            df_inputs = pd.read_csv(inputs_path, delimiter=",", decimal=".") 
            df_outputs = pd.read_csv(outputs_path, delimiter=",", decimal=".")
            
            return df_inputs, df_outputs

        else:

            # Asumimos que los CSV ya contienen los datos preprocesados
            X_train = pd.read_parquet(os.path.join(inputs_path, 'X_train.parquet'), sep=",")
            y_train = pd.read_parquet(os.path.join(inputs_path, 'y_train.parquet'), sep=",")
            X_val   = pd.read_parquet(os.path.join(inputs_path, 'X_val.parquet'), sep=",")
            y_val   = pd.read_parquet(os.path.join(inputs_path, 'y_val.parquet'), sep=",")
            X_test  = pd.read_parquet(os.path.join(inputs_path, 'X_test.parquet'), sep=",")
            y_test  = pd.read_parquet(os.path.join(inputs_path, 'y_test.parquet'), sep=",")
            
            return X_train, y_train, X_val, y_val, X_test, y_test


    def actualizar_yaml(self, config_yml_path, dict_hyperparams):
        """
        Actualiza un archivo YAML con los valores proporcionados en dict_hyperparams,
        siguiendo las rutas especificadas en la sección idk_params del YAML.
        Preserva comentarios y formato original del archivo.
        """
        # 1) Verificar que el archivo existe
        if not os.path.exists(config_yml_path):
            raise FileNotFoundError(f"El archivo {config_yml_path} no existe")
        
        # 2) Cargar el YAML preservando comentarios
        yaml_parser = YAML()
        yaml_parser.preserve_quotes = True
        yaml_parser.indent(mapping=2, sequence=4, offset=2)
        with open(config_yml_path, 'r', encoding='utf-8') as f:
            config = yaml_parser.load(f)
        
        # 3) Aplanar dict_hyperparams (sacar todos los pares param: value)
        flat_params = {}
        def _flatten(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    _flatten(v)
                else:
                    flat_params[k] = v
        _flatten(dict_hyperparams)
        
        # 4) Verificar que existe la sección idk_params
        if 'idk_params' not in config:
            raise KeyError("La sección 'idk_params' no existe en el archivo YAML de destino.")
        
        # 5) Iterar sobre cada parámetro “plano”
        for param, value in flat_params.items():
            if param not in config['idk_params']:
                print(f"Advertencia: El parámetro '{param}' no está definido en idk_params y se ignora")
                continue
            
            # Obtener la ruta (lista de claves) donde debe escribirse
            param_path = config['idk_params'][param]
            
            # Navegar por el YAML hasta la clave final y asignar el valor
            target = config
            for idx, key in enumerate(param_path):
                if key not in target:
                    print(f"Advertencia: Clave '{key}' no existe en la ruta {param_path[:idx+1]}")
                    break
                
                # Si es el último paso de la ruta, actualizar
                if idx == len(param_path) - 1:
                    target[key] = value
                else:
                    target = target[key]
        
        # 6) Volver a escribir el archivo, preservando formato y comentarios
        with open(config_yml_path, 'w', encoding='utf-8') as f:
            yaml_parser.dump(config, f)
        
        print(f"Archivo {config_yml_path} actualizado correctamente preservando formato y comentarios")
            
if __name__ == "__main__":

    # TESTEAR LA LECTURA DEL YAML ################################################################################################################

    # Ruta al archivo YAML de prueba
    config_path = "config.yml"  # Actualiza esta ruta con la ubicación de tu archivo YAML de prueba

    # Inicializar DataLoader y leer el archivo YAML
    data_loader = DataLoader()
    config_dict = data_loader.read_yml(config_path)

    # Imprimir el diccionario resultante
    print(config_dict)
    print(f"\nDiccionario de hiperparámetros: {config_dict['hyperparams']}")
    print(f"Parametros por defecto: {data_loader.default_params}")
    print(f"Parametros por defecto Neural Network: {data_loader.default_params['neural_network']}")