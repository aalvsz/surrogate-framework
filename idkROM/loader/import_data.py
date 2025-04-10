import os
import pandas as pd
import yaml

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
        validation_mode = preprocess.get('validation mode', 'cross')
        imputer = preprocess.get('imputer', 'simple')
        scaler = preprocess.get('scaler', 'minmax')
        filter_method = preprocess.get('filter method', 'isolation forest')
        validation = preprocess.get('validation', 'single')
        test_size = preprocess.get('test size', 0.15)
        validation_size = preprocess.get('validation size', 0)

        # Modelo
        model_config = config['model']
        model_type = model_config['type']
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
            'validation mode': validation_mode,
            'imputer': imputer,
            'scaler': scaler,
            'filter method': filter_method,
            'validation': validation,
            'test size': test_size,
            'validation size': validation_size,
            'model type': model_type,
            'mode': mode,
            'hyperparams': hyper_params,
            'eval metrics': metrics,
            'plot': plot,
            'save': save
        }

    def load_data(self, input_path=None, output_path=None, data_source="raw"):
        

        if data_source.lower() == "raw":
            # Nos quedamos con el directorio (sin el nombre del archivo)
            self.results_path = os.path.dirname(os.path.dirname(input_path))
            self.results_path = os.path.join(self.results_path, "results")
            
            # Leer datos usando coma como separador decimal
            df_inputs = pd.read_csv(input_path, delimiter=",", decimal=".") 
            df_outputs = pd.read_csv(output_path, delimiter=",", decimal=".")
            
            return df_inputs, df_outputs

        else:
            self.results_path = os.path.dirname(os.path.dirname(output_path))
            self.results_path = os.path.join(self.results_path, "results")

            # Asumimos que los CSV ya contienen los datos preprocesados
            X_train = pd.read_csv(os.path.join(self.results_path, 'X_train.csv'), sep=",")
            y_train = pd.read_csv(os.path.join(self.results_path, 'y_train.csv'), sep=",")
            X_val   = pd.read_csv(os.path.join(self.results_path, 'X_val.csv'), sep=",")
            y_val   = pd.read_csv(os.path.join(self.results_path, 'y_val.csv'), sep=",")
            X_test  = pd.read_csv(os.path.join(self.results_path, 'X_test.csv'), sep=",")
            y_test  = pd.read_csv(os.path.join(self.results_path, 'y_test.csv'), sep=",")
            
            return X_train, y_train, X_val, y_val, X_test, y_test


    def actualizar_yaml(self, config_yml_path, dict_hyperparams):
        """
        Actualiza un archivo YAML con los valores proporcionados en dict_hyperparams,
        siguiendo las rutas especificadas en la sección idk_params del YAML.
        Preserva comentarios y formato original del archivo.
        
        Args:
            config_yml_path (str): Ruta al archivo YAML a modificar
            dict_hyperparams (dict): Diccionario con los parámetros a actualizar
            
        Returns:
            None: El archivo YAML se modifica directamente
        """
        # Verificar que el archivo existe
        if not os.path.exists(config_yml_path):
            raise FileNotFoundError(f"El archivo {config_yml_path} no existe")
        
        # Cargar el archivo YAML preservando comentarios
        with open(config_yml_path, 'r', encoding='utf-8') as file:
            # Usamos ruamel.yaml que preserva comentarios y formato
            from ruamel.yaml import YAML
            yaml_parser = YAML()
            yaml_parser.preserve_quotes = True
            yaml_parser.indent(mapping=2, sequence=4, offset=2)
            config = yaml_parser.load(file)
        
        # Verificar que existe la sección idk_params
        if 'idk_params' not in config:
            raise KeyError("La sección 'idk_params' no existe en el archivo YAML")
        
        # Iterar sobre los parámetros a modificar
        for param, value in dict_hyperparams.items():
            # Verificar que el parámetro existe en idk_params
            if param not in config['idk_params']:
                print(f"Advertencia: El parámetro '{param}' no está definido en idk_params y será ignorado")
                continue
            
            # Obtener la ruta para el parámetro
            param_path = config['idk_params'][param]
            
            # Navegar a través de la ruta en el YAML
            target = config
            for i, key in enumerate(param_path):
                # Verificar que todas las claves existen en el camino
                if key not in target:
                    print(f"Advertencia: La clave '{key}' no existe en la ruta {param_path[:i+1]}")
                    break
                
                # Si es el último elemento, actualizamos el valor
                if i == len(param_path) - 1:
                    target[key] = value
                else:
                    # Si no es el último, seguimos navegando
                    target = target[key]
        
        # Guardar los cambios en el archivo YAML preservando comentarios y formato
        with open(config_yml_path, 'w', encoding='utf-8') as file:
            yaml_parser.dump(config, file)
        
        print(f"Archivo {config_yml_path} actualizado correctamente preservando comentarios y formato")

        
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