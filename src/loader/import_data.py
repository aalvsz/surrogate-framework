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
        pass

    def read_yml(self, config_path):
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
        raw = preprocess.get('raw', True)
        scaler = preprocess.get('scaler', 'StandardScaler')
        validation = preprocess.get('validation', 'single')
        test_size = preprocess.get('test_size', 0.15)

        # Modelo
        model_config = config['model']
        model_type = model_config['type']
        hyper_params_config = model_config['hyper_params']
        mode = hyper_params_config['mode']

        # Definir hiperparámetros por defecto
        default_params = {
            'NN': {
                'n_layers': 2,
                'n_neurons': 10,
                'activation': 'relu',
                'optimizer': 'adam',
                'loss': 'mse',
                'metrics': 'mae',
                'epochs': 100
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

        # Crear diccionario de hiperparámetros
        if mode == 'manual':
            hyper_params = hyper_params_config['params']
        elif mode == 'best':
            hyper_params = {k: v for k, v in hyper_params_config['params'].items() if isinstance(v, list)}
        else:  # Default
            hyper_params = default_params.get(model_type.lower(), {})

        # Evaluación
        evaluate = config['evaluate']
        eval_metrics = evaluate.get('metrics', 'mae')
        plot = evaluate.get('plot', False)
        save = evaluate.get('save', False)

        return {
            'data_input': data_input,
            'data_output': data_output,
            'raw': raw,
            'scaler': scaler,
            'validation': validation,
            'test_size': test_size,
            'model_type': model_type,
            'mode': mode,
            'hyper_params': hyper_params,
            'eval_metrics': eval_metrics,
            'plot': plot,
            'save': save
        }



    def load_data(self, file_path, data_source="raw"):

        # nos quedamos con el directorio (sin el nombre del archivo)
        self.results_path = os.path.dirname(os.path.dirname(file_path))
        self.results_path = os.path.join(self.results_path, "results")

        if data_source.lower() == "raw":
            # Leer datos con delimitador y decimal apropiado
            df = pd.read_csv(file_path, delimiter=",", decimal=".")

            return df

        else:
            # Asumimos que los CSV ya contienen los datos preprocesados
            X_train = pd.read_csv(os.path.join(self.results_path, 'X_train.csv'), sep=",")
            y_train = pd.read_csv(os.path.join(self.results_path, 'y_train.csv'), sep=",")
            X_val   = pd.read_csv(os.path.join(self.results_path, 'X_val.csv'), sep=",")
            y_val   = pd.read_csv(os.path.join(self.results_path, 'y_val.csv'), sep=",")
            X_test  = pd.read_csv(os.path.join(self.results_path, 'X_test.csv'), sep=",")
            y_test  = pd.read_csv(os.path.join(self.results_path, 'y_test.csv'), sep=",")

            return X_train, y_train, X_val, y_val, X_test, y_test

        
if __name__ == "__main__":
    # Ruta al archivo YAML de prueba
    config_path = "main.yml"  # Actualiza esta ruta con la ubicación de tu archivo YAML de prueba

    # Inicializar DataLoader y leer el archivo YAML
    data_loader = DataLoader()
    config_dict = data_loader.read_yml(config_path)

    # Imprimir el diccionario resultante
    print(config_dict)
    print(f"Diccionario de hiperparametros: {config_dict['hyper_params']}")