import os
from src.loader.import_data import DataLoader
from src.pre.preprocessing import Pre
from src.models.neural_network import NeuralNetworkROM
from src.models.gaussian_process import GaussianProcessROM
from src.models.rbf import RBFROM
from src.models.polynomial_response_surface import PolynomialResponseSurface
from src.models.svr import SVRROM


if __name__ == "__main__":

    random_state = 41

    loader = DataLoader()
    config_dict = loader.read_yml(os.getcwd() + "/config.yml")
    print(f"Diccionario de configuracion: {config_dict}")
    inputs_file_path = config_dict['data inputs']
    outputs_file_path = config_dict['data outputs']
    data_source = config_dict['read mode']
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
        scaler_type = 'minmax'
        filter_method = 'isolation_forest'
        (X_train_normalized, y_train_normalized,
                X_val_normalized, y_val_normalized,
                X_test_normalized, y_test_normalized) = preprocessor.pre_process_data(inputs_df, outputs_df, 0.7, 0.15, 0.15, scaler_type, filter_method, random_state)
        print(f"Datos preprocesados.")


    else: 
        # Cargamos los datos preprocesados
        loader = DataLoader()
        (X_train_normalized, y_train_normalized,
        X_val_normalized, y_val_normalized,
        X_test_normalized, y_test_normalized) = loader.load_data(output_path=outputs_file_path, data_source="pre")
        print(f"Datos preprocesados cargados.")

    rom_config = {
    'input_dim': X_train_normalized.shape[1],
    'output_dim': y_train_normalized.shape[1],
    'hyperparams': hyperparams if mode in ['best', 'manual'] else default_params[model_name],
    'mode': mode,
    'model_name': model_name
    }

    print(f"\nConfiguracion del ROM: {rom_config}")
    
    ##### SE CONFIGURA EL MODELO
    if model_name.lower() == "neural_network":
        # creamos un modelo de ROM solo con atributos, o sea sin crear una red neuronal
        model = NeuralNetworkROM(rom_config, random_state=random_state)
        #best_params = model.search_best_hyperparams(X_train_normalized, y_train_normalized, X_val_normalized, y_val_normalized, iterations=10, cv_folds=5, random_state=random_state)
        #rom_config['hyperparams'] = best_params	

    elif model_name.lower() == "gaussian_process":
        model = GaussianProcessROM(rom_config, random_state=random_state)
    
    elif model_name.lower() == "rbf":
        model = RBFROM(rom_config, random_state=random_state)

    elif model_name.lower() == "response_surface":
        model = PolynomialResponseSurface(rom_config, random_state=random_state)

    elif model_name.lower() == "svr":
        model = SVRROM(rom_config, random_state=random_state)
    
    ##### ENTRENAMIENTO
    model.train(X_train_normalized, y_train_normalized, X_val_normalized, y_val_normalized)

    ##### PREDICCIONES
    y_pred = model.predict(X_test_normalized)

    ##### EVALUACION
    model.evaluate(X_test_normalized, y_test_normalized, y_pred, eval_metrics)
    
        