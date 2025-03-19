import sys
import os
import joblib
from PyQt6.QtWidgets import QApplication
#from src.gui.main_window import ROMApp  # Asegúrate de que este archivo es el correcto
from src.loader.import_data import DataLoader
from src.pre.preprocessing import Pre
from src.models.neural_network import NeuralNetworkROM
from src.models.gaussian_process import GaussianProcessROM
from src.models.rbf import RBFROM
from src.models.polynomial_response_surface import PolynomialResponseSurface
from src.models.svr import SVRROM
from src.tools.search_hyperparams import search_best_hyperparameters

"""if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ROMApp()
    window.show()
    sys.exit(app.exec())"""


if __name__ == "__main__":

    import sys
    if len(sys.argv) < 4:
        print("Uso: python idkROM.py <ruta_csv> <tipo_datos: raw/processed> <modelo: neural_network/gaussian_process>")
        sys.exit(1)

    file_path = sys.argv[1]
    data_source = sys.argv[2] #raw or processed 
    model_name = sys.argv[3] #neural_network or gaussian_process

    # indice de la columna del ultimo input
    last_input_var = 7

    if data_source == "raw":
        # Cargamos los datos y extraemos df con inputs y outputs
        loader = DataLoader()
        df = loader.load_data(file_path, data_source)
        print(f"Datos cargados.")

        # Normalizamos los inputs y outputs
        pre_processor = Pre()
        X_train, y_train, X_val, y_val, X_test, y_test = pre_processor.split_dataset(df, last_input_var)
        (X_train_normalized, y_train_normalized,
        X_val_normalized, y_val_normalized,
        X_test_normalized, y_test_normalized) = pre_processor.filter_and_scale(X_train, y_train, X_val, y_val,
                                                                             X_test, y_test, scaler_type='minmax')
        print(f"Datos normalizados con dos scalers (inputs y outputs).")


    else: 
        # Cargamos los datos preprocesados
        loader = DataLoader()
        (X_train_normalized, y_train_normalized,
        X_val_normalized, y_val_normalized,
        X_test_normalized, y_test_normalized) = loader.load_data(file_path, data_source)
        print(f"Datos preprocesados cargados.")


    X_train = X_train_normalized
    y_train = y_train_normalized
    X_val = X_val_normalized
    y_val = y_val_normalized
    X_test = X_test_normalized
    y_test = y_test_normalized

    #print("ATENCION")
    #print(X_train.shape[1], y_train.shape[1], X_val.shape[1], y_val.shape[1], X_test.shape[1], y_test.shape[1])

    # Buscar los mejores hiperparámetros para el modelo
    best_params = search_best_hyperparameters(model_name, X_train, y_train, search_type='random', n_iter=10)

    # Configuración de ejemplo para el modelo
    if model_name.lower() == "neural_network":
        model = NeuralNetworkROM(**best_params)

    elif model_name.lower() == "gaussian_process":
        model = GaussianProcessROM(**best_params)
    
    elif model_name.lower() == "rbf":
        model = RBFROM(**best_params)

    elif model_name.lower() == "response_surface":
        model = PolynomialResponseSurface(**best_params)

    elif model_name.lower() == "svr":
        model = SVRROM(**best_params)
    
    # Entrenar el modelo con el subconjunto de entrenamiento y validar sobre el subconjunto de validación
    model.train(X_train, y_train, X_val, y_val)

    # Realizar predicciones sobre el subconjunto de test
    y_pred = model.predict(X_test)

    # Cargamos el scaler de salida guardado en el preprocesamiento
    output_scaler = joblib.load(os.path.join(loader.results_path, 'output_scaler.pkl'))
    if data_source == 'raw' and pre_processor.scaler_type == 'minmax':
        print("Mínimos originales de los outputs:", output_scaler.data_min_)
        print("Máximos originales de los outputs:", output_scaler.data_max_)

    # Evaluar el modelo utilizando las predicciones
    model.evaluate(X_test, y_test, y_pred, output_scaler=output_scaler)
    
        