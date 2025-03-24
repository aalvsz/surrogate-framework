import os
import joblib
import streamlit as st
from src.loader.import_data import DataLoader
from src.pre.preprocessing import Pre
from src.models.neural_network import NeuralNetworkROM
from src.models.gaussian_process import GaussianProcessROM
from src.models.rbf import RBFROM
from src.models.polynomial_response_surface import PolynomialResponseSurface
from src.models.svr import SVRROM

# Función para cargar los datos y la configuración
def load_data(config_path):
    loader = DataLoader()
    config_dict = loader.read_yml(config_path)
    inputs_file_path = config_dict['data inputs']
    outputs_file_path = config_dict['data outputs']
    data_source = config_dict['read mode']
    model_name = config_dict['model type']
    hyperparams = config_dict['hyperparams']
    mode = config_dict['mode']
    default_params = loader.default_params
    
    return inputs_file_path, outputs_file_path, data_source, model_name, hyperparams, mode, default_params, loader

# Función de preprocesamiento de datos
def preprocess_data(inputs_df, outputs_df):
    pre_processor = Pre()
    X_train, y_train, X_val, y_val, X_test, y_test = pre_processor.split_dataset(inputs_df, outputs_df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42)
    
    # Normalizar los datos
    X_train_normalized, y_train_normalized, X_val_normalized, y_val_normalized, X_test_normalized, y_test_normalized = pre_processor.filter_and_scale(X_train, y_train, X_val, y_val, X_test, y_test, scaler_type='minmax')
    
    return X_train_normalized, y_train_normalized, X_val_normalized, y_val_normalized, X_test_normalized, y_test_normalized

# Función para seleccionar el modelo
def select_model(model_name, rom_config):
    if model_name.lower() == "neural_network":
        return NeuralNetworkROM(rom_config)
    elif model_name.lower() == "gaussian_process":
        return GaussianProcessROM(rom_config)
    elif model_name.lower() == "rbf":
        return RBFROM(rom_config)
    elif model_name.lower() == "response_surface":
        return PolynomialResponseSurface(rom_config)
    elif model_name.lower() == "svr":
        return SVRROM(rom_config)
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")

# Lógica principal para la ejecución de flujo completo
def run_pipeline(config_path):
    inputs_file_path, outputs_file_path, data_source, model_name, hyperparams, mode, default_params, loader = load_data(config_path)

    if data_source == "raw":
        # Cargar los datos y extraer los DataFrames con inputs y outputs
        inputs_df, outputs_df = loader.load_data(inputs_file_path, outputs_file_path, data_source="raw")
        st.write("✅ Datos cargados.")
        
        # Preprocesar los datos
        X_train_normalized, y_train_normalized, X_val_normalized, y_val_normalized, X_test_normalized, y_test_normalized = preprocess_data(inputs_df, outputs_df)
        st.write("✅ Datos preprocesados.")
    else:
        # Cargar los datos preprocesados
        X_train_normalized, y_train_normalized, X_val_normalized, y_val_normalized, X_test_normalized, y_test_normalized = loader.load_data(inputs_file_path, outputs_file_path, data_source)
        st.write("✅ Datos preprocesados cargados.")
    
    # Configuración para el ROM
    rom_config = {
        'input_dim': inputs_df.shape[1],
        'output_dim': outputs_df.shape[1],
        'hyperparams': hyperparams if mode in ['best', 'manual'] else default_params[model_name],
        'mode': mode,
        'model_name': model_name
    }

    st.write("Configuración del modelo:", rom_config)

    # Selección del modelo
    model = select_model(model_name, rom_config)
    
    # Entrenar el modelo
    st.write("Entrenando el modelo...")
    model.train(X_train_normalized, y_train_normalized, X_val_normalized, y_val_normalized)
    st.write("✅ Entrenamiento completo.")

    # Realizar predicciones
    y_pred = model.predict(X_test_normalized)
    st.write("✅ Predicción realizada.")

    # Cargar el scaler de salida
    output_scaler = joblib.load(os.path.join(loader.results_path, 'output_scaler.pkl'))

    # Evaluar el modelo
    st.write("Evaluando el modelo...")
    model.evaluate(X_test_normalized, y_test_normalized, y_pred, output_scaler=output_scaler)
    st.write("✅ Evaluación completa.")

# Interfaz de Streamlit
def run_streamlit_app():
    st.title("ROM (Reduced Order Model) App")
    
    config_path = st.text_input("Ruta al archivo de configuración", "./config.yml")
    
    if os.path.exists(config_path):
        if st.button("Ejecutar pipeline"):
            run_pipeline(config_path)
            st.write("Pipeline ejecutado exitosamente.")
    else:
        st.error("El archivo de configuración no existe. Por favor, verifica la ruta.")

# Ejecuta la app de Streamlit directamente
run_streamlit_app()

# streamlit run main.py (cambiarle el nombre a main.py)    


