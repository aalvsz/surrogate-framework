import sys
import io
import os
import streamlit as st
from src.loader.import_data import DataLoader
from src.pre.preprocessing import Pre
from src.models.neural_network import NeuralNetworkROM
from src.models.gaussian_process import GaussianProcessROM
from src.models.rbf import RBFROM
from src.models.polynomial_response_surface import PolynomialResponseSurface
from src.models.svr import SVRROM

# Función para capturar los prints de evaluate
def capture_output(func, *args, **kwargs):
    # Redirigir el print a un StringIO
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    # Ejecutar la función
    func(*args, **kwargs)
    
    # Restaurar la salida estándar
    sys.stdout = sys.__stdout__
    
    # Devolver el contenido capturado
    return captured_output.getvalue()

# Función para cargar y procesar los datos
def load_and_process_data():
    loader = DataLoader()
    config_dict = loader.read_yml(os.getcwd() + "/config.yml")
    inputs_file_path = config_dict['data inputs']
    outputs_file_path = config_dict['data outputs']
    data_source = config_dict['read mode']
    
    if data_source == "raw":
        inputs_df, outputs_df = loader.load_data(input_path=inputs_file_path, output_path=outputs_file_path, data_source="raw")
        preprocessor = Pre()
        scaler_type = 'minmax'
        filter_method = 'isolation_forest'
        X_train_normalized, y_train_normalized, X_val_normalized, y_val_normalized, X_test_normalized, y_test_normalized = preprocessor.pre_process_data(
            inputs_df, outputs_df, 0.7, 0.15, 0.15, scaler_type, filter_method, random_state=41)
        return X_train_normalized, y_train_normalized, X_val_normalized, y_val_normalized, X_test_normalized, y_test_normalized, config_dict
    else:
        (X_train_normalized, y_train_normalized, X_val_normalized, y_val_normalized, X_test_normalized, y_test_normalized) = loader.load_data(
            output_path=outputs_file_path, data_source="pre")
        return X_train_normalized, y_train_normalized, X_val_normalized, y_val_normalized, X_test_normalized, y_test_normalized, config_dict

# Función para configurar el modelo
def configure_model(model_name, rom_config):
    if model_name.lower() == "neural_network":
        model = NeuralNetworkROM(rom_config, random_state=41)
    elif model_name.lower() == "gaussian_process":
        model = GaussianProcessROM(rom_config, random_state=41)
    elif model_name.lower() == "rbf":
        model = RBFROM(rom_config, random_state=41)
    elif model_name.lower() == "response_surface":
        model = PolynomialResponseSurface(rom_config, random_state=41)
    elif model_name.lower() == "svr":
        model = SVRROM(rom_config, random_state=41)
    return model

# Función para entrenar y predecir el modelo
def train_and_predict(model, X_train_normalized, y_train_normalized, X_val_normalized, y_val_normalized, X_test_normalized):
    model.train(X_train_normalized, y_train_normalized, X_val_normalized, y_val_normalized)
    y_pred = model.predict(X_test_normalized)
    return y_pred

# Configuración de Streamlit
st.title('ROM Model Training and Evaluation')

# Pestañas para navegar
tab1, tab2, tab3 = st.tabs(["Ejecutar Modelo", "Hiperparámetros y Modelo", "Métricas y Gráfico"])

with tab1:
    st.header("Ejecutar Modelo")
    
    # Cargar y procesar datos
    X_train_normalized, y_train_normalized, X_val_normalized, y_val_normalized, X_test_normalized, y_test_normalized, config_dict = load_and_process_data()
    
    model_name = config_dict['model type']
    hyperparams = config_dict['hyperparams']
    mode = config_dict['mode']
    eval_metrics = config_dict['eval metrics']
    default_params = DataLoader().default_params
    
    rom_config = {
        'input_dim': X_train_normalized.shape[1],
        'output_dim': y_train_normalized.shape[1],
        'hyperparams': hyperparams if mode in ['best', 'manual'] else default_params[model_name],
        'mode': mode,
        'model_name': model_name
    }

    # Configurar el modelo
    model = configure_model(model_name, rom_config)
    
    # Mostrar los mensajes que se imprimen en la consola durante el entrenamiento
    st.write("Configurando el modelo...")
    st.write(f"Modelo elegido: {model_name}")
    st.write(f"Configuración: {rom_config}")
    
    # Ejecutar el entrenamiento y hacer predicciones
    if st.button('Entrenar y Predecir'):
        st.write("Entrenando el modelo...")
        y_pred = train_and_predict(model, X_train_normalized, y_train_normalized, X_val_normalized, y_val_normalized, X_test_normalized)
        st.session_state.y_pred = y_pred  # Guardar las predicciones en el estado de la sesión
        st.write("Predicciones realizadas.")

with tab2:
    st.header("Hiperparámetros y Modelo")
    
    # Mostrar hiperparámetros y el modelo seleccionado
    st.write(f"Modelo elegido: {model_name}")
    st.write("Hiperparámetros:")
    st.json(config_dict['hyperparams'])

with tab3:
    st.header("Métricas y Gráfico")
    
    # Verificar si se ha generado y_pred
    if 'y_pred' in st.session_state:
        y_pred = st.session_state.y_pred
        st.write("Evaluando el modelo...")
        
        # Capturar los prints de la función evaluate y mostrarlo en Streamlit
        eval_output = capture_output(model.evaluate, X_test_normalized, y_test_normalized, y_pred, eval_metrics)
        
        # Mostrar los prints capturados
        st.text(eval_output)
        
        # Ruta del gráfico generado por evaluate()
        image_path = os.path.join(os.getcwd(), 'images', f'error_metrics_{model_name}.png')
        
        # Verificar si el archivo existe
        if os.path.exists(image_path):
            st.image(image_path, caption=f"Gráfico de Error Metrics para {model_name}", use_column_width=True)
        else:
            st.write("No se encontró el gráfico de error metrics.")
    else:
        st.write("Primero entrena el modelo para generar predicciones.")
