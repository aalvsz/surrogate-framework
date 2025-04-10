import os
import pickle
import streamlit as st
from idkROM.idkROM import idkROM
import io
import sys
from contextlib import redirect_stdout

# --- CONFIGURACIONES INICIALES ---
st.set_page_config(page_title="idkROM App", layout="wide")
random_state = 5

# Diccionarios de hiperparámetros
model_configs = {
    "Neural Network": {'n_capas': 2, 'n_neuronas': 32, 'activation': 'ReLU',
                       'dropout_rate': 0.1, 'optimizer_nn': 'Adam', 'lr': 0.01,
                       'lr_step': 1000, 'lr_decrease_rate': 0.5, 'epochs': 5000,
                       'batch_size': 64, 'patience': 100, 'cv_folds': 5,
                       'convergence_threshold': 1e-5},
    "Gaussian Process": {'kernel_gp': 'RBF'},
    "Polynomial Response Surface": {'degree': '3'},
    "Radial Basis Function": {'alpha': '1.1'},
    "Support Vector Regression": {'tolerance': 1e-4, 'epsilon': 0.3}
}

# --- STREAMLIT TABS ---
tab1, tab2, tab3 = st.tabs(["Entrenamiento", "Resultados", "Consola"])

# --- TAB 1: Entrenamiento ---
with tab1:
    st.title("Entrenamiento de Modelos ROM")

    model_choice = st.selectbox("Selecciona el modelo ROM:", list(model_configs.keys()))

    if st.button("Entrenar modelo"):
        # Capturar stdout (prints)
        f = io.StringIO()
        with redirect_stdout(f):
            rom_instance = idkROM(random_state)
            rom_instance.idk_run(model_configs[model_choice])

            if hasattr(rom_instance, 'model') and rom_instance.model is not None:
                save_rom_path = os.path.join(os.getcwd(), 'idksim', 'rom.pkl')
                print(f"Guardando el modelo ROM en: {save_rom_path}")
                try:
                    with open(save_rom_path, 'wb') as f_model:
                        pickle.dump(rom_instance.model, f_model)
                    print("Modelo guardado exitosamente.")
                    st.session_state.model_path = save_rom_path
                except Exception as e:
                    print(f"Error al guardar el modelo: {e}")
            else:
                print("Error: No se encontró el modelo entrenado.")

        # Guardar consola y modelo actual
        st.session_state.output_log = f.getvalue()
        st.session_state.last_model_name = model_choice
        st.success("Entrenamiento finalizado. Consulta los resultados y consola.")

# --- TAB 2: Resultados ---
with tab2:
    st.title("Resultados del Modelo")

    if "last_model_name" in st.session_state:
        model_key = st.session_state.last_model_name
        image_filename = f"error_metrics_{model_key.replace(' ', '_')}.png"
        image_path = os.path.join("images", image_filename)

        if os.path.exists(image_path):
            st.image(image_path, caption=f"Métricas de error para {model_key}", use_column_width=True)
        else:
            st.warning(f"No se encontró la imagen: {image_filename}")
    else:
        st.info("Entrena un modelo primero para ver resultados.")

    if "model_path" in st.session_state and os.path.exists(st.session_state.model_path):
        with open(st.session_state.model_path, "rb") as f:
            st.download_button("📥 Descargar modelo ROM", f, file_name="rom.pkl")

# --- TAB 3: Consola ---
with tab3:
    st.title("Salida de Consola")

    if "output_log" in st.session_state:
        st.text_area("Log de ejecución:", st.session_state.output_log, height=400)
    else:
        st.info("Ejecuta un entrenamiento para ver la consola.")


# streamlit run main_streamlit.py