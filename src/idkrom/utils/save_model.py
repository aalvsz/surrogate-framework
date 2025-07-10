import pickle
import os

def save_rom_instance(rom_instance, rom_config):
    """
    Guarda una instancia de modelo ROM (Reduced Order Model) en un archivo .pkl.

    Args:
        rom_instance: Instancia del modelo ROM entrenado.
        rom_config (dict): Diccionario de configuración del modelo ROM, debe contener al menos:
            - 'model_name': Nombre del modelo.
            - 'output_folder': Carpeta donde se guardará el archivo.

    Returns:
        None. El modelo se guarda en disco como archivo pickle.
    """
    # Guardar el modelo
    # --- Guardar el modelo y el scaler juntos ---
    print(f"rom_instance: {rom_instance}")

    rom_name = rom_config['model_name']
    save_rom_path = os.path.join(rom_config['output_folder'], f'{rom_name}_object.pkl')
    print(f"Guardando el modelo ROM en: {save_rom_path}")
    try:
        with open(save_rom_path, 'wb') as f:
            pickle.dump(rom_instance, f)
        return print(f"Modelo guardado exitosamente en {save_rom_path}.")
    except Exception as e:
        print(f"Error al guardar el modelo: {e}")