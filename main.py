import os
import pickle
from idkrom.model import idkROM
from contextlib import redirect_stdout

# OBJETIVO
# que el usuario no tenga la mano sobre el modelo

random_state = 5

# en el config.yml tenemos el input data
# entonces solo con leer el diccionario en el run,
# se ejecutaran secuencialmente las demas funciones
# es decir: load(), load.preprocess(), create_model(), evaluate()


# para optimizar hiperparametros
NN_dict_hyperparams = {'n_capas': 2, 'n_neuronas': 32, 'activation': 'ReLU',
                        'dropout_rate': 0.1, 'optimizer_nn': 'Adam', 'lr': 0.01,
                         'lr_decrease_rate': 0.5, 'epochs': 5000,
                          'batch_size': 64, 'patience': 100, 'cv_folds': 5,
                            'convergence_threshold': 1e-5}

GP_dict_hyperparams = {'kernel_gp': 'RBF'}

PRS_dict_hyperparams = {'degree': '3'}

RBF_dict_hyperparams = {'alpha': '1.1'}

SVR_dict_hyperparams = {'tolerance': 1e-4, 'epsilon': 0.3}

# Silenciar prints
with open(os.devnull, 'w') as fnull:
    with redirect_stdout(fnull):
        rom_instance = idkROM(random_state)
        rom_instance.idk_run(NN_dict_hyperparams)

# Guardar el modelo
# --- Guardar el modelo y el scaler juntos ---
if hasattr(rom_instance, 'model') and rom_instance.model is not None:

    save_rom_path = os.path.join(os.getcwd(), 'idksim', 'idkROM_model.pkl')
    print(f"Guardando el modelo ROM en: {save_rom_path}")
    try:
        with open(save_rom_path, 'wb') as f:
            pickle.dump(rom_instance, f)
        print("Modelo guardado exitosamente.")
    except Exception as e:
        print(f"Error al guardar el modelo: {e}")
else:
    print("Error: No se encontró el modelo entrenado en la instancia de idkROM después de la ejecución.")