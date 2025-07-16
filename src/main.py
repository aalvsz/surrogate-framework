from idkrom.model import idkROM

"""
main.py

Script principal para ejecutar el pipeline de entrenamiento, predicción y evaluación
de modelos ROM (Reduced Order Model) usando el framework idkROM.

Este archivo está diseñado para que el usuario no tenga que manipular directamente
las clases internas del modelo. Toda la configuración de datos y parámetros se realiza
a través del archivo config.yml, permitiendo una ejecución sencilla y reproducible.

Flujo principal:
    1. Lee la configuración desde config.yml.
    2. Ejecuta secuencialmente: load(), preprocess(), create_model(), evaluate().
    3. Permite la optimización de hiperparámetros pasando un diccionario opcional.

Variables de ejemplo para hiperparámetros de distintos modelos están incluidas,
pero por defecto se ejecuta el modelo y configuración definidos en config.yml.

Uso:
    Simplemente ejecuta este script para correr el pipeline completo.
    Puedes modificar el diccionario de hiperparámetros y pasarlo a idk_run()
    para realizar optimización manual.

Ejemplo de uso para optimización:
    rom_instance.idk_run(NN_dict_hyperparams)
"""

random_state = 11

# Diccionarios de ejemplo para optimización de hiperparámetros
NN_dict_hyperparams = {'n_capas': 5, 'n_neuronas': 32, 'activation': 'ReLU',
                        'dropout_rate': 0.1, 'optimizer_nn': 'Adam', 'lr': 0.01,
                         'lr_decrease_rate': 0.5, 'epochs': 5000,
                          'batch_size': 64, 'patience': 100, 'cv_folds': 5,
                            'convergence_threshold': 1e-5}

GP_dict_hyperparams = {'kernel_gp': 'RBF'}

PRS_dict_hyperparams = {'degree': '3'}

RBF_dict_hyperparams = {'alpha': '1.1'}

SVR_dict_hyperparams = {'tolerance': 1e-4, 'epsilon': 0.3}

# Ejemplo para silenciar prints durante la ejecución (descomentable)
"""with open(os.devnull, 'w') as fnull:
    with redirect_stdout(fnull):
        rom_instance = idkROM(random_state)
        rom_instance.idk_run(NN_dict_hyperparams)"""

# Instancia y ejecución principal del pipeline
rom_instance = idkROM(random_state, config_yml_path="D:\idk_framework\idkROM\src\config.yml")
rom_instance.run_idkROM(dict_hyperparams=None)