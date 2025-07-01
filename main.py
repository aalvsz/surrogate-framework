from idkrom.model import idkROM

# OBJETIVO
# que el usuario no tenga la mano sobre el modelo

random_state = 11

# en el config.yml tenemos el input data
# entonces solo con leer el diccionario en el run,
# se ejecutaran secuencialmente las demas funciones
# es decir: load(), load.preprocess(), create_model(), evaluate()


# para optimizar hiperparametros
NN_dict_hyperparams = {'n_capas': 5, 'n_neuronas': 32, 'activation': 'ReLU',
                        'dropout_rate': 0.1, 'optimizer_nn': 'Adam', 'lr': 0.01,
                         'lr_decrease_rate': 0.5, 'epochs': 5000,
                          'batch_size': 64, 'patience': 100, 'cv_folds': 5,
                            'convergence_threshold': 1e-5}

GP_dict_hyperparams = {'kernel_gp': 'RBF'}

PRS_dict_hyperparams = {'degree': '3'}

RBF_dict_hyperparams = {'alpha': '1.1'}

SVR_dict_hyperparams = {'tolerance': 1e-4, 'epsilon': 0.3}

# Silenciar prints
"""with open(os.devnull, 'w') as fnull:
    with redirect_stdout(fnull):
        rom_instance = idkROM(random_state)
        rom_instance.idk_run(NN_dict_hyperparams)"""

rom_instance = idkROM(random_state)
rom_instance.idk_run(dict_hyperparams=None)