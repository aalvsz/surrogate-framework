from new_main import idkROM


# OBJETIVO
# que el usuario no tenga la mano sobre el modelo

random_state = 41

# en el config.yml tenemos el input data
# entonces solo con leer el diccionario en el run,
# se ejecutaran secuencialmente las demas funciones
# es decir: load(), load.preprocess(), create_model(), evaluate()
config_yml_path = "config.yml"
"""model_name = 'neural_network'
idkROM.Modelo(model_name, config_yml_path)"""

# para obtener un predict sin más


# para optimizar hiperparametros
NN_dict_hyperparams = {'n_neuronas': 5, 'n_capas': 2, 'activation': 'Tanh',
                        'dropout_rate': 0.2, 'optimizer_nn': 'Adam', 'lr': 0.01,
                          'epochs': 1000}

GP_dict_hyperparams = {'kernel_gp': 'RBF', 'constant_kernel': 2,'matern_nu': 1.5,
                        'expsine_periodicity': 1.0, 'alpha_gp': 0.0001,
                        'optimizer_gp': 'fmin_l_bfgs_b', 'n_restarts_optimizer': 0}


idkROM(random_state).idk_run(config_yml_path, NN_dict_hyperparams)
