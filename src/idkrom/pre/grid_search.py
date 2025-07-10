import yaml
import torch
import os
import pandas as pd
from datetime import datetime
from skorch import NeuralNetRegressor
from skorch.callbacks import LRScheduler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from idkrom.architectures.neural_network import FeedforwardNet
from idkrom.pre.preprocessing import PreWrapper

def load_pre_from_yaml(yaml_path):
    """
    Carga la configuración de preprocesamiento y modelo desde un archivo YAML,
    y construye un PreWrapper compatible con scikit-learn para pipelines.

    Args:
        yaml_path (str): Ruta al archivo YAML de configuración.

    Returns:
        tuple: (pre_wrapper, output_folder, inputs_file, outputs_file)
            - pre_wrapper: Instancia de PreWrapper para usar en un pipeline.
            - output_folder: Carpeta de salida para resultados.
            - inputs_file: Ruta al archivo de entradas.
            - outputs_file: Ruta al archivo de salidas.
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    skorch_cfg = config['sklearn modules']
    preprocess_cfg = config['preprocess']

    model_name = skorch_cfg.get('model name', 'unnamed')
    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M")
    output_folder = os.path.join(skorch_cfg.get('output folder', 'results'), 'skorch', f"{model_name}_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)

    pre_wrapper = PreWrapper(
        model_name=model_name,
        output_folder=output_folder,
        discrete_inputs=preprocess_cfg.get('discrete inputs', []),
        test_size=preprocess_cfg.get('test size', 0.15),
        validation_size=preprocess_cfg.get('validation size', 0.0),
        imputer_type=preprocess_cfg.get('imputer', 'simple'),
        scaler_type=preprocess_cfg.get('scaler', 'minmax'),
        filter_method=preprocess_cfg.get('filter method', 'isolation forest'),
        random_state=preprocess_cfg.get('random_state', 1),
        #output_scalers=preprocess_cfg.get('scaler', 'minmax')
    )

    return pre_wrapper, output_folder, config['data']['input'], config['data']['output']

# =========================
# Ejemplo de uso: Grid Search con PyTorch y scikit-learn
# =========================
# Este script permite usar GridSearchCV de scikit-learn para optimizar hiperparámetros
# de modelos de PyTorch (usando skorch) y pipelines de preprocesamiento.
# Es ideal para experimentos reproducibles y búsqueda de hiperparámetros en redes neuronales.

# Cargar YAML y crear preprocesador
yaml_path = r'D:\idk_framework\idkROM\src\config.yml'  # ruta a tu archivo de configuración
pre_wrapper, output_folder, inputs_file, outputs_file = load_pre_from_yaml(yaml_path)

# Crear red neuronal compatible con skorch
net = NeuralNetRegressor(
    module=FeedforwardNet,
    module__input_dim=16,
    module__output_dim=3,
    module__n_layers=2,
    module__n_neurons=64,
    module__activation='relu',
    module__dropout_rate=0.2,
    module__output_activation=None,
    max_epochs=100,
    lr=0.01,
    optimizer=torch.optim.Adam,
    batch_size=32,
    verbose=0,
    iterator_train__shuffle=True,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Callbacks para ajustar el learning rate durante el entrenamiento
step_sizes = [10, 50, 100]
gammas = [0.1, 0.5, 0.9]
patiences = [10, 50, 100]
factors = [0.1, 0.5]

step_lr_callbacks = [
    [LRScheduler(policy=torch.optim.lr_scheduler.StepLR, step_size=ss, gamma=gm)]
    for ss in step_sizes for gm in gammas
]
reduce_lr_callbacks = [
    [LRScheduler(policy=torch.optim.lr_scheduler.ReduceLROnPlateau, patience=pt, factor=fc)]
    for pt in patiences for fc in factors
]

all_callbacks = step_lr_callbacks + reduce_lr_callbacks

# Pipeline de preprocesamiento + modelo, compatible con scikit-learn
pipe = Pipeline([
    ('pre', pre_wrapper),
    ('model', net)
])

# Definición de la grilla de hiperparámetros para GridSearchCV
param_grid = {
    'model__lr': [1e-3],
    # Puedes agregar más hiperparámetros aquí, por ejemplo:
    # 'model__module__n_layers': [2, 3],
    # 'model__module__n_neurons': [32, 64, 128],
    # 'model__callbacks': all_callbacks,
}

gs = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=2,
    verbose=0,
    error_score='raise'
)

# Cargar datos de entrada y salida
X = pd.read_csv(inputs_file)
y = pd.read_csv(outputs_file)

# Entrenar y buscar los mejores hiperparámetros
gs.fit(X, y)
print("Mejores parámetros:", gs.best_params_)
print("Mejor score:", gs.best_score_)
