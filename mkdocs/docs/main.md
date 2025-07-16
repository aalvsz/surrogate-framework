# Ejecución

Existen dos maneras de ejecutar `idkROM` a día de hoy.

**Opcional**: podemos darle unos hiperparámetros personalizados como argumento a la función `idk_run`.

## Opción 1

1. Instalamos `idkSIM` como paquete desde GitLab.
Podemos usar los comandos

```
pip install "idksimulation @ git+https://kodea.danobatgroup.com/dip/precision/ideko/simulation/idksimulation.git@main"
pip install "idkrom @ git+https://kodea.danobatgroup.com/dip/precision/ideko/simulation/idkROM.git@main"
pip install "idkdoe @ git+https://kodea.danobatgroup.com/dip/precision/ideko/simulation/idkdoe.git@main"
pip install "idkopt @ git+https://kodea.danobatgroup.com/dip/precision/ideko/simulation/idkopt.git@main"

```
accediendo con nuestras credenciales de Kodea.


2. Creamos un script simple como el siguiente:

```python
from idkrom.model import idkROM

random_state = 11
rom_instance = idkROM(random_state, config_yml_path="path_a_tu_yaml")
rom_instance.idk_run(dict_hyperparams="aquí_tu_diccionario")

```

La forma del diccionario de hiperparámetros puede ser, por ejemplo para una red neuronal:
```python
NN_dict_hyperparams = {
    'n_capas': 5,
    'n_neuronas': 32,
    'activation': 'ReLU',
    'dropout_rate': 0.1,
    'optimizer_nn': 'Adam',
    'lr': 0.01,
    'lr_decrease_rate': 0.5,
    'epochs': 5000,
    'batch_size': 64,
    'patience': 100,
    'cv_folds': 5,
    'convergence_threshold': 1e-5
}
```

### Ejemplo de ejecución
Este sería un yaml de configuración de ejemplo para idkROM:
```yaml


data:
  input: C:\Users\aalvarezsanz\OneDrive - DanobatGroup\Documentos\idk_framework\DOE_datos\Carga carro\inputs.csv
  output: C:\Users\aalvarezsanz\OneDrive - DanobatGroup\Documentos\idk_framework\DOE_datos\Carga carro\outputs.csv

preprocess:
  read mode: raw
  preprocessed data path: None
  validation mode: single # single to use X_val and y_val, cross for cross-validation
  imputer: simple # simple, missing indicator or knn
  scaler: minmax # minmax, standard or robust
  filter method: isolation forest # isolation forest, lof (local outlier factor) or iqr (interquartile range)
  test size: 0.10
  validation size: 0.10


evaluate:
  output folder: D:\idk_framework\idksimulation\results
  metrics: mse
  
model:
  type: neural_network
  discrete inputs: [N_trans]
  hyperparams:
    mode: manual
    params:
      n_layers: 2
      n_neurons: 128
      activation: ReLU
      output activation: 
      dropout_rate: 0.1
      optimizer: Adam
      learning_rate: 0.001
      lr_step: 200
      lr_decrease_rate: 0.5
      epochs: 1000
      batch_size: 64
      patience: 100
      cv_folds: 5
      convergence_threshold: 1e-09



idk_params:

  input_data: [data, input]
  output_data: [data, output]
  data_source: [preprocess, read mode]
  validation_mode: [preprocess, validation mode]
  imputer: [preprocess, imputer]
  scaler: [preprocess, scaler]
  filter_method: [preprocess, filter method]
  test_size: [preprocess, test size]
  validation_size: [preprocess, validation size]

#neural network
  n_capas: [model, hyperparams, params, n_layers]
  n_neuronas: [model, hyperparams, params, n_neurons]
  activation: [model, hyperparams, params, activation]
  dropout_rate: [model, hyperparams, params, dropout_rate]
  optimizer_nn: [model, hyperparams, params, optimizer]
  lr: [model, hyperparams, params, learning_rate]
  lr_step: [model, hyperparams, params, lr_step]
  lr_decrease_rate: [model, hyperparams, params, lr_decrease_rate]
  epochs: [model, hyperparams, params, epochs]
  batch_size: [model, hyperparams, params, batch_size]
  patience: [model, hyperparams, params, patience]
  cv_folds: [model, hyperparams, params, cv_folds]
  convergence_threshold: [model, hyperparams, params, convergence_threshold]

#gaussian process
  kernel_gp: [model, hyperparams, params, kernel]
  constant_kernel: [model, hyperparams, params, constant_kernel]
  matern_nu: [model, hyperparams, params, matern_nu]
  expsine_periodicity: [model, hyperparams, params, expsine_periodicity]
  alpha_gp: [model, hyperparams, params, alpha]
  optimizer_gp: [model, hyperparams, params, optimizer]
  n_restarts_optimizer: [model, hyperparams, params, n_restarts_optimizer]

#rbf
  alpha_rbf: [model, hyperparams, params, alpha]
  kernel_rbf: [model, hyperparams, params, kernel]
  gamma_rbf: [model, hyperparams, params, gamma]
  degree_rbf: [model, hyperparams, params, degree]

#response surface
  degree_poly: [model, hyperparams, params, kernel]
  interaction_only: [model, hyperparams, params, constant_kernel]
  include_bias: [model, hyperparams, params, matern_nu]
  order: [model, hyperparams, params, expsine_periodicity]
  fit_intercept: [model, hyperparams, params, alpha]
  positive: [model, hyperparams, params, positive]

#svr
  kernel_svr: [model, hyperparams, params, kernel]
  degree_svr: [model, hyperparams, params, degree]
  gamma_svr: [model, hyperparams, params, gamma]
  tolerance: [model, hyperparams, params, tolerance]
  C: [model, hyperparams, params, C]
  epsilon: [model, hyperparams, params, epsilon]



```

---

#### Sección data
Aquí definiremos las rutas al CSV de inputs (encabezados y valores), y al CSV de outputs. Los delimitadores deben ser: ',' para separación y '.' para decimales.

---

#### Sección preprocess
Aquí se ajustarán los siguientes parámetros del preprocesamiento. para más detalles, consultar la documentación oficial de Sklearn (https://scikit-learn.org/stable/index.html).

- **`read mode`**:  
  - `raw`: los datos no han sido preprocesados previamente.  
  - `pre`: los datos ya han sido preprocesados, por lo que las opciones anteriores (como `data`) no tienen efecto.

- **`preprocessed data path`**:  
  Ruta al archivo preprocesado o `None`.

- **`validation mode`**:  
  - `single`: divide el conjunto en `train`, `validation` y `test`.  
  - `cross`: utiliza validación cruzada (*cross-validation*).

- **`imputer`**:  
  Método de imputación de valores faltantes:
  - `simple`  
  - `missing indicator`  
  - `knn`

- **`scaler`**:  
  Método de normalización o escalado de variables:
  - `minmax`  
  - `standard`  
  - `robust`

- **`filter method`**:  
  Método para detección y filtrado de *outliers*:
  - `isolation forest`  
  - `lof` (local outlier factor)  
  - `iqr` (interquartile range)

- **`test size`**:  
  Tamaño del conjunto de test, expresado como fracción (valor entre 0 y 1).

- **`validation size`**:  
  Tamaño del conjunto de validación (entre 0 y 1), solo aplicable si `validation mode` es `single`.

---

#### Sección evaluate
- output folder: ruta donde se guardarán todos los resultados del entrenamiento, scalers, métricas, etc.
- metrics: `mse`, `mae` o `mape`

---

#### Sección model
- **`type`**: tipo de modelo ROM a utilizar (`neural network`, `gaussian process`, `svr`, etc.).
- **`discrete inputs`**: lista de variables categóricas o discretas del conjunto de entrada (opcional).
- **`hyperparams`**: diccionario de hiperparámetros específicos del modelo.
- **`mode`**:
  - `manual`: usa los hiperparámetros definidos explícitamente.
  - `cv`: activa búsqueda por validación cruzada usando `skorch` (en desarrollo).

---

##### Hiperparámetros específicos para `neural network`

- **`n_layers`**: número de capas ocultas intermedias (entero ≥ 1).
- **`n_neurons`**: número de neuronas por capa oculta (entero ≥ 1). Se asume un mismo valor para todas las capas.
- **`activation`**: función de activación de las capas ocultas (`ReLU`, `Tanh`, `Sigmoid`).
- **`output activation`**: función de activación de la capa de salida (`ReLU`, `Tanh`, `Sigmoid`, `None`).
- **`dropout_rate`**: tasa de *dropout* (fracción de neuronas desactivadas aleatoriamente por capa), valor entre 0 y 1.
- **`optimizer`**: optimizador para el entrenamiento (`Adam`, `SGD`, etc.).
- **`learning rate`**: tasa de aprendizaje inicial (valor entre 0 y 1).
- **`lr_step`**: número de épocas tras las cuales se ajustará el *learning rate* mediante `StepLR`.
- **`lr_decrease_rate`**: factor multiplicativo aplicado al *learning rate* cada `lr_step` épocas (entre 0 y 1).
- **`epochs`**: número máximo de épocas de entrenamiento.
- **`batch_size`**: tamaño de los lotes de entrenamiento (típicamente 32, 64 o 128).
- **`patience`**: número de épocas sin mejora antes de aplicar *early stopping*. Se evalúa en función del umbral `convergence_threshold`.
- **`cv_folds`**: número de particiones (*folds*) para validación cruzada (solo si se selecciona `validation: cross`).
- **`convergence_threshold`**: umbral mínimo de mejora en la `training loss` entre épocas sucesivas para continuar el entrenamiento.

#### Sección idk_params
##### Sistema de asignación de parámetros (`idk_params`)

La sección `idk_params` del archivo de configuración actúa como un **mapa de redirección de claves**. Permite que cualquier clave que se pase al método `idk_run()` como diccionario de hiperparámetros (por ejemplo, `NN_dict_hyperparams`) sea vinculada directamente con su correspondiente entrada dentro del archivo `config.yml`.

Esto permite que los usuarios **sobrescriban valores del YAML sin modificar el archivo directamente**, simplemente pasando un diccionario como argumento.

---

##### Funcionamiento

- Cada clave de `idk_params` representa un **alias o sinónimo** que el usuario puede usar como entrada en el diccionario `dict_hyperparams`.
- Su valor asociado es una **ruta de acceso** que indica **dónde en el YAML de configuración se encuentra el parámetro real que debe modificarse**.
- Esa ruta es una lista que se interpreta como niveles jerárquicos en el YAML (`sección -> subsección -> parámetro`).

---

##### Ejemplo de uso

Supongamos que en tu archivo `config.yml` tienes:

```yaml
model:
  hyperparams:
    params:
      n_layers: 4
```
Y en `idk_params` defines:
```yaml
n_capas: [model, hyperparams, params, n_layers]
```
Entonces si el usuario llama a :
```python
rom_instance.idk_run({"n_capas": 6})
```
La clave `n_capas` será redirigida internamente a `model → hyperparams → params → n_layers`, y ese valor (`4`) será sobrescrito por `6`.


## Opción 2

Se trata de desplazarse al `main.py` de la carpeta `src`, y simplemente ejecutarlo, cambiando el path del YAML de configuración que utilizaremos.
