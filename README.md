# idkROM Project

## Overview
idkROM is a Python-based application designed for data modeling and analysis. It provides various machine learning models, including neural networks, Gaussian processes, and radial basis functions, to facilitate predictive modeling. The application supports both raw and preprocessed data, allowing users to load, preprocess, and analyze data efficiently.

## Features
- **Data Loading**: Load data from various sources using the `DataLoader` class.
- **Data Preprocessing**: Normalize and split datasets using the `Pre` class.
- **Modeling**: Implement various machine learning models:
  - Neural Network (NeuralNetworkROM)
  - Gaussian Process (GaussianProcessROM)
  - Radial Basis Function (RBFROM)
- **Hyperparameter Optimization**: Search for the best hyperparameters using the `search_best_hyperparameters` function.
- **Evaluation**: Evaluate model performance on test datasets.

## Installation
To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage
To run the application, use the following command:

```bash
python main.py
```

- you must have previously filled every field of the config.yml file


## Directory Structure
```
idkROM
├── docs                # Documentation files
├── src                 # Source code
│   ├── gui            # GUI implementation
│   ├── loader         # Data loading functionality
│   ├── models         # Machine learning models
│   ├── pre            # Data preprocessing
│   └── visualization  # Utility functions
├── main.py            # Main entry point of the application
├── main_streamlit.py  # Streamlit executable file (GUI)
├── config.yml         # Argument configuration file
├── mkdocs.yml         # MkDocs configuration file
└── README.md          # Project overview and instructions
```

## Interpretación de las métricas de errores
- **1. Predicciones vs. Valores Reales:**

 - **Qué muestra:** Este gráfico compara directamente las predicciones de tu modelo con los valores reales correspondientes del conjunto de prueba. El eje x representa los valores reales, y el eje y representa las predicciones.
Cómo interpretarlo:
 - **Ajuste ideal:** Si el modelo fuera perfecto, todos los puntos caerían exactamente sobre la línea diagonal (y=x) que se incluye en el gráfico. Cuanto más cerca estén los puntos de esta línea, mejor será el rendimiento del modelo.
Sobreestimación: Los puntos que se encuentran por encima de la línea diagonal indican casos en los que el modelo sobreestimó el valor real.
Subestimación: Los puntos que se encuentran por debajo de la línea diagonal indican casos en los que el modelo subestimó el valor real.
 - **Dispersión:** La dispersión de los puntos alrededor de la línea diagonal te da una idea de la variabilidad de los errores del modelo. Una mayor dispersión sugiere errores más grandes y menos consistencia en las predicciones.
Tendencias: Observa si hay alguna tendencia sistemática en los errores (por ejemplo, si el modelo tiende a sobreestimar para valores bajos y subestimar para valores altos, o viceversa).

- **2. Distribución de Errores:**

 - **Qué muestra:** Este histograma visualiza la frecuencia con la que ocurren diferentes magnitudes de error en las predicciones de tu modelo. El eje x representa la magnitud del error (valor real - predicción), y el eje y representa la frecuencia de cada rango de error.
Cómo interpretarlo:
Distribución ideal: Lo ideal es que la distribución de errores esté centrada en cero y tenga una forma aproximadamente normal (una campana). Esto indicaría que los errores son aleatorios y no hay un sesgo sistemático en las predicciones (es decir, el modelo no tiende consistentemente a sobreestimar o subestimar).
 - **Sesgo:** Si el histograma está desplazado hacia la derecha (cola larga hacia valores positivos), sugiere que el modelo tiende a subestimar. Si está desplazado hacia la izquierda (cola larga hacia valores negativos), sugiere que el modelo tiende a sobreestimar.
Magnitud de los errores: La dispersión del histograma te da una idea de la magnitud típica de los errores. Una distribución más ancha indica errores más grandes en general.
 - **Valores atípicos:** Barras aisladas lejos del cuerpo principal de la distribución podrían indicar valores atípicos o casos donde el modelo tuvo un rendimiento particularmente malo.


- **3. Errores vs. Predicciones:**

 - **Qué muestra:** Este gráfico de dispersión examina si existe alguna relación entre las predicciones del modelo y los errores que comete. El eje x representa las predicciones del modelo, y el eje y representa los errores correspondientes.
Cómo interpretarlo:
 - **Patrón ideal:** Lo ideal es que los puntos estén dispersos aleatoriamente alrededor de la línea horizontal en y=0, sin ningún patrón discernible. Esto indicaría que los errores del modelo son independientes del valor predicho y que el modelo tiene un rendimiento consistente en todo el rango de predicciones.
Heterocedasticidad: Si la dispersión de los errores aumenta o disminuye a medida que cambian las predicciones (formando una especie de embudo), esto sugiere heterocedasticidad, lo que significa que la varianza de los errores no es constante.
 - **Patrones curvos:** Un patrón curvo en los errores podría indicar que el modelo no está capturando alguna relación no lineal en los datos.
Sesgo dependiente de la predicción: Si los errores tienden a ser consistentemente positivos o negativos en ciertos rangos de predicciones, podría indicar un sesgo en el modelo para esos valores.


- **4. Curvas de Pérdida de Entrenamiento y Validación (solo para NeuralNetworkROM):**

 - **Qué muestra:** Este gráfico lineal muestra cómo evoluciona la función de pérdida (por ejemplo, el error cuadrático medio) durante el proceso de entrenamiento de una red neuronal. El eje x representa las épocas de entrenamiento, y el eje y representa el valor de la pérdida tanto para el conjunto de entrenamiento como para el conjunto de validación.
Cómo interpretarlo:
 - **Aprendizaje:** Idealmente, ambas curvas de pérdida deberían disminuir con el tiempo, lo que indica que el modelo está aprendiendo de los datos.
Sobreajuste (Overfitting): Si la pérdida de entrenamiento continúa disminuyendo mientras que la pérdida de validación comienza a aumentar o se estanca, esto sugiere que el modelo está empezando a sobreajustarse a los datos de entrenamiento y no generaliza bien a datos nuevos (el conjunto de validación).
Subajuste (Underfitting): Si ambas curvas de pérdida permanecen relativamente altas y no disminuyen significativamente, esto indica que el modelo probablemente no es lo suficientemente complejo o no se ha entrenado lo suficiente para capturar los patrones en los datos.
 - **Ajuste óptimo:** Un buen ajuste se suele observar cuando ambas pérdidas disminuyen a un valor bajo y se estabilizan, con la pérdida de validación ligeramente superior a la pérdida de entrenamiento.
Brecha entre curvas: Una gran diferencia entre la pérdida de entrenamiento y la pérdida de validación también puede ser una señal de sobreajuste o de que los conjuntos de entrenamiento y validación son muy diferentes.



## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.