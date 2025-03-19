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
python main.py <ruta_csv> <tipo_datos: raw/processed> <modelo: neural_network/gaussian_process/rbf>
```

- `<ruta_csv>`: Path to the CSV file containing the data.
- `<tipo_datos>`: Specify whether the data is raw or preprocessed.
- `<modelo>`: Choose the model to use (neural_network, gaussian_process, or rbf).

## Directory Structure
```
idkROM
├── docs                # Documentation files
├── src                 # Source code
│   ├── gui            # GUI implementation
│   ├── loader         # Data loading functionality
│   ├── models         # Machine learning models
│   ├── pre            # Data preprocessing
│   └── tools          # Utility functions
├── main.py            # Main entry point of the application
├── mkdocs.yml         # MkDocs configuration file
└── README.md          # Project overview and instructions
```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.