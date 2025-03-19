# Classes Documentation

## DataLoader
The `DataLoader` class is responsible for loading data from various sources, including raw and preprocessed data.

### Methods
- `load_data(file_path: str, data_source: str) -> DataFrame`
  - Loads data from the specified file path and data source (raw or processed).

## Pre
The `Pre` class handles data preprocessing tasks, including splitting datasets and normalizing inputs and outputs.

### Methods
- `split_dataset(df: DataFrame, last_input_var: int) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]`
  - Splits the dataset into training, validation, and test sets based on the last input variable index.
  
- `preprocessing(X_train: DataFrame, y_train: DataFrame, X_val: DataFrame, y_val: DataFrame, X_test: DataFrame, y_test: DataFrame, scaler_type: str) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]`
  - Normalizes the input and output datasets using the specified scaler type (e.g., 'minmax').

## NeuralNetworkROM
The `NeuralNetworkROM` class implements a neural network model for regression or classification tasks.

### Methods
- `train(X_train: DataFrame, y_train: DataFrame, X_val: DataFrame, y_val: DataFrame) -> None`
  - Trains the neural network model using the training data and validates it with the validation data.
  
- `predict(X_test: DataFrame) -> DataFrame`
  - Makes predictions on the test dataset.
  
- `evaluate(X_test: DataFrame, y_test: DataFrame, y_pred: DataFrame, output_scaler: Any) -> None`
  - Evaluates the model's performance using the test dataset and predictions.

## GaussianProcessROM
The `GaussianProcessROM` class implements a Gaussian process model for regression tasks.

### Methods
- `train(X_train: DataFrame, y_train: DataFrame, X_val: DataFrame, y_val: DataFrame) -> None`
  - Trains the Gaussian process model using the training data and validates it with the validation data.
  
- `predict(X_test: DataFrame) -> DataFrame`
  - Makes predictions on the test dataset.
  
- `evaluate(X_test: DataFrame, y_test: DataFrame, y_pred: DataFrame, output_scaler: Any) -> None`
  - Evaluates the model's performance using the test dataset and predictions.

## RBFROM
The `RBFROM` class implements a radial basis function model for regression tasks.

### Methods
- `train(X_train: DataFrame, y_train: DataFrame, X_val: DataFrame, y_val: DataFrame) -> None`
  - Trains the radial basis function model using the training data and validates it with the validation data.
  
- `predict(X_test: DataFrame) -> DataFrame`
  - Makes predictions on the test dataset.
  
- `evaluate(X_test: DataFrame, y_test: DataFrame, y_pred: DataFrame, output_scaler: Any) -> None`
  - Evaluates the model's performance using the test dataset and predictions.