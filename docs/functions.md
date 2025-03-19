# Functions Documentation

## DataLoader Class

### Methods

#### `load_data(file_path: str, data_source: str) -> pd.DataFrame`
Loads data from the specified file path and data source (either raw or processed).

- **Parameters**:
  - `file_path`: The path to the data file.
  - `data_source`: The type of data to load ('raw' or 'processed').
- **Returns**: A pandas DataFrame containing the loaded data.

---

## Pre Class

### Methods

#### `split_dataset(df: pd.DataFrame, last_input_var: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]`
Splits the DataFrame into training, validation, and test sets.

- **Parameters**:
  - `df`: The DataFrame containing the data.
  - `last_input_var`: The index of the last input variable.
- **Returns**: A tuple containing the training inputs, training outputs, validation inputs, validation outputs, test inputs, and test outputs.

#### `preprocessing(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, scaler_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]`
Normalizes the input and output data using the specified scaler type.

- **Parameters**:
  - `X_train`: Training input data.
  - `y_train`: Training output data.
  - `X_val`: Validation input data.
  - `y_val`: Validation output data.
  - `X_test`: Test input data.
  - `y_test`: Test output data.
  - `scaler_type`: The type of scaler to use ('minmax' or others).
- **Returns**: A tuple containing the normalized training inputs, training outputs, validation inputs, validation outputs, test inputs, and test outputs.

---

## search_best_hyperparameters Function

### `search_best_hyperparameters(model_name: str, X_train: np.ndarray, y_train: np.ndarray, search_type: str, n_iter: int) -> Dict[str, Any]`
Searches for the best hyperparameters for the specified model using the given search type.

- **Parameters**:
  - `model_name`: The name of the model ('neural_network', 'gaussian_process', or 'rbf').
  - `X_train`: Training input data.
  - `y_train`: Training output data.
  - `search_type`: The type of search to perform ('random' or others).
  - `n_iter`: The number of iterations for the search.
- **Returns**: A dictionary containing the best hyperparameters found during the search.