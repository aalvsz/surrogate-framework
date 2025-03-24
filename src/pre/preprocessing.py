import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer
from ydata_profiling import ProfileReport

class Pre:
    """
    A class for preprocessing data, including splitting into training, validation, and test sets,
    creating box plots, and normalizing data.
    """

    def __init__(self):
        """
        Initializes the Preprocessing class.
        Sets up input and output dataframes and creates the output folder if it doesn't exist.
        """
        self.inputs = None
        self.outputs = None
        self.scaler_type = None
        
        self.output_folder = os.path.join(os.getcwd(), 'results')
        os.makedirs(os.path.dirname(self.output_folder), exist_ok=True)


    def boxplot(self, data: pd.DataFrame):
        """
        Generates and saves box plots for each column in the input DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame for which to generate box plots.
        """
        # Create a box plot for each variable
        fig, axes = plt.subplots(nrows=(len(data.columns) + 3) // 4, ncols=4, figsize=(20, 5 * ((len(data.columns) + 3) // 4)))

        for i, column in enumerate(data.columns):
            row, col = divmod(i, 4)
            axes[row, col].boxplot(data[column])
            # axes[row, col].set_title(f'Box plot of {column}')
            axes[row, col].set_xlabel(column)
            axes[row, col].set_ylabel('Value')

        # Hide any empty subplots
        for j in range(i + 1, len(axes.flatten())):
            fig.delaxes(axes.flatten()[j])

        plt.savefig(os.path.join(self.output_folder, "preprocessed_data.png"))
        plt.close(fig)


    def filter_outliers_data(self, X: pd.DataFrame, y: pd.DataFrame, subset: str, method: str = 'isolation_forest', random_state=None):
        """
        Filters outliers using the specified criteria for a given dataset.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.DataFrame): Target values.
            subset (str): Subset name for report saving.
            method (str): Method to use for outlier detection ('isolation_forest', 'lof', 'iqr').

        Returns:
            tuple: Filtered input and target data.
        """
        concat_X_y = pd.concat([X, y], axis=1)

        # Profiling report
        report = ProfileReport(concat_X_y, title='Report', minimal=True, progress_bar=False)
        report.to_file(os.path.join(self.output_folder, f'train_profiling_report_{subset}.html'))

        if method == 'isolation_forest':
            # Using IsolationForest to filter outliers
            iso_forest = IsolationForest(contamination=0.05, random_state=random_state)
            outliers = iso_forest.fit_predict(concat_X_y)
            concat_X_y = concat_X_y[outliers == 1]

        elif method == 'lof':
            # Local Outlier Factor (LOF)
            lof = LocalOutlierFactor(n_neighbors=20, random_state=random_state)
            outliers = lof.fit_predict(concat_X_y)
            concat_X_y = concat_X_y[outliers == 1]

        elif method == 'iqr':
            # Interquartile Range (IQR) for outlier detection
            Q1 = concat_X_y.quantile(0.25)
            Q3 = concat_X_y.quantile(0.75)
            IQR = Q3 - Q1

            # Filtering out rows outside of the IQR range
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            concat_X_y = concat_X_y[~((concat_X_y < lower_bound) | (concat_X_y > upper_bound)).any(axis=1)]

        else:
            raise ValueError(f"Unsupported method: {method}")

        # Separate filtered X and y
        X_filtered = concat_X_y[X.columns]
        y_filtered = concat_X_y[y.columns]

        return X_filtered, y_filtered


    def get_scaler(self, scaler_type: str):
        """
        Returns the chosen scaler.
        
        Args:
            scaler_type (str): The type of scaler to use ('minmax', 'standard', 'robust').

        Returns:
            scaler: The scaler object to use for normalization.
        
        Raises:
            ValueError: If the specified scaler type is not supported.
        """
        if scaler_type == 'minmax':
            return MinMaxScaler()
        elif scaler_type == 'standard':
            return StandardScaler()
        elif scaler_type == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Scaler '{scaler_type}' is not supported.")

    def split_dataset(self, inputs_df: pd.DataFrame, outputs_df: pd.DataFrame,
                        train_size: float = 0.7, val_size: float = 0.15,
                        test_size: float = 0.15, random_state: int = None):
        """
        Divide los DataFrames de inputs y outputs en conjuntos de entrenamiento, validación y prueba.
        
        Args:
            inputs_df (pd.DataFrame): DataFrame que contiene las variables de entrada.
            outputs_df (pd.DataFrame): DataFrame que contiene las variables de salida.
            train_size (float): Proporción para el conjunto de entrenamiento (por defecto 0.7).
            val_size (float): Proporción para el conjunto de validación (por defecto 0.15).
            test_size (float): Proporción para el conjunto de prueba (por defecto 0.15).
            random_state (int, optional): Semilla para reproducibilidad.
        
        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        # Verificar que las proporciones sumen 1
        assert train_size + val_size + test_size == 1.0, "Las proporciones deben sumar 1."

        # Primera división: entrenamiento vs. (validación + prueba)
        X_train, X_temp, y_train, y_temp = train_test_split(
            inputs_df, outputs_df, train_size=train_size, random_state=random_state)
        
        # División de la parte temporal en validación y prueba
        # Se calcula la proporción de test sobre el total de (validación + prueba)
        test_ratio = test_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_ratio, random_state=random_state)
        
        return X_train, y_train, X_val, y_val, X_test, y_test


    def pre_process_data(self, inputs_df: pd.DataFrame, outputs_df: pd.DataFrame,
                          train_size: float = 0.7, val_size: float = 0.15,
                            test_size: float = 0.15, scaler_type: str = 'minmax',
                             filter_method: str = 'isolation_forest', random_state=None):
        """
        Applies preprocessing steps to the input data, including outlier removal, normalization,
        and generating a profiling report. The scaler is fitted only on the training set and applied
        to validation and test sets.

        Args:
            X_train (pd.DataFrame): Training input data.
            y_train (pd.DataFrame): Training output data.
            X_val (pd.DataFrame): Validation input data.
            y_val (pd.DataFrame): Validation output data.
            X_test (pd.DataFrame): Test input data.
            y_test (pd.DataFrame): Test output data.

        Returns:
            tuple: Normalized training, validation, and test sets.
        """

        # We split the dataset into train, validation and test subsets
        X_train, y_train, X_val, y_val, X_test, y_test = self.split_dataset(inputs_df, outputs_df,
                                                                             train_size, val_size,
                                                                               test_size, random_state)

        # Apply outlier filtering consistently across all sets
        X_train_filtered, y_train_filtered = self.filter_outliers_data(X_train, y_train, 'train', filter_method, random_state)
        X_val_filtered, y_val_filtered = self.filter_outliers_data(X_val, y_val, 'validation', filter_method, random_state)
        X_test_filtered, y_test_filtered = self.filter_outliers_data(X_test, y_test, 'test', filter_method, random_state)

        # Impute missing values if any
        imputer = SimpleImputer(strategy='mean')  # You can change strategy as needed
        X_train_filtered = pd.DataFrame(imputer.fit_transform(X_train_filtered), columns=X_train_filtered.columns)
        X_val_filtered = pd.DataFrame(imputer.transform(X_val_filtered), columns=X_val_filtered.columns)
        X_test_filtered = pd.DataFrame(imputer.transform(X_test_filtered), columns=X_test_filtered.columns)

        # Set the scaler type
        self.scaler_type = scaler_type

        # Get the appropriate scaler
        input_scaler = self.get_scaler(scaler_type)
        output_scaler = self.get_scaler(scaler_type)

        # Scale inputs with an independent scaler
        X_train_normalized = pd.DataFrame(input_scaler.fit_transform(X_train_filtered), columns=X_train_filtered.columns)
        X_val_normalized = pd.DataFrame(input_scaler.transform(X_val_filtered), columns=X_val_filtered.columns)
        X_test_normalized = pd.DataFrame(input_scaler.transform(X_test_filtered), columns=X_test_filtered.columns)
        joblib.dump(input_scaler, os.path.join(self.output_folder, 'input_scaler.pkl'))

        # Scale outputs with another independent scaler
        y_train_normalized = pd.DataFrame(output_scaler.fit_transform(y_train_filtered), columns=y_train.columns)
        y_val_normalized = pd.DataFrame(output_scaler.transform(y_val_filtered), columns=y_val.columns)
        y_test_normalized = pd.DataFrame(output_scaler.transform(y_test_filtered), columns=y_test.columns)
        joblib.dump(output_scaler, os.path.join(self.output_folder, 'output_scaler.pkl'))

        # Save preprocessed data in multiple CSV files according to the subset they belong to (input/output and train/validation/test)
        X_train_normalized.to_csv(os.path.join(self.output_folder, 'X_train.csv'), index=False)
        y_train_normalized.to_csv(os.path.join(self.output_folder, 'y_train.csv'), index=False)
        X_val_normalized.to_csv(os.path.join(self.output_folder, 'X_val.csv'), index=False)
        y_val_normalized.to_csv(os.path.join(self.output_folder, 'y_val.csv'), index=False)
        X_test_normalized.to_csv(os.path.join(self.output_folder, 'X_test.csv'), index=False)
        y_test_normalized.to_csv(os.path.join(self.output_folder, 'y_test.csv'), index=False)

        print(f'Preprocessed data saved.')

        return (X_train_normalized, y_train_normalized,
                X_val_normalized, y_val_normalized,
                X_test_normalized, y_test_normalized)
