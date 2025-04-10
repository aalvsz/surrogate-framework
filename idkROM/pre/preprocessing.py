import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer, MissingIndicator, KNNImputer
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


    def filter_outliers_data(self, X: pd.DataFrame, y: pd.DataFrame, subset: str, method: str = 'isolation forest', random_state=None):
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

        if method == 'isolation forest':
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


    def get_imputer(self, imputer_type: str):
        """
        Returns the chosen imputer.
        
        Args:
            imputer_type (str): The type of imputer to use ('simple', 'missing indicator', 'knn', 'iterative').

        Returns:
            imputer: The imputer object to use for missing values.
        
        Raises:
            ValueError: If the specified imputer type is not supported.
        """
        if imputer_type == 'simple':
            return SimpleImputer(strategy='mean')
        elif imputer_type == 'missing indicator':
            return MissingIndicator()
        elif imputer_type == 'knn':
            return KNNImputer()
        else:
            raise ValueError(f"Imputer '{imputer_type}' is not supported.")
        

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
                      test_size: float = 0.15, validation_size: float = 0.15,
                      random_state: int = None):
        """
        Divide los DataFrames de inputs y outputs en conjuntos de entrenamiento, validación y prueba.

        Args:
            inputs_df (pd.DataFrame): DataFrame que contiene las variables de entrada.
            outputs_df (pd.DataFrame): DataFrame que contiene las variables de salida.
            test_size (float): Proporción para el conjunto de prueba (por defecto 0.15).
            validation_size (float): Proporción del *conjunto original* para el conjunto de validación (por defecto 0.15).
                                     Debe ser tal que test_size + validation_size < 1.
            random_state (int, optional): Semilla para reproducibilidad en ambas divisiones.

        Returns:
            tuple:
                Si validation_size > 0: (X_train, y_train, X_val, y_val, X_test, y_test)
                Si validation_size <= 0 o None: (X_train, y_train, X_test, y_test)

        Raises:
            ValueError: Si test_size o validation_size no son válidos, o si test_size + validation_size >= 1.
        """
        split_sets = []

        if validation_size is None:
            validation_size = 0.0 # Tratar None como 0 para simplificar

        if not (0 <= test_size < 1):
             raise ValueError("test_size debe estar entre 0 y 1.")
        if not (0 <= validation_size < 1):
             raise ValueError("validation_size debe estar entre 0 y 1.")
        if test_size + validation_size >= 1:
            raise ValueError("La suma de test_size y validation_size debe ser menor que 1.")

        # 1. Dividir en (entrenamiento + validación) y prueba
        # El tamaño del conjunto temporal de entrenamiento será 1 - test_size
        X_train, X_test, y_train, y_test = train_test_split(
            inputs_df, outputs_df,
            test_size=test_size,
            random_state=random_state
        )


        # Si no se necesita conjunto de validación, devolver directamente
        if validation_size == 0:
            print(f"División: Train={1-test_size:.2%}, Test={test_size:.2%}")
            split_sets.extend([X_train, y_train, X_test, y_test])
            return split_sets

        # 2. Dividir (entrenamiento + validación) en entrenamiento y validación
        # Calcular la proporción de validación relativa al tamaño de X_train_val
        relative_val_size = validation_size / (1 - test_size)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=relative_val_size,
            random_state=random_state
        )
        split_sets.extend([X_train, y_train, X_test, y_test, X_val, y_val])

        final_train_size = 1 - test_size - validation_size
        print(f"División: Train={final_train_size:.2%}, Validation={validation_size:.2%}, Test={test_size:.2%}")
        # Devolver en el orden convencional: train, validation, test
        return split_sets


    def pre_process_data(self, inputs_df: pd.DataFrame, outputs_df: pd.DataFrame,
                            test_size: float = 0.15, validation_size: float = 0, imputer_type: str = 'simple', 
                            scaler_type: str = 'minmax', filter_method: str = 'isolation_forest',
                            random_state=None):
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
        split_sets = self.split_dataset(inputs_df, outputs_df, test_size, validation_size, random_state)
        
        # Apply outlier filtering consistently across all sets
        X_train, y_train = self.filter_outliers_data(split_sets[0], split_sets[1], 'train', filter_method, random_state)

        # NO FILTRAMOS EL SET DE TEST NI EL DE VALIDACION
        """X_test_filtered, y_test_filtered = self.filter_outliers_data(split_sets[2], split_sets[3], 'test', filter_method, random_state)"""

        # Impute missing values if any
        imputer = self.get_imputer(imputer_type)
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(imputer.transform(split_sets[2]), columns=split_sets[2].columns)

        # Set the scaler type
        self.scaler_type = scaler_type

        # Get the appropriate scaler
        input_scaler = self.get_scaler(scaler_type)
        output_scaler = self.get_scaler(scaler_type)
        # Scale inputs with an independent scaler
        X_train = pd.DataFrame(input_scaler.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(input_scaler.transform(X_test), columns=X_test.columns)
        joblib.dump(input_scaler, os.path.join(self.output_folder, 'input_scaler.pkl'))

        # Scale outputs with another independent scaler
        y_train = pd.DataFrame(output_scaler.fit_transform(y_train), columns=split_sets[1].columns)
        y_test = pd.DataFrame(output_scaler.transform(split_sets[3]), columns=split_sets[3].columns)
        joblib.dump(output_scaler, os.path.join(self.output_folder, 'output_scaler.pkl'))

        # Save preprocessed data in multiple CSV files according to the subset they belong to (input/output and train/validation/test)
        X_train.to_csv(os.path.join(self.output_folder, 'X_train.csv'), index=False)
        y_train.to_csv(os.path.join(self.output_folder, 'y_train.csv'), index=False)
        X_test.to_csv(os.path.join(self.output_folder, 'X_test.csv'), index=False)
        y_test.to_csv(os.path.join(self.output_folder, 'y_test.csv'), index=False)
        data_after_split = [X_train, y_train, X_test, y_test]

        if len(split_sets) > 4:
            # X_val_filtered, y_val_filtered = self.filter_outliers_data(split_sets[4], split_sets[5], 'validation', filter_method, random_state)
            X_val = pd.DataFrame(imputer.transform(split_sets[4]), columns=split_sets[4].columns)
            X_val = pd.DataFrame(input_scaler.transform(X_val), columns=X_val.columns)
            y_val = pd.DataFrame(output_scaler.transform(split_sets[5]), columns=split_sets[5].columns)
            data_after_split.extend([X_val, y_val])    
            X_val.to_csv(os.path.join(self.output_folder, 'X_train.csv'), index=False)
            y_val.to_csv(os.path.join(self.output_folder, 'y_train.csv'), index=False)
        else:
            data_after_split.extend([None, None])    



        print(f'Preprocessed data saved.')

        return data_after_split, input_scaler, output_scaler
