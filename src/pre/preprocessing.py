import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
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


    def filter_outliers_data(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Filters outliers using the same criteria for a given dataset.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.DataFrame): Target values.

        Returns:
            tuple: Filtered input and target data.
        """
        data = pd.concat([X, y], axis=1)
        #report = ProfileReport(data, title='Report', minimal=True)
        #report.to_file(os.path.join(self.output_folder, 'train_profiling_report.html'))

        data_filtered = data[(data['Q_total_int'] <= 0.00025) &
                            (data['h_mean_int'] <= 0.0002) &
                            (data['h_mean_ext'] <= 0.00015)]
        X_filtered = data_filtered[X.columns]
        y_filtered = data_filtered[y.columns]

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
        

    def filter_and_scale(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame,
                       y_val: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, scaler_type: str = 'minmax'):
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
        # Apply outlier filtering consistently across all sets
        X_train_filtered, y_train_filtered = self.filter_outliers_data(X_train, y_train)
        X_val_filtered, y_val_filtered = self.filter_outliers_data(X_val, y_val)
        X_test_filtered, y_test_filtered = self.filter_outliers_data(X_test, y_test)

        # Set the scaler type
        self.scaler_type = scaler_type

        # Get the appropriate scaler
        input_scaler = self.get_scaler(scaler_type)
        output_scaler = self.get_scaler(scaler_type)

        # Scale inputs with an independent scaler
        X_train_normalized = pd.DataFrame(input_scaler.fit_transform(X_train_filtered), columns=X_train.columns)
        X_val_normalized = pd.DataFrame(input_scaler.transform(X_val_filtered), columns=X_val.columns)
        X_test_normalized = pd.DataFrame(input_scaler.transform(X_test_filtered), columns=X_test.columns)
        joblib.dump(input_scaler, os.path.join(self.output_folder, 'input_scaler.pkl'))

        # Scale outputs with another independent scaler
        y_train_normalized = pd.DataFrame(output_scaler.fit_transform(y_train_filtered), columns=y_train.columns)
        y_val_normalized = pd.DataFrame(output_scaler.transform(y_val_filtered), columns=y_val.columns)
        y_test_normalized = pd.DataFrame(output_scaler.transform(y_test_filtered), columns=y_test.columns)
        joblib.dump(output_scaler, os.path.join(self.output_folder, 'output_scaler.pkl'))
        if scaler_type == 'minmax':
            print("Original minimums of the outputs:", output_scaler.data_min_)
            print("Original maximums of the outputs:", output_scaler.data_max_)

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


    @staticmethod
    def split_dataset(df: pd.DataFrame, last_input_var: int, train_size: float = 0.7,
                       val_size: float = 0.15, test_size: float = 0.15, random_state: int = None):
        """
        Splits a DataFrame into training, validation, and test sets.

        Args:
            df (pd.DataFrame): Complete DataFrame with features and target.
            last_input_var (int): Index of the last input variable column.
            train_size (float): Proportion for training (default 0.7).
            val_size (float): Proportion for validation (default 0.15).
            test_size (float): Proportion for test (default 0.15).
            random_state (int, optional): Seed for reproducibility. Defaults to None.

        Returns:
            tuple: A tuple containing the training, validation, and test sets (X_train, y_train, X_val, y_val, X_test, y_test).
        """
        assert train_size + val_size + test_size == 1.0, "The proportions must sum to 1."

        # Separate inputs (first columns) and outputs (remaining columns)
        X = df.iloc[:, :last_input_var]
        y = df.iloc[:, last_input_var:]

        # First split into training set and temporary set (validation + test)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, random_state=random_state)

        # Then split the temporary set into validation and test sets
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=random_state)

        return X_train, y_train, X_val, y_val, X_test, y_test