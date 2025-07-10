import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer, MissingIndicator, KNNImputer
from ydata_profiling import ProfileReport

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class PreWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper para el preprocesamiento de datos compatible con scikit-learn.

    Permite integración sencilla en pipelines y facilita el manejo de escaladores y datos procesados.
    """

    def __init__(
        self,
        model_name='skorch_model',
        output_folder='results',
        discrete_inputs=[],
        test_size=0.15,
        validation_size=0.0,
        imputer_type='simple',
        scaler_type='minmax',
        filter_method='isolation_forest',
        random_state=42,
        output_scalers=None
    ):
        """
        Inicializa el PreWrapper.

        Args:
            model_name (str): Nombre del modelo.
            output_folder (str): Carpeta de salida para guardar resultados.
            discrete_inputs (list): Lista de columnas de entrada discretas.
            test_size (float): Proporción del conjunto de prueba.
            validation_size (float): Proporción del conjunto de validación.
            imputer_type (str): Tipo de imputador a usar.
            scaler_type (str): Tipo de escalador para variables continuas.
            filter_method (str): Método para filtrar outliers.
            random_state (int): Semilla para reproducibilidad.
            output_scalers (dict): Diccionario de escaladores para las salidas.
        """
        self.model_name = model_name
        self.output_folder = output_folder
        self.discrete_inputs = discrete_inputs
        self.test_size = test_size
        self.validation_size = validation_size
        self.imputer_type = imputer_type
        self.scaler_type = scaler_type
        self.filter_method = filter_method
        self.random_state = random_state
        self.output_scalers = output_scalers

        self.X_train_ = None
        self.y_train_ = None

        # Internals
        self.input_scaler_ = None
        self.output_scalers_ = None
        self.output_path_ = None

    def fit(self, X, y):
        """
        Ajusta el preprocesamiento a los datos de entrada y salida.

        Args:
            X (pd.DataFrame or np.ndarray): Datos de entrada.
            y (pd.DataFrame or np.ndarray): Datos de salida.

        Returns:
            self: El objeto ajustado.
        """
        from idkrom.pre.preprocessing import Pre

        pre = Pre(self.model_name, self.output_folder)

        data, input_scaler, output_scalers, output_path = pre.pre_process_data(
            X,
            y,
            discrete_inputs=self.discrete_inputs,
            test_size=self.test_size,
            validation_size=self.validation_size,
            imputer_type=self.imputer_type,
            scaler_type=self.scaler_type,
            filter_method=self.filter_method,
            random_state=self.random_state,
            output_scalers=self.output_scalers
        )

        self.X_train_, self.y_train_, _, _, _, _, *_ = data
        self.input_scaler_ = input_scaler
        self.output_scalers_ = output_scalers
        self.output_path_ = output_path

        return self

    def transform(self, X):
        """
        Transforma los datos de entrada usando el escalador entrenado.

        Args:
            X (pd.DataFrame or np.ndarray): Datos de entrada.

        Returns:
            pd.DataFrame: Datos transformados.
        """
        X_df = pd.DataFrame(X)
        cont = [c for c in X_df.columns if c not in self.discrete_inputs]
        disc = self.discrete_inputs

        X_cont = pd.DataFrame(self.input_scaler_.transform(X_df[cont]), columns=cont)
        X_disc = X_df[disc].reset_index(drop=True)
        X_final = pd.concat([X_cont, X_disc], axis=1)[X_df.columns]

        return X_final.to_numpy(dtype='float32')

    def fit_transform(self, X, y):
        """
        Ajusta el preprocesamiento y transforma los datos de entrada.

        Args:
            X (pd.DataFrame or np.ndarray): Datos de entrada.
            y (pd.DataFrame or np.ndarray): Datos de salida.

        Returns:
            pd.DataFrame: Datos transformados.
        """
        self.fit(X, y)
        return self.transform(X)

    def get_output_scalers(self):
        """
        Obtiene los escaladores de salida usados.

        Returns:
            dict: Diccionario de escaladores de salida.
        """
        return self.output_scalers_

    def get_output_path(self):
        """
        Obtiene la ruta de salida donde se guardaron los datos procesados.

        Returns:
            str: Ruta de la carpeta de salida.
        """
        return self.output_path_

    def get_input_scaler(self):
        """
        Obtiene el escalador de entrada usado.

        Returns:
            scaler: Escalador de entrada.
        """
        return self.input_scaler_

    def get_processed_train_data(self):
        """
        Obtiene los datos de entrenamiento procesados.

        Returns:
            tuple: (X_train, y_train) procesados.
        """
        return self.X_train_, self.y_train_



class Pre:
    """
    Clase para el preprocesamiento de datos, incluyendo división en conjuntos, filtrado de outliers,
    imputación, escalado y guardado de resultados.
    """

    def __init__(self, model_name: str, output_folder):
        """
        Inicializa la clase de preprocesamiento.

        Args:
            model_name (str): Nombre del modelo.
            output_folder (str): Carpeta de salida para guardar resultados.
        """
        self.model_name = model_name
        self.output_folder = output_folder
        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M")
        
        self.output_folder = os.path.join(self.output_folder, "idkrom_training", f"{self.model_name}_{timestamp}")
        os.makedirs(self.output_folder, exist_ok=True)


    def boxplot(self, data: pd.DataFrame):
        """
        Genera y guarda boxplots para cada columna del DataFrame.

        Args:
            data (pd.DataFrame): DataFrame de entrada para graficar.
        """
        # Create a box plot for each variable
        fig, axes = plt.subplots(nrows=(len(data.columns) + 3) // 4, ncols=4, figsize=(20, 5 * ((len(data.columns) + 3) // 4)))

        for i, column in enumerate(data.columns):
            row, col = divmod(i, 4)
            axes[row, col].boxplot(data[column])
            axes[row, col].set_xlabel(column)
            axes[row, col].set_ylabel('Value')

        # Hide any empty subplots
        for j in range(i + 1, len(axes.flatten())):
            fig.delaxes(axes.flatten()[j])

        plt.savefig(os.path.join(self.output_folder, "preprocessed_data.png"))
        plt.close(fig)


    def filter_outliers_data(self, X: pd.DataFrame, y: pd.DataFrame, subset: str, method: str = 'isolation forest', random_state=None):
        """
        Filtra outliers usando el método especificado.

        Args:
            X (pd.DataFrame): Variables de entrada.
            y (pd.DataFrame): Variables de salida.
            subset (str): Nombre del subset para guardar reportes.
            method (str): Método para detección de outliers ('isolation_forest', 'lof', 'iqr').
            random_state (int, optional): Semilla para reproducibilidad.

        Returns:
            tuple: (X_filtrado, y_filtrado)
        """
        concat_X_y = pd.concat([X, y], axis=1)

        # Profiling report
        report = ProfileReport(concat_X_y, title='Report', minimal=True, progress_bar=False)
        report.to_file(os.path.join(self.output_folder, f'train_profiling_report_{subset}.html'))

        if method == 'isolation forest':
            iso_forest = IsolationForest(contamination=0.05, random_state=random_state)
            outliers = iso_forest.fit_predict(concat_X_y)
            concat_X_y = concat_X_y[outliers == 1]

        elif method == 'lof':
            lof = LocalOutlierFactor(n_neighbors=20)
            outliers = lof.fit_predict(concat_X_y)
            concat_X_y = concat_X_y[outliers == 1]

        elif method == 'iqr':
            Q1 = concat_X_y.quantile(0.25)
            Q3 = concat_X_y.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            concat_X_y = concat_X_y[~((concat_X_y < lower_bound) | (concat_X_y > upper_bound)).any(axis=1)]

        else:
            raise ValueError(f"Unsupported method: {method}")

        X_filtered = concat_X_y[X.columns].reset_index(drop=True)
        y_filtered = concat_X_y[y.columns].reset_index(drop=True)

        return X_filtered, y_filtered


    def get_imputer(self, imputer_type: str):
        """
        Devuelve el imputador seleccionado.

        Args:
            imputer_type (str): Tipo de imputador ('simple', 'missing indicator', 'knn').

        Returns:
            imputer: Objeto imputador.

        Raises:
            ValueError: Si el tipo de imputador no es soportado.
        """
        if imputer_type == 'simple':
            return SimpleImputer(strategy='mean')
        elif imputer_type == 'missing indicator':
            return MissingIndicator()
        elif imputer_type == 'knn':
            return KNNImputer()
        else:
            raise ValueError(f"Imputer '{imputer_type}' is not supported.")        


    def split_dataset(self, inputs_df: pd.DataFrame, outputs_df: pd.DataFrame,
                      test_size: float = 0.15, validation_size: float = 0.15,
                      random_state: int = None):
        """
        Divide los DataFrames de entrada y salida en conjuntos de entrenamiento, validación y prueba.

        Args:
            inputs_df (pd.DataFrame): Variables de entrada.
            outputs_df (pd.DataFrame): Variables de salida.
            test_size (float): Proporción para el conjunto de prueba.
            validation_size (float): Proporción para el conjunto de validación.
            random_state (int, optional): Semilla para reproducibilidad.

        Returns:
            tuple: Si validation_size > 0: (X_train, y_train, X_test, y_test, X_val, y_val)
                   Si validation_size <= 0: (X_train, y_train, X_test, y_test)

        Raises:
            ValueError: Si los tamaños no son válidos.
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
        relative_val_size = validation_size / (1 - test_size)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=relative_val_size,
            random_state=random_state
        )
        split_sets.extend([X_train, y_train, X_test, y_test, X_val, y_val])

        final_train_size = 1 - test_size - validation_size
        print(f"División: Train={final_train_size:.2%}, Validation={validation_size:.2%}, Test={test_size:.2%}")
        return split_sets


    def get_scaler(self, scaler_type: str):
        """
        Devuelve el escalador seleccionado.

        Args:
            scaler_type (str): Tipo de escalador ('minmax', 'standard', 'robust').

        Returns:
            scaler: Objeto escalador.

        Raises:
            ValueError: Si el tipo de escalador no es soportado.
        """
        if scaler_type == 'minmax': return MinMaxScaler()
        if scaler_type == 'standard': return StandardScaler()
        if scaler_type == 'robust': return RobustScaler()
        raise ValueError(f"Scaler '{scaler_type}' not supported.")


    def pre_process_data(
        self,
        inputs_df: pd.DataFrame,
        outputs_df: pd.DataFrame,
        discrete_inputs: list[str],
        test_size: float = 0.15,
        validation_size: float = 0,
        imputer_type: str = 'simple',
        scaler_type: str = 'minmax',
        filter_method: str = 'isolation_forest',
        random_state=None,
        output_scalers: dict = None
    ):
        """
        Realiza el preprocesamiento completo: división, imputación, filtrado de outliers,
        escalado de entradas y salidas, y guardado de resultados.

        Args:
            inputs_df (pd.DataFrame): Variables de entrada.
            outputs_df (pd.DataFrame): Variables de salida.
            discrete_inputs (list): Lista de columnas discretas.
            test_size (float): Proporción para el conjunto de prueba.
            validation_size (float): Proporción para el conjunto de validación.
            imputer_type (str): Tipo de imputador.
            scaler_type (str): Tipo de escalador para entradas.
            filter_method (str): Método para filtrar outliers.
            random_state (int, optional): Semilla para reproducibilidad.
            output_scalers (dict, optional): Diccionario de escaladores para salidas.

        Returns:
            tuple: (data, input_scaler, scalers_aplicados, output_folder)
                - data: lista con los conjuntos procesados.
                - input_scaler: escalador de entrada.
                - scalers_aplicados: diccionario de escaladores de salida.
                - output_folder: ruta de la carpeta de salida.
        """
        # 1) split
        split_sets = self.split_dataset(inputs_df, outputs_df, test_size, validation_size, random_state)
        X_tr, y_tr, X_te, y_te, *rest = split_sets

        # 2) imputation
        imputer = self.get_imputer(imputer_type)
        X_tr_imp = pd.DataFrame(imputer.fit_transform(X_tr), columns=X_tr.columns)
        X_te_imp = pd.DataFrame(imputer.transform(X_te), columns=X_te.columns)

        X_tr_imp = X_tr_imp.reset_index(drop=True)
        y_tr = y_tr.reset_index(drop=True)

        # 3) filter outliers on train
        X_tr_imp, y_tr = self.filter_outliers_data(X_tr_imp, y_tr, 'train', filter_method, random_state)

        # 4) split cont vs disc
        cont = [c for c in X_tr_imp.columns if c not in discrete_inputs]
        disc = discrete_inputs
        X_tr_cont, X_tr_disc = X_tr_imp[cont], X_tr_imp[disc]
        X_te_cont, X_te_disc = X_te_imp[cont], X_te_imp[disc]

        # 5) scale continuous only
        input_scaler = self.get_scaler(scaler_type)
        X_tr_cont_s = pd.DataFrame(input_scaler.fit_transform(X_tr_cont), columns=cont, index=X_tr_cont.index)
        X_te_cont_s = pd.DataFrame(input_scaler.transform(X_te_cont), columns=cont, index=X_te_cont.index)
        joblib.dump(input_scaler, os.path.join(self.output_folder, 'input_scaler.pkl'))

        # reconstruct
        X_tr_final = pd.concat([X_tr_cont_s, X_tr_disc.reset_index(drop=True)], axis=1)[X_tr_imp.columns]
        X_te_final = pd.concat([X_te_cont_s, X_te_disc.reset_index(drop=True)], axis=1)[X_te_imp.columns]

        # 6) scale outputs fully
        if output_scalers is None:
            output_scalers = {col: scaler_type for col in y_tr.columns}  # usa uno para todos

        scalers_aplicados = {}
        scaled_tr_cols = []
        scaled_te_cols = {}

        for col in y_tr.columns:
            scaler = self.get_scaler(output_scalers.get(col, scaler_type))
            tr_scaled = scaler.fit_transform(y_tr[[col]])
            te_scaled = scaler.transform(y_te[[col]])
            
            scaled_tr_cols.append(pd.DataFrame(tr_scaled, columns=[col], index=y_tr.index))
            scaled_te_cols[col] = pd.DataFrame(te_scaled, columns=[col], index=y_te.index)
            
            scalers_aplicados[col] = scaler

        # Unir columnas de golpe (esto evita el PerformanceWarning)
        y_tr_scaled = pd.concat(scaled_tr_cols, axis=1)
        y_te_scaled = pd.concat([scaled_te_cols[col] for col in y_tr.columns], axis=1)

        joblib.dump(scalers_aplicados, os.path.join(self.output_folder, 'output_scaler.pkl'))

        # 7) save Parquet files en lugar de CSV
        parquet_path = os.path.join(self.output_folder, 'data')
        os.makedirs(parquet_path, exist_ok=True)

        X_tr_final.to_parquet(os.path.join(parquet_path, 'X_train.parquet'), index=False)
        y_tr_scaled.to_parquet(os.path.join(parquet_path, 'y_train.parquet'), index=False)
        X_te_final.to_parquet(os.path.join(parquet_path, 'X_test.parquet'), index=False)
        y_te_scaled.to_parquet(os.path.join(parquet_path, 'y_test.parquet'), index=False)

        data = [X_tr_final, y_tr_scaled, X_te_final, y_te_scaled]
        # validation
        if rest:
            X_val, y_val = rest
            X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
            X_val_cont = X_val_imp[cont]; X_val_disc = X_val_imp[disc]
            X_val_cont_s = pd.DataFrame(input_scaler.transform(X_val_cont), columns=cont)
            X_val_final = pd.concat([X_val_cont_s, X_val_disc.reset_index(drop=True)], axis=1)[X_val_imp.columns]
            scaled_val_cols = []

            for col in y_val.columns:
                scaler = scalers_aplicados[col]
                scaled_col = pd.Series(scaler.transform(y_val[[col]]).flatten(), name=col)
                scaled_val_cols.append(scaled_col)

            y_val_scaled = pd.concat(scaled_val_cols, axis=1)

            X_val_final.to_parquet(os.path.join(parquet_path, 'X_val.parquet'), index=False)
            y_val_scaled.to_parquet(os.path.join(parquet_path, 'y_val.parquet'), index=False)

            data.extend([X_val_final, y_val_scaled])
        else:
            data.extend([None, None])

        with open(os.path.join(self.output_folder, 'output_scaler_meta.json'), 'w') as f:
            json.dump({col: type(scaler).__name__ for col, scaler in scalers_aplicados.items()}, f, indent=2)

        print("Preprocessed with selective scaling and saved to Parquet.")
        return data, input_scaler, scalers_aplicados, self.output_folder