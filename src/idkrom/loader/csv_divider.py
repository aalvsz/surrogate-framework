import pandas as pd

def split_dataset(file_path, num_input_cols):
    """
    Divide un archivo CSV en dos archivos: uno con las columnas de entrada y otro con las de salida.

    Args:
        file_path (str): Ruta al archivo CSV original.
        num_input_cols (int): Número de columnas de entrada (las primeras columnas).

    Returns:
        None. Guarda dos archivos CSV: uno para inputs y otro para outputs.
    """
    # Cargar el dataset
    df = pd.read_csv(file_path, delimiter=";", decimal=",")
    
    # Separar inputs y outputs
    inputs = df.iloc[:, :num_input_cols]
    outputs = df.iloc[:, num_input_cols:]
    
    # Crear nombres de archivo de salida
    input_file = file_path.replace('.csv', '_inputs.csv')
    output_file = file_path.replace('.csv', '_outputs.csv')
    
    # Guardar los archivos
    inputs.to_csv(input_file, index=False)
    outputs.to_csv(output_file, index=False)
    
    print(f"Inputs guardados en: {input_file}")
    print(f"Outputs guardados en: {output_file}")

# Ejemplo de uso
if __name__ == "__main__":
    """
    Ejemplo de uso del script para dividir un archivo CSV en archivos de entradas y salidas.
    Modifica 'file_path' y 'num_input_cols' según tu caso.
    """
    file_path = r"D:\idkROM\idkROM\data\main_platoHydr_idkFEM_DOE_Results copy.csv"
    num_input_cols = 5
    split_dataset(file_path, num_input_cols)