import pytest
import os, sys
import numpy as np
import pandas as pd

# Añadimos src al sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from idkrom.model import idkROM

@pytest.fixture
def dummy_rom_config(tmp_path):
    # Le damos hiperparámetros mínimos consistentes con el modelo de red neuronal
    hyperparams = {
        "n_layers": 1,
        "n_neurons": 8,
        "learning_rate": 0.01,
        "lr_step": 10,
        "lr_decrease_rate": 0.5,
        "activation": "ReLU",
        "dropout_rate": 0.1,
        "optimizer": "Adam",
        "epochs": 3,
        "batch_size": 4,
        "cv_folds": 2,
        "patience": 5,
        "convergence_threshold": 1e-4
    }
    return {
        "validation_mode": "single",
        "input_dim": 4,
        "output_dim": 2,
        "discrete_inputs": [],
        "hyperparams": hyperparams,
        "mode": "manual",
        "model_name": "neural_network",
        "output_folder": str(tmp_path),
        "eval_metrics": "mse"
    }

@pytest.fixture
def dummy_data():
    # X_train, y_train, X_test, y_test, X_val, y_val
    X_train = pd.DataFrame(np.random.rand(20, 4), columns=[f"x{i}" for i in range(4)])
    y_train = pd.DataFrame(np.random.rand(20, 2), columns=["out1", "out2"])
    X_test = pd.DataFrame(np.random.rand(5, 4), columns=[f"x{i}" for i in range(4)])
    y_test = pd.DataFrame(np.random.rand(5, 2), columns=["out1", "out2"])
    X_val = pd.DataFrame(np.random.rand(5, 4), columns=[f"x{i}" for i in range(4)])
    y_val = pd.DataFrame(np.random.rand(5, 2), columns=["out1", "out2"])
    return [X_train, y_train, X_test, y_test, X_val, y_val]


def test_train_and_predict(dummy_rom_config, dummy_data):
    rom = idkROM(random_state=42)
    y_pred, model = rom.train_and_predict(dummy_rom_config, dummy_data)

    assert y_pred.shape[0] == dummy_data[2].shape[0], "número de filas de predicciones no coincide con X_test"
    assert y_pred.shape[1] == dummy_rom_config["output_dim"], "dimensión de salida incorrecta"
    assert hasattr(model, "predict"), "el modelo no tiene método predict"


def test_evaluate(dummy_rom_config, dummy_data, tmp_path):
    rom = idkROM(random_state=42)

    y_test = dummy_data[3]
    y_pred = np.random.rand(y_test.shape[0], y_test.shape[1])

    # Simulamos escalers
    from sklearn.preprocessing import MinMaxScaler
    scalers = {}
    for col in y_test.columns:
        scaler = MinMaxScaler()
        scaler.fit(y_test[[col]])
        scalers[col] = scaler
    rom.output_scaler = scalers
    rom.output_folder = str(tmp_path)

    results = rom.evaluate(y_test, y_pred, dummy_rom_config)

    # resultado consistente
    assert isinstance(results, dict)
    assert "metric" in results
    assert os.path.exists(os.path.join(rom.output_folder, "metrics_results.json"))
    assert os.path.exists(os.path.join(rom.output_folder, "predicciones_test.csv"))
    assert os.path.exists(os.path.join(rom.output_folder, "valores_esperados_test.csv"))
