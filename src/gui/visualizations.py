import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_comparison_html(y_train, predictions, model_name="neural_network"):
    # Crear los DataFrames de las salidas reales y predichas
    y_train = pd.DataFrame(y_train, columns=[f'Real_{i}' for i in range(y_train.shape[1])])
    predictions = pd.DataFrame(predictions, columns=[f'Predicted_{i}' for i in range(predictions.shape[1])])
    
    # Crear una figura de subgráficos
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Real vs Predicted Values", "Prediction Error")
    )
    
    # Comparación entre valores reales y predichos
    for col in y_train.columns:
        fig.add_trace(
            go.Scatter(x=y_train.index, y=y_train[col], mode='lines', name=f"Real_{col}"),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=predictions.index, y=predictions[col], mode='lines', name=f"Predicted_{col}"),
            row=1, col=1
        )
    
    # Error de predicción (diferencia entre predicción y valor real)
    for col in y_train.columns:
        error = predictions[col] - y_train[col]
        fig.add_trace(
            go.Scatter(x=y_train.index, y=error, mode='lines', name=f"Error_{col}"),
            row=1, col=2
        )
    
    # Ajustes de diseño
    fig.update_layout(
        title=f'Comparison between Real and Predicted Values - {model_name}',
        height=600,
        showlegend=True
    )
    
    # Guardar el gráfico como archivo HTML
    html_path = f"{model_name}_comparison.html"
    fig.write_html(html_path)
    
    print(f"HTML file saved as: {html_path}")
