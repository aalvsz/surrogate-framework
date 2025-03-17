import os
import torch
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Template
from sklearn.metrics import mean_squared_error, r2_score

class ModelReportGenerator:
    def __init__(self, model, train_losses, val_losses, X_train, y_train, X_test, y_test, model_name="model"):
        self.model = model
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name

        # Detect model type and set hyperparameters accordingly
        self.hyperparameters = {}
        if hasattr(model, 'hidden_layers'):  # Neural Network
            self.hyperparameters = {
                'Hidden Layers': model.hidden_layers,
                'Neurons per Layer': model.neurons_per_layer,
                'Learning Rate': model.learning_rate,
                'Epochs': model.num_epochs,
                'Optimizer': model.optimizer,
                'Activation Function': model.activation_function
            }
        elif hasattr(model, 'kernel'):  # Gaussian Process
            self.hyperparameters = {
                'Kernel': model.kernel,
                'Noise': model.noise,
                'Optimizer': model.optimizer
            }
        else:
            raise ValueError("Model type not recognized")

        # Output directory
        self.output_dir = os.path.join(os.getcwd(), 'results', model_name)
        os.makedirs(self.output_dir, exist_ok=True)

    def create_convergence_graph(self):
        """Create convergence plot for training"""
        if not self.train_losses:
            return  # No convergence plot for Gaussian Process
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(self.train_losses))), y=self.train_losses, mode='lines', name='Training Loss'))
        if self.val_losses:
            fig.add_trace(go.Scatter(x=list(range(len(self.val_losses))), y=self.val_losses, mode='lines', name='Validation Loss'))

        fig.update_layout(
            title="Convergence Plot",
            xaxis_title="Epochs",
            yaxis_title="Loss",
            template="plotly_dark"
        )
        fig.write_html(os.path.join(self.output_dir, 'convergence_plot.html'))

    def create_metrics_graph(self, metrics_df):
        """Generate metrics graph like MSE and R²"""
        fig = go.Figure()
        fig.add_trace(go.Bar(x=metrics_df.columns, y=metrics_df.iloc[0, :-1], name='Metrics'))

        fig.update_layout(
            title="Model Metrics",
            xaxis_title="Metric",
            yaxis_title="Value",
            template="plotly_dark"
        )
        fig.write_html(os.path.join(self.output_dir, 'metrics_plot.html'))

    def generate_html_report(self, metrics_df):
        """Generate an HTML report summarizing the training and evaluation"""
        template_path = os.path.join(os.path.dirname(__file__), 'report_template.html')
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                html_template = f.read()
        else:
            html_template = """
            <html>
            <head><title>{{ model_name }} - Training Summary</title></head>
            <body>
                <h1>{{ model_name }} Training Summary</h1>
                <h2>Metrics</h2>
                <table border="1">
                    <tr><th>MSE</th><td>{{ mse }}</td></tr>
                    <tr><th>R²</th><td>{{ r2 }}</td></tr>
                </table>
                <h2>Hyperparameters</h2>
                <table border="1">
                    {% for param, value in hyperparameters.items() %}
                    <tr><th>{{ param }}</th><td>{{ value }}</td></tr>
                    {% endfor %}
                </table>
                <h2>Convergence Plot</h2>
                <iframe src="convergence_plot.html" width="100%" height="500"></iframe>
                <h2>Metrics Plot</h2>
                <iframe src="metrics_plot.html" width="100%" height="500"></iframe>
            </body>
            </html>
            """
        
        template = Template(html_template)
        html_output = template.render(
            model_name=self.model_name,
            mse=metrics_df['MSE'][0],
            r2=metrics_df['R²'][0],
            hyperparameters=self.hyperparameters
        )

        with open(os.path.join(self.output_dir, f'{self.model_name}_training_summary.html'), 'w') as f:
            f.write(html_output)
        print(f"Report saved at: {os.path.join(self.output_dir, f'{self.model_name}_training_summary.html')}")

    def save_model_and_metrics(self):
        """Save metrics and generate the complete HTML report"""
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        metrics_df = pd.DataFrame([{'MSE': mse, 'R²': r2}])

        self.create_convergence_graph()
        self.create_metrics_graph(metrics_df)
        self.generate_html_report(metrics_df)

        print("Model and metrics saved successfully!")
