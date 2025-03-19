# Project Description

The idkROM project is a machine learning application designed to facilitate the modeling and prediction of data using various algorithms. It provides a user-friendly interface and supports multiple data sources, allowing users to load, preprocess, and analyze data efficiently.

## Purpose

The primary purpose of this project is to implement different machine learning models, including neural networks, Gaussian processes, and radial basis function models. The application aims to provide users with tools to train these models, evaluate their performance, and make predictions based on input data.

## Features

- **Data Loading**: The application can load raw and preprocessed data from various sources, making it flexible for different use cases.
- **Data Preprocessing**: It includes functionality for splitting datasets and normalizing inputs and outputs to ensure optimal model performance.
- **Model Training**: Users can train different types of models, including neural networks, Gaussian processes, and radial basis functions, with the ability to search for the best hyperparameters.
- **Evaluation and Prediction**: The application provides methods for evaluating model performance and making predictions on test datasets.

## Usage

To use the idkROM application, follow these steps:

1. **Install Dependencies**: Ensure that all required libraries and dependencies are installed.
2. **Prepare Data**: Have your data ready in CSV format, either raw or preprocessed.
3. **Run the Application**: Execute the main script with the appropriate command-line arguments to load data, preprocess it, and train the desired model.

Example command:
```
python main.py <ruta_csv> <tipo_datos: raw/processed> <modelo: neural_network/gaussian_process/rbf>
```

4. **View Results**: After training, the application will output evaluation metrics and predictions based on the test dataset.

This project serves as a comprehensive tool for users interested in applying machine learning techniques to their data analysis tasks.