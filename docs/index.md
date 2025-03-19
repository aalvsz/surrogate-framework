# idkROM Documentation

Welcome to the idkROM project documentation. This project is designed for loading, preprocessing, and modeling data using various MOR (Model order Reduction) techniques. Below you will find an overview of the project's structure, its components, and how to use it.

## Project Overview

idkROM is a ROM selection framework that allows users to load data from different sources, preprocess it, and apply various models such as neural networks, Gaussian processes, and radial basis functions. The framework is designed to be flexible and user-friendly, making it suitable for both beginners and experienced practitioners in the field of machine learning.

## Getting Started

To get started with idkROM, you need to have Python installed on your machine. You can then clone the repository and install the required dependencies. 

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd idkROM
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Usage

To run the application, use the following command:
```
python main.py <ruta_csv> <tipo_datos: raw/processed> <modelo: neural_network/gaussian_process/rbf>
```

Replace `<ruta_csv>` with the path to your CSV file, `<tipo_datos>` with either `raw` or `processed`, and `<modelo>` with the desired model type.

## Documentation Structure

- **Classes**: Detailed documentation of the classes used in the project.
- **Functions**: Comprehensive descriptions of the functions available in the project.
- **Descriptions**: General information about the project, its purpose, and usage guidelines.

For more detailed information, please refer to the respective sections in the documentation.