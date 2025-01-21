# Quantum Neural Network Classification Models using Diabetes Dataset
This project explores the use of Quantum Neural Networks (QNN) for classification tasks using the Diabetes dataset. The goal is to demonstrate the application of quantum computing techniques, specifically quantum circuits and machine learning, to a classical machine learning problem: diabetes prediction based on clinical data.

## Author
Galih Putra Pratama

## Overview
In this project, the Quantum Neural Network is used to classify diabetes outcomes (0 or 1) based on clinical features. The dataset used is the Diabetes dataset, which includes information like age, BMI, insulin levels, and more. The project applies quantum machine learning techniques to predict whether a patient has diabetes, comparing the quantum model's performance to traditional machine learning models.

### Key Components:
- Dataset: The Diabetes dataset contains clinical data and a binary outcome indicating whether a patient has diabetes (Outcome).
- Quantum Neural Network (QNN): The model is built using a quantum circuit for data encoding and a quantum neural network for classification.
- Quantum Circuit Design: The quantum circuit uses 8 qubits (one for each feature in the dataset) to encode input data and perform a series of rotations for feature transformation.
- Optimization: The model is trained using the COBYLA optimizer, and a callback is used to track the progress of the training.
- Evaluation: The trained QNN is evaluated using classification metrics like precision, recall, accuracy, and F1-score, and confusion matrices for both training and testing sets are generated. Additionally, decision boundaries are visualized using PCA (Principal Component Analysis) to represent the model's classification regions.
### Requirements
To run the code, you'll need the following libraries:
- qiskit==0.46.1: For building and simulating quantum circuits.
- qiskit-aer: Backend for quantum simulators.
- qiskit_machine_learning==0.7.2: For integrating machine learning algorithms with quantum circuits.
- pylatexenc: For handling LaTeX expressions.
  
To install the dependencies, run the following:

```bash
pip install qiskit==0.46.1 qiskit-aer qiskit_machine_learning==0.7.2 pylatexenc
```
## Data Preprocessing
The Diabetes dataset is loaded from a CSV file, with preprocessing steps that include:
-Feature Selection: The dataset's features are separated from the target variable (Outcome).
-Normalization: The features are scaled using MinMaxScaler for better performance in quantum circuits.
-Data Splitting: The dataset is split into training and testing sets (80:20 split).

## Quantum Circuit Design
The quantum circuit is designed with the following steps:
- Data Encoding: Input data is encoded using rotations (RX, RY, and RZ) for each qubit.
- Quantum Operations: The circuit includes quantum gates like CX for entangling qubits and additional rotations for ansatz.
- Optimization: Parameters (theta, phi) are optimized using the COBYLA algorithm.
  
## Model Training
The QNN classifier is trained using the training data, and the time taken for training is recorded.

## Evaluation Metrics
The performance of the Quantum Neural Network is evaluated using:
- Classification Report: Precision, recall, accuracy, and F1-score.
- Confusion Matrix: To visualize the classification performance on training and test data.
- ROC Curve and AUC: To evaluate the model's discrimination ability.
  
## Visualization
- Decision Boundary: The decision boundary is visualized using PCA to reduce dimensionality and show how the model classifies the data.
- Learning Curve: The objective function value is plotted over iterations during training.
  
## Results
After training the model, confusion matrices and classification reports for both training and testing datasets are provided, showing the quantum model’s accuracy in predicting diabetes outcomes.

## Running the Code
To run the project, follow these steps:

1. Mount Google Drive (if using Google Colab):
```bash
from google.colab import drive
drive.mount('/content/drive')
```
2. Run the Main Function: The fit() method trains the Quantum Neural Network model. Example:
```bash
qnn.fit(x_train, y_train)
```
3. Evaluate the Model: Once the model is trained, you can use the predict() method to make predictions and evaluate the results using classification_report() and confusion_matrix().

This README provides a clear and concise overview of your Quantum Neural Network project using the Diabetes dataset, explaining the process from data preprocessing to model evaluation.
