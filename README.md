# Multilayer-Perceptron 42
## Breast Cancer Classification with Neural Networks

## Overview
This project implements a multilayer perceptron (neural network) to classify breast cancer as either malignant or benign using the [Wisconsin](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names) Breast Cancer dataset. The implementation includes a complete neural network framework with various components for training, evaluation, and visualization.


## Dataset
The project uses the Wisconsin Breast Cancer Dataset which contains:

- Features computed from breast mass images
- Binary classification: Malignant (M) or Benign (B)
- 30 features including radius, texture, perimeter, area, smoothness, etc.
- Each row represents a single case with corresponding measurements


## Requirements
numpy
matplotlib
pandas
scikit-learn
seaborn


## Project Structure
```
├── data/
│   └── data.csv         # Wisconsin Breast Cancer Dataset
├── srcs/
│   ├── model/
│   │   ├── activation.py    # Activation functions (ReLU, Sigmoid, Softmax)
│   │   ├── layer.py        # Neural network layers
│   │   ├── loss.py         # Loss functions
│   │   ├── metrics.py      # Evaluation metrics
│   │   ├── neural_network.py # Main neural network implementation
│   │   ├── optimizers.py   # Optimization algorithms (Adam, RMSProp)
│   │   └── preprocess.py   # Data preprocessing utilities
│   └── programs/
│       └── multiple_model_train.py # Training script for multiple models
```


## Features

### Neural Network Components

- **Layers**: Dense layers with customizable neurons

- **Activation Functions**: 
    - **ReLU**
    - **Sigmoid**
    - **Softmax**

- **Optimizers**:
    - **Adam**
    - **RMSProp**
    - **SGD**

- **Loss Functions**:
    - **Binary Cross-Entropy**
    - **Mean Squared Error**
    - **Categorical Cross-Entropy**


### Training Features
- **Batch training support**
- **Early stopping with patience**
- **Learning rate decay**
- **Validation set evaluation**
- **Model saving and loading**

### Evaluation Metrics
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

### Visualization
- **Training history plots**
- **Learning curves**
- **Confusion matrix visualization**


## Usage

1. Install dependencies:
```bash
make virtual
make create # Create data, depo and predictions folders
```

2. Prepare the dataset:
```bash:README.md
python programs/preprocess.py --input path/to/dataset.csv
```

3. Train multiple models:
```bash:README.md
python programs/multiple_model_train.py --train path/to/train.csv --valid path/to/valid.csv --epochs 50 --batch_size 8 --learning_rate 0.005
```

4. Use the trained model for predictions:
```bash:README.md
python programs/predict.py --model path/to/model.npy --input path/to/valid.csv --output path/to/predictions.csv
```

### Performance Metrics
The model evaluates performance using:
- **Confusion Matrix**
- **Precision, Recall, and F1 Score**
- **Training and Validation Accuracy/Loss Curves**


### Contributing
Feel free to submit issues and enhancement requests.
