import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_accuracy(predictions, y):
    predictions = np.argmax(predictions, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    return accuracy

def calculate_recall(y_pred, y_true):
    """
    Calculates recall for binary classification.
    Args:
        y_pred (numpy.ndarray): One-hot encoded predicted values (e.g., [[1, 0], [0, 1]]).
        y_true (numpy.ndarray): True labels (e.g., [0, 1, 1, 0]).
    Returns:
        float: Recall value.
    """
    # True Positives (TP) and False Negatives (FN)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return recall

def calculate_precision(y_pred, y_true):
    """
    Calculates precision for binary classification.
    Args:
        y_pred (numpy.ndarray): One-hot encoded predicted values (e.g., [[1, 0], [0, 1]]).
        y_test (numpy.ndarray): True labels (e.g., [0, 1, 1, 0]).
    Returns:
        float: Precision value.
    """
    # True Positives (TP) and False Positives (FP)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return precision

def calculate_f1(y_pred, y_true):
    """
    Calculates F1 score for binary classification.
    Args:
        y_pred (numpy.ndarray): One-hot encoded predicted values (e.g., [[1, 0], [0, 1]]).
        y_true (numpy.ndarray): True labels (e.g., [0, 1, 1, 0]).
    Returns:
        float: F1 score.
    """
    precision = calculate_precision(y_pred, y_true)
    recall = calculate_recall(y_pred, y_true)

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def confusion_matrix(y_pred, y_true, class_names=['Malignant', 'Benign']):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    if class_names is None:
        class_names = ['Class 0', 'Class 1']
    
    cm = np.array([[tp, fn], [fp, tn]])
    
    print(f"\nTrue Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=False)
    
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

