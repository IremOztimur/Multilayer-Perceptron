import numpy as np

def calculate_accuracy(predictions, y):
    predictions = np.argmax(predictions, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    return accuracy

def calculate_recall(y_pred, y_test):
    """
    Calculates recall for binary classification.
    Args:
        y_pred (numpy.ndarray): One-hot encoded predicted values (e.g., [[1, 0], [0, 1]]).
        y_test (numpy.ndarray): True labels (e.g., [0, 1, 1, 0]).
    Returns:
        float: Recall value.
    """
    # True Positives (TP) and False Negatives (FN)
    tp = np.sum((y_pred == 1) & (y_test == 1))
    fn = np.sum((y_pred == 0) & (y_test == 1))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return recall

def calculate_precision(y_pred, y_test):
    """
    Calculates precision for binary classification.
    Args:
        y_pred (numpy.ndarray): One-hot encoded predicted values (e.g., [[1, 0], [0, 1]]).
        y_test (numpy.ndarray): True labels (e.g., [0, 1, 1, 0]).
    Returns:
        float: Precision value.
    """
    # True Positives (TP) and False Positives (FP)
    tp = np.sum((y_pred == 1) & (y_test == 1))
    fp = np.sum((y_pred == 1) & (y_test == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return precision

def calculate_f1(y_pred, y_test):
    """
    Calculates F1 score for binary classification.
    Args:
        y_pred (numpy.ndarray): One-hot encoded predicted values (e.g., [[1, 0], [0, 1]]).
        y_test (numpy.ndarray): True labels (e.g., [0, 1, 1, 0]).
    Returns:
        float: F1 score.
    """
    precision = calculate_precision(y_pred, y_test)
    recall = calculate_recall(y_pred, y_test)

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1