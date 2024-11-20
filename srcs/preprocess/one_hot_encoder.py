import numpy as np

def to_categorical(x, num_classes=None):
    """ Converts a class vector (integers) to binary class matrix.
    
        Args:
            x (numpy array): class vector to be converted.
            num_classes (int): total number of classes.
        
        Returns:
            A binary matrix representation of the input as a NumPy array. The class axis is placed last.
    
    """
    if not num_classes:
        num_classes = np.max(x) + 1
    one_hot_labels = np.zeros((x.size, num_classes))
    one_hot_labels[np.arange(x.size), x] = 1
    
    return one_hot_labels

