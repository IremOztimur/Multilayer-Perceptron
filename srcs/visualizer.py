import matplotlib.pyplot as plt

def plot_metrics(history):
    """
    Plot loss and accuracy over epochs for training and validation.

    Args:
        history: Dictionary containing loss and accuracy metrics.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot Loss
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    if 'val_loss' in history:
        plt.plot(epochs, history['val_loss'], label='Val Loss', marker='o', linestyle = 'dotted')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_accuracy'], label='Train Accuracy', marker='o')
    if 'val_accuracy' in history:
        plt.plot(epochs, history['val_accuracy'], label='Val Accuracy', marker='o')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_step_loss(step_loss):
    plt.plot(step_loss)
    plt.title('Loss at Each Step')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.show()