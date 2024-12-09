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
    
    

def plot_learning_curves(models_history, metrics=["loss", "accuracy"]):
    """
    Plots multiple learning curves (loss and accuracy) on the same graph.

    Args:
        models_history (dict): A dictionary where keys are model names, and values are dictionaries
                               containing 'train' and 'validation' metrics over epochs.
        metrics (list): List of metrics to plot. Default is ["loss", "accuracy"].

    Example of models_history:
        {
            "Model A": {
                "train": [losses_over_epochs, accuracies_over_epochs],
                "validation": [val_losses_over_epochs, val_accuracies_over_epochs]
            },
            "Model B": {
                "train": [losses_over_epochs, accuracies_over_epochs],
                "validation": [val_losses_over_epochs, val_accuracies_over_epochs]
            },
        }
    """
    for metric in metrics:
        plt.figure(figsize=(10, 6))

        for model_name, history in models_history.items():
            metric_idx = 0 if metric == "loss" else 1
            train_metric = history["train"][metric_idx]
            val_metric = history["validation"][metric_idx]
            
            plt.plot(train_metric, label=f"{model_name} - Train {metric.capitalize()}", marker='o')
            plt.plot(val_metric, label=f"{model_name} - Validation {metric.capitalize()}", marker='o', linestyle="dotted")

        plt.title(f"Comparison of {metric.capitalize()} Across Models", fontsize=14)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.legend(loc="best", fontsize=10)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
