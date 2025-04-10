import os
import matplotlib.pyplot as plt
import numpy as np

def ensure_dir_exists(dir_path):
    """
    Ensure that the directory exists; if not, create it.

    Args:
        dir_path (str): Path to the directory.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def save_figure(fig, path, dpi=300):
    """
    Save a Matplotlib figure to a file.

    Args:
        fig (matplotlib.figure.Figure): Figure object to save.
        path (str): File path where the figure will be saved.
        dpi (int): Resolution in dots per inch.
    """
    ensure_dir_exists(os.path.dirname(path))
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved to {path}")

def plot_loss(history, output_path=None):
    """
    Plot training and validation loss over epochs.

    Args:
        history: Keras History object from model training.
        output_path (str, optional): Path to save the loss plot.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(history.history.get('loss', []), label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    if output_path:
        save_figure(plt.gcf(), output_path)
    plt.show()

def plot_accuracy(history, output_path=None):
    """
    Plot training and validation accuracy over epochs.

    Args:
        history: Keras History object from model training.
        output_path (str, optional): Path to save the accuracy plot.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(history.history.get('accuracy', []), label='Train Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    if output_path:
        save_figure(plt.gcf(), output_path)
    plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues, output_path=None):
    """
    Plot a confusion matrix.

    Args:
        cm (array-like): Confusion matrix.
        classes (list): List of class labels.
        normalize (bool): Whether to normalize the confusion matrix.
        title (str): Title for the plot.
        cmap: Colormap instance for the plot.
        output_path (str, optional): Path to save the plot.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if output_path:
        save_figure(plt.gcf(), output_path)
    plt.show()
