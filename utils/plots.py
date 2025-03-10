"""
plots utils
"""
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def generate_confusion_matrix_plot(y_base, y_pred, plot_filepath):
    """
    Generates a confusion matrix plot, save as a png file.
    :param y_base: y base data
    :param y_pred: y prediction data
    :param plot_filepath: path to save plot
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_base, y_pred)

    # Create confusion matrix display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # Plot and save the figure
    fig, ax = plt.subplots(figsize=(6, 6))  # Set figure size
    disp.plot(cmap="Blues", ax=ax)
    plt.title("Confusion Matrix")  # Add title
    plt.savefig(plot_filepath, dpi=300, bbox_inches="tight")