"""
plots utils
"""
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb
import matplotlib.cm as cm
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
import pandas as pd
import os


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
    fig, ax = plt.subplots(figsize=(5, 5))  # Set figure size
    disp.plot(cmap="Blues", ax=ax)
    plt.title("Confusion Matrix")  # Add title
    plt.savefig(plot_filepath, dpi=300, bbox_inches="tight")

def xgboost_plot_features_relevance(model, plot_features_relevance_path: str):
    """
    XGBoost features relevance plot
    :param model: XGBoost model
    :param plot_features_relevance_path: path to save plot
    """
    xgb.plot_importance(model, importance_type="gain")  # "gain" is recommended
    plt.title("Feature Importance (Gain)")
    plt.savefig(plot_features_relevance_path, dpi=300, bbox_inches="tight")


def plot_instances_by_features(df, feature_a, feature_b, feature_c, labels, plot_filepath: str):
    # Plot in 3D
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Generate a dynamic colormap for labels
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    colormap = cm.get_cmap("tab10", num_labels)
    # Assign colors based on labels
    for i, label in enumerate(unique_labels):
        idx = labels == label
        ax.scatter(df.loc[idx, feature_a], df.loc[idx, feature_b], df.loc[idx, feature_c],
                   color=colormap(i), label=f'Class {label}', alpha=0.7)

    ax.set_xlabel(feature_a)
    ax.set_ylabel(feature_b)
    ax.set_zlabel(feature_c)
    ax.set_title("3D Scatter Plot of Instances by Label")
    ax.legend()

    plt.savefig(plot_filepath, dpi=300, bbox_inches="tight")


def plot_instances_by_features_interactive(df, feature_a, feature_b, feature_c):

    # Create an interactive 3D scatter plot using Plotly
    fig = px.scatter_3d(df, x=feature_a, y=feature_b, z=feature_c,
                        color="Label", title="Interactive 3D Scatter Plot",
                        opacity=0.7)

    # Show the interactive plot
    fig.show()

def plot_latent_space_vae(z_tensors, z_labels, results_filepath,
                          market_size=2,
                          plot_columns=['PCA1', 'PCA2', 'PCA3'],
                          plot_title='VAE Latent Space (PCA - 3 Components)'):
    # reduce dimensionality
    pca = PCA(n_components=3)
    # apply PCA transformation
    z_transformed_pca = pca.fit_transform(z_tensors)
    # plot the latent space in 3D
    df_pca_plot = pd.DataFrame(z_transformed_pca, columns=plot_columns)
    df_pca_plot['Label'] = z_labels
    fig_pca = px.scatter_3d(df_pca_plot,
                            x=plot_columns[0],
                            y=plot_columns[1],
                            z=plot_columns[2],
                            color='Label',
                            title=plot_title,
                            labels={'color': 'True Label'})
    fig_pca.update_traces(marker=dict(size=market_size, opacity=0.7))
    # save plot
    fig_pca.write_html(results_filepath)

