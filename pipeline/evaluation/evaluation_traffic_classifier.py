"""
Evaluation step
"""
import os
import xgboost as xgb
import pandas as pd
from sklearn.metrics import f1_score
import mlflow
from utils.plots import generate_confusion_matrix_plot


def evaluation(model_filepath: str, data_test_filepath: str, results_folder_path: str):
    """
    evaluate classifier on test data
    :param model_filepath: model filepath
    :param data_test_filepath: test data filepath
    :param results_folder_path: results folder path
    """
    traffic_df = pd.read_csv(data_test_filepath)

    # get features and label
    y_test = traffic_df.pop('Label')
    X_test = traffic_df.copy()

    # Load the model
    loaded_model = xgb.XGBClassifier()
    loaded_model.load_model(model_filepath)

    # Use the loaded model
    y_pred = loaded_model.predict(X_test)

    # Compute F1-score directly
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    # generate confusion matrix
    confusion_matrix_plot_filepath = os.path.join(results_folder_path, "confusion_matrix_evaluation.png")
    generate_confusion_matrix_plot(y_test, y_pred, confusion_matrix_plot_filepath)
    # confusion matrix as artifact
    mlflow.log_artifact(confusion_matrix_plot_filepath)

    # print F1 metrics
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")

    # log metric in MLFlow
    mlflow.log_metric('f1_macro', f1_macro)
    mlflow.log_metric('f1_weighted', f1_macro)
