"""
Evaluation step
"""
import os
import xgboost as xgb
import pandas as pd
from sklearn.metrics import f1_score
import mlflow
from utils.plots import generate_confusion_matrix_plot, plot_instances_by_features
from sklearn.preprocessing import LabelEncoder
import joblib


def evaluation(model_filepath: str, data_test_filepath: str, results_folder_path: str,
               label_encoder_model_filepath:str) -> dict:
    """
    evaluate classifier on test data
    :param model_filepath: model filepath
    :param data_test_filepath: test data filepath
    :param results_folder_path: results folder path
    :param label_encoder_model_filepath: results folder path

    """
    print("read test traffic data")
    traffic_df = pd.read_csv(data_test_filepath)
    label_encoder_model: LabelEncoder = joblib.load(label_encoder_model_filepath)

    # get features and label
    y_test = traffic_df.pop('Label')
    y_test = label_encoder_model.transform(y_test)
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

    # generate plot feature relevance by instance in 3D
    feature_a = 'Idle Mean'
    feature_b = 'Packet Length Variance'
    feature_c = 'Total Length of Bwd Packets'
    instances_features_plot_filepath = os.path.join(results_folder_path, "relevant_features_instances_3d.png")
    plot_instances_by_features(traffic_df, feature_a, feature_b, feature_c, y_pred, instances_features_plot_filepath)
    # relevant features vs instances in 3D as artifact
    mlflow.log_artifact(confusion_matrix_plot_filepath)

    # print F1 metrics
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")

    # log metric in MLFlow
    mlflow.log_metric('f1_macro', f1_macro)
    mlflow.log_metric('f1_weighted', f1_macro)

    """# TODO: interactive PLOT
    traffic_df['Label'] = y_test.astype(str)
    plot_instances_by_features_interactive(traffic_df, feature_a, feature_b, feature_c)"""
    # return metric
    eval_data = {
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
    }

    return eval_data

