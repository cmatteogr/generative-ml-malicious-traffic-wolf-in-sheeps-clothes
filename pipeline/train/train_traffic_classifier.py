"""
train traffic classifier
"""

import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
import mlflow
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import pandas as pd
import os

from utils.constants import TRAFFIC_CLASSIFIER_MODEL_FILENAME
from utils.plots import generate_confusion_matrix_plot, xgboost_plot_features_relevance
from utils.utils import generate_profiling_report
from skl2onnx.common.data_types import FloatTensorType
import skl2onnx


def train(data_train_filepath: str, results_folder_path: str) -> tuple:
    """
    Train traffic classification model
    :param data_train_filepath: data file path
    :param results_folder_path: results folder path
    """
    # read traffic data
    print("read train traffic data")
    traffic_df = pd.read_csv(data_train_filepath)
    title = "Train dataset Profiling"
    report_name = 'traffic_train_dataset_profiling_discriminator'
    report_filepath = os.path.join(results_folder_path, f"{report_name}.html")
    type_schema = {'Label': "categorical"}
    generate_profiling_report(report_filepath=report_filepath, title=title, data_filepath=data_train_filepath,
                              type_schema=type_schema, minimal=True)

    # get features and label
    y_train = traffic_df.pop('Label')
    X_train = traffic_df.copy()
    n_features = len(X_train.columns)

    # Define XGBoost model
    xgb_model = XGBClassifier(objective='multi:softprob', num_class=len(np.unique(y_train)), device="cuda",
                              eval_metric='mlogloss')
    # define parameter ranges for tuning
    param_grid = {
        'learning_rate': Real(1e-3, 3, prior='log-uniform'),
        'max_depth': Integer(3, 30),
        # NOTE: in run time a warning appears because max_depth applies only to gbtree and dart boosters, as they use decision trees
        'n_estimators': Integer(5, 500, prior='log-uniform'),
        'booster': Categorical(['gbtree', 'gblinear', 'dart']),
    }

    # set the optimization method applied
    mlflow.log_param("optimization_method", "BayesSearchCV")
    # bayesian cv models
    opt = BayesSearchCV(xgb_model, param_grid, n_iter=50, cv=5, random_state=0, verbose=2)

    # executes bayesian optimization
    _ = opt.fit(X_train, y_train)
    print(opt.score(X_train, y_train))

    # Best model from grid search
    best_model = opt.best_estimator_

    # Predict on test set
    y_pred = best_model.predict(X_train)

    # Compute F1-score directly
    f1_macro = f1_score(y_train, y_pred, average='macro')
    f1_weighted = f1_score(y_train, y_pred, average='weighted')

    # print F1 metrics
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")

    # log metric in MLFlow
    mlflow.log_metric('f1_macro_train', f1_macro)
    mlflow.log_metric('f1_weighted_train', f1_macro)

    # generate confusion matrix
    confusion_matrix_plot_filepath = os.path.join(results_folder_path, "confusion_matrix_train.png")
    generate_confusion_matrix_plot(y_train, y_pred, confusion_matrix_plot_filepath)
    # confusion matrix as artifact
    mlflow.log_artifact(confusion_matrix_plot_filepath)

    # save model
    model_filepath = os.path.join(results_folder_path, TRAFFIC_CLASSIFIER_MODEL_FILENAME)
    best_model.save_model(model_filepath)
    # log model as artifact
    mlflow.log_artifact(model_filepath)

    # save model onnx
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    # convert to ONNX and save
    model_onnx_filepath = os.path.join(results_folder_path, 'xgb_server_traffic_classifier.onnx')
    onnx_model = skl2onnx.convert_sklearn(xgb_model, initial_types=initial_type)
    with open(model_onnx_filepath, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # plot model features relevance
    features_relevance_model_plot = os.path.join(results_folder_path, "features_relevance_model.png")
    xgboost_plot_features_relevance(best_model, features_relevance_model_plot)
    # log plot feature relevance model as artifact
    mlflow.log_artifact(features_relevance_model_plot)

    # return model filepath
    return model_filepath, model_onnx_filepath

