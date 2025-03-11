"""
train traffic classifier
"""

import numpy as np
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
import mlflow
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import pandas as pd
import os

from utils.constants import TRAFFIC_CLASSIFICATION_MODEL_FILENAME
from utils.plots import generate_confusion_matrix_plot, xgboost_plot_features_relevance


def train(data_train_filepath: str, results_folder_path: str) -> str:
    """
    Train traffic classification model
    :param data_train_filepath: data file path
    :param results_folder_path: results folder path
    """
    # read traffic data
    print("read train traffic data")
    traffic_df = pd.read_csv(data_train_filepath)

    # get features and label
    y_train = traffic_df.pop('Label')
    X_train = traffic_df.copy()

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
    model_filepath = os.path.join(results_folder_path, TRAFFIC_CLASSIFICATION_MODEL_FILENAME)
    best_model.save_model(model_filepath)
    # log model as artifact
    mlflow.log_artifact(model_filepath)

    # plot model features relevance
    features_relevance_model_plot = os.path.join(results_folder_path, "features_relevance_model.png")
    xgboost_plot_features_relevance(best_model, features_relevance_model_plot)
    # log plot feature relevance model as artifact
    mlflow.log_artifact(features_relevance_model_plot)

    # return model filepath
    return model_filepath

