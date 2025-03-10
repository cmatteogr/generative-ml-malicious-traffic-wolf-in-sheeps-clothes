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
from utils.plots import generate_confusion_matrix_plot


def train(data_train_filepath, results_folder_path):
    """
    Train traffic classification model
    :param data_train_filepath: data file path
    :param results_folder_path: results folder path
    """
    traffic_df = pd.read_csv(data_train_filepath)

    # get features and label
    y_train = traffic_df.pop('Label')
    X_train = traffic_df.copy()

    # Define XGBoost model
    xgb_model = XGBClassifier(objective='multi:softprob', num_class=len(np.unique(y_train)), device="cuda",
                              eval_metric='mlogloss', use_label_encoder=False, tree_method="gpu_hist")
    # Define parameter grid for tuning
    param_grid = {
        'learning_rate': Real(1e-3, 3, prior='log-uniform'),
        'max_depth': Integer(3, 30, prior='uniform'),
        'n_estimators': Integer(5, 500, prior='log-uniform'),
        'booster': Categorical(['gbtree', 'gblinear', 'dart']),
    }

    # log-uniform: understand as search over p = exp(x) by varying x
    opt = BayesSearchCV(xgb_model, param_grid, n_iter=50, random_state=0, verbose=2)

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
    mlflow.log_metric('f1_macro', f1_macro)
    mlflow.log_metric('f1_weighted', f1_macro)

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


