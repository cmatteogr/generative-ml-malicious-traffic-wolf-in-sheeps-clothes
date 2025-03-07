import numpy as np
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
import mlflow
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def train(data_train_filepath):

    traffic_df = pd.read_csv(data_train_filepath)

    # get features and label
    y_train = traffic_df.pop('Label')
    X_train = traffic_df.copy()

    # Define XGBoost model
    xgb_model = XGBClassifier(objective='multi:softprob', num_class=len(np.unique(y_train)),
                              eval_metric='mlogloss', use_label_encoder=False, tree_method='gpu_hist')
    # Define parameter grid for tuning
    param_grid = {
        'learning_rate': Real(1e-3, 3, prior='log-uniform'),
        'max_depth': Real(3, 30, prior='uniform'),
        'n_estimators': Real(5, 1e+6, prior='log-uniform'),
        'booster': Categorical(['gbtree', 'gblinear', 'dart']),
    }

    # log-uniform: understand as search over p = exp(x) by varying x
    opt = BayesSearchCV(xgb_model, param_grid, n_iter=50, random_state=0)

    # executes bayesian optimization
    _ = opt.fit(X_train, y_train)

    # Grid Search with F1 scoring
    grid_search = GridSearchCV(xgb_model, param_grid, scoring='f1_macro', cv=3, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best model from grid search
    best_model = grid_search.best_estimator_

    # Predict on test set
    y_pred = best_model.predict(X_train)

    # Compute F1-score directly
    f1_macro = f1_score(y_train, y_pred, average='macro')
    f1_weighted = f1_score(y_train, y_pred, average='weighted')

    mlflow.log_metric('f1_macro', f1_macro)

    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
