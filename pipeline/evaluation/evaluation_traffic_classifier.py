"""
Evaluation step
"""

import xgboost as xgb
import pandas as pd
from sklearn.metrics import f1_score

def evaluation(model_filepath: str, data_test_filepath: str):
    """
    evaluate classifier on test data
    :param model_filepath: model filepath
    :param data_test_filepath: test data filepath
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

    # print F1 metrics
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
