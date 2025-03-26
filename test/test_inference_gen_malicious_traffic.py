from datetime import datetime, timedelta

import mlflow
import pytest

import pandas as pd

def test_inference_price_regression_model(input_model):
    # define the model filepath
    model_artifact_path = f"/home/cesarealice/PycharmProjects/generative-ml-malicious-traffic-wolf-in-sheeps-clothes/results/malicious_traffic_classifier_model"
    # load model
    traffic_classification_model = mlflow.pyfunc.load_model(model_artifact_path)
    # load inference dataset
    data_filepath ='/home/cesarealice/PycharmProjects/generative-ml-malicious-traffic-wolf-in-sheeps-clothes/test/inference.csv'
    traffic_df = pd.read_csv(data_filepath)
    traffic_data = {
        'traffic_df': traffic_df,
    }

    predictions = traffic_classification_model.predict(traffic_data)
    print(predictions)
    pass


def test_split():
    filepath = '/home/cesarealice/PycharmProjects/generative-ml-malicious-traffic-wolf-in-sheeps-clothes/data/Weekly-WorkingHours_report.csv'
    data: pd.DataFrame = pd.read_csv(filepath)
    data = data.sample(50)
    data.to_csv('/home/cesarealice/PycharmProjects/generative-ml-malicious-traffic-wolf-in-sheeps-clothes/test/inference.csv', index=False)