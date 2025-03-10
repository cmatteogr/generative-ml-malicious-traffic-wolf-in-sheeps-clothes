"""
traffic classifier pipeline
"""
import mlflow

from pipeline.preprocess.preprocess import preprocessing
from utils.constants import RELEVANT_COLUMNS, VALID_TRAFFIC_TYPES, VALID_PORT_RANGE

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Create a new MLFlow Experiment
mlflow.set_experiment("gen_ml_malicious")

# Start an MLFlow run
with mlflow.start_run():

    base_traffic_filepath = '/home/cesarealice/PycharmProjects/generative-ml-malicious-traffic-wolf-in-sheeps-clothes/data/Weekly-WorkingHours_report.csv'

    base_traffic_cleaned_filepath = preprocessing(base_traffic_filepath, relevant_column=RELEVANT_COLUMNS,
                                                  valid_traffic_types=VALID_TRAFFIC_TYPES,
                                                  valid_port_range=VALID_PORT_RANGE)
    print(base_traffic_cleaned_filepath)






