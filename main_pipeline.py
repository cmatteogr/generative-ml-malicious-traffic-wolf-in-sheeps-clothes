"""
traffic classifier pipeline
"""
import mlflow

from pipeline.evaluation.evaluation_traffic_classifier import evaluation
from pipeline.preprocess.preprocess import preprocessing
from pipeline.train.train_traffic_classifier import train
from utils.constants import RELEVANT_COLUMNS, VALID_TRAFFIC_TYPES, VALID_PORT_RANGE

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Create a new MLFlow Experiment
mlflow.set_experiment("gen_ml_malicious")

# Start an MLFlow run
with mlflow.start_run():
    # results folder
    results_folder_path = './results'

    # define base traffic filepath
    base_traffic_filepath = './data/Weekly-WorkingHours_report.csv'

    # preprocess data
    train_traffic_filepath, test_traffic_filepath = preprocessing(base_traffic_filepath, results_folder_path,
                                                                  relevant_column=RELEVANT_COLUMNS,
                                                                  valid_traffic_types=VALID_TRAFFIC_TYPES,
                                                                  valid_port_range=VALID_PORT_RANGE)
    # train traffic classifier model
    model_filepath = train(train_traffic_filepath, results_folder_path)

    # evaluate model
    evaluation(model_filepath, test_traffic_filepath, results_folder_path)

