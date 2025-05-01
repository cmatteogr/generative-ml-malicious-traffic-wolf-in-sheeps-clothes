"""
traffic classifier pipeline
"""
import os
import shutil
from ml_models.malicious_traffic_classifier import MaliciousTrafficClassifierModel
from mlflow_api.mlflow_client import MlflowClientHandler
import mlflow
from pipeline.evaluation.evaluation_traffic_classifier import evaluation
from pipeline.preprocess.preprocess_generator import preprocessing
from pipeline.train.train_traffic_latent_variable_gan import train
from utils.constants import RELEVANT_COLUMNS, VALID_TRAFFIC_TYPES, VALID_PORT_RANGE

# Set the MLflow tracking server URI to log experiments and models
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Create or set the MLflow experiment under which runs will be logged
mlflow.set_experiment("generative_ml_malicious_generator")

# Define the name for the registered model in MLflow Model Registry
model_deployed_name = 'generative_ml_malicious_traffic_generator'

# Start an MLflow run context to log parameters, metrics, and artifacts
with mlflow.start_run() as run:
    # Define the local directory path to store results and artifacts generated during the run
    results_folder_path = './results'

    # Define the file path for the base input traffic data CSV file
    base_traffic_filepath = './data/Weekly-WorkingHours_report.csv'

    # --- Preprocessing Step ---
    # Execute the preprocessing function to clean and prepare the data
    # It returns paths to the processed train/test sets and any preprocessing artifacts (like scalers)
    train_traffic_filepath, test_traffic_filepath, preprocess_artifacts = preprocessing(
        base_traffic_filepath,
        results_folder_path,
        relevant_column=RELEVANT_COLUMNS,         # Columns to keep
        valid_traffic_types=VALID_TRAFFIC_TYPES, # Allowed traffic types
        valid_port_range=VALID_PORT_RANGE        # Allowed port range
    )

    # --- Training Step ---
    # Train the traffic classifier model using the preprocessed training data
    # It returns the file path to the saved trained model artifact
    model_filepath = train(train_traffic_filepath, results_folder_path)
