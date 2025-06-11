"""
traffic classifier pipeline
"""
import mlflow
from pipeline.preprocess.preprocess_generator_vae_gan import preprocessing
from pipeline.train.train_traffic_b_tcvae_gan import train as train_generator
from pipeline.train.train_traffic_classifier import train as train_discriminator
from utils.constants import RELEVANT_COLUMNS, VALID_TRAFFIC_TYPES, VALID_PORT_RANGE, MLFLOW_HOST, MLFLOW_PORT

# Set the MLflow tracking server URI to log experiments and models
mlflow.set_tracking_uri(uri=f"http://{MLFLOW_HOST}:{MLFLOW_PORT}")

# Create or set the MLflow experiment under which runs will be logged
mlflow.set_experiment("generative_ml_malicious_generator_b_tcvae_gan")

# Define the name for the registered model in MLflow Model Registry
model_deployed_name = 'generative_ml_malicious_generator_b_tcvae_gan'

# Start an MLflow run context to log parameters, metrics, and artifacts
with mlflow.start_run() as run:
    # Define the local directory path to store results and artifacts generated during the run
    results_folder_path = './results/generative_b_tcvae_gan'

    # Define the file path for the base input traffic data CSV file
    base_traffic_filepath = './data/Weekly-WorkingHours_report.csv'

    # --- Preprocessing Data ---
    train_traffic_filepath, test_traffic_filepath, preprocess_artifacts = preprocessing(
        base_traffic_filepath,
        results_folder_path,
        relevant_column=RELEVANT_COLUMNS,         # Columns to keep
        valid_traffic_types=VALID_TRAFFIC_TYPES, # Allowed traffic types
        valid_port_range=VALID_PORT_RANGE        # Allowed port range
    )

    # --- Training Discriminator ---
    discriminator_filepath, discriminator_onnx_filepath = train_discriminator(train_traffic_filepath, results_folder_path)

    # --- Training Generator ---
    # discriminator_filepath='./results/generative_b_tcvae_gan/xgb_server_traffic_classifier.json'
    model_filepath = train_generator(
        traffic_data_filepath=train_traffic_filepath,
        results_folder_path=results_folder_path,
        discriminator_filepath=discriminator_filepath,
        batch_size=4096)
