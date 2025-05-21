"""
traffic classifier pipeline
"""
import os
import shutil
from ml_models.malicious_traffic_classifier import MaliciousTrafficClassifierModel
from mlflow_api.mlflow_client import MlflowClientHandler
import mlflow
from pipeline.evaluation.evaluation_traffic_discriminator import evaluation
from pipeline.preprocess.preprocess_discriminator import preprocessing
from pipeline.train.train_traffic_classifier import train
from utils.constants import RELEVANT_COLUMNS, VALID_TRAFFIC_TYPES, VALID_PORT_RANGE, MLFLOW_HOST, MLFLOW_PORT

# Set the MLflow tracking server URI to log experiments and models
mlflow.set_tracking_uri(uri=f"http://{MLFLOW_HOST}:{MLFLOW_PORT}")

# Create or set the MLflow experiment under which runs will be logged
mlflow.set_experiment("generative_ml_malicious_discriminator")

# Define the name for the registered model in MLflow Model Registry
model_deployed_name = 'generative_ml_malicious_discriminator'
# Define the primary metric used for evaluating and comparing model performance
eval_metric = 'f1_macro'

# Start an MLflow run context to log parameters, metrics, and artifacts
with mlflow.start_run() as run:
    # Define the local directory path to store results and artifacts generated during the run
    results_folder_path = './results/discriminator'

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

    # --- Evaluation Step ---
    # Evaluate the trained model using the preprocessed test data
    # It returns a dictionary containing evaluation metrics (e.g., f1_macro)
    eval_data = evaluation(model_filepath, test_traffic_filepath, results_folder_path)

    # --- Model Deployment Logic ---
    # Query the MLflow Model Registry to get the metric of the currently deployed model version
    try:
        # Initialize the MLflow client handler
        mlflow_client_handler = MlflowClientHandler()
        # Retrieve the specified metric for the latest 'Production' or 'Staging' version of the model
        deployed_metric = mlflow_client_handler.get_model_metric(model_name=model_deployed_name, metric_name=eval_metric)
    except Exception as e:
        # Handle cases where the model might not be registered yet or other errors occur
        if 'no model versions found' not in str(e).lower(): # Re-raise unexpected errors
            raise e
        else: # If no model versions are found, assume deployed metric is 0 for comparison
            print(f"No deployed model '{model_deployed_name}' found. Proceeding with deployment.")
            deployed_metric = 0

    print(f"Deployed model '{model_deployed_name}' {eval_metric}: {deployed_metric}")
    print(f"Newly trained model {eval_metric}: {eval_data[eval_metric]}")

    # Compare the newly trained model's performance with the deployed model's performance
    if deployed_metric is not None and eval_data[eval_metric] < deployed_metric:
        # If the deployed model is better or equal, raise an exception and stop the pipeline
        # Note: Using '<' means we only deploy if the new model is strictly better. Use '<=' to deploy if equal.
        raise Exception(f"Deployed model is better than the newly trained model: "
                        f"Deployed -> {deployed_metric}, Trained -> {eval_data[eval_metric]}. Stopping deployment.")

    # --- Model Registration ---
    # If the new model is better (or no model was deployed), proceed to register it
    print(f"New model performs better. Registering model '{model_deployed_name}'...")

    # Combine preprocessing artifacts (like scalers) with the model file path
    artifacts = preprocess_artifacts | {'model': model_filepath}

    # Log the model to MLflow using the pyfunc flavor
    # This packages the model, its dependencies (conda.yaml), code, and artifacts
    mlflow.pyfunc.log_model(
        artifact_path=model_deployed_name, # Path within the MLflow run's artifacts
        python_model=MaliciousTrafficClassifierModel(), # The custom Python model instance
        artifacts=artifacts, # Dictionary of artifact names and their local paths
        conda_env="conda_malicious_traffic_classifier.yaml", # Path to the Conda environment file
        code_path=["ml_models", # List of local code directories needed by the model
                   "pipeline",
                   "utils"],
        registered_model_name=model_deployed_name # Register the model with this name in the Model Registry
    )
    print(f"Model logged and registered as '{model_deployed_name}'.")

    # --- Save Model Locally (Optional) ---
    # Define the local path where the model package will be saved
    model_artifact_path = os.path.join(results_folder_path, 'malicious_traffic_classifier_model_package')
    # Remove the destination directory if it already exists to avoid errors
    if os.path.exists(model_artifact_path):
        shutil.rmtree(model_artifact_path)
        print(f"Removed existing local model package directory: {model_artifact_path}")

    # Define specific code files required by the model (subset of code_path for log_model)
    # This might be useful if only specific files are needed for local saving/loading
    code_paths_save = ['pipeline/preprocess/preprocess_base.py', 'utils/constants.py', 'ml_models/malicious_traffic_classifier.py'] # Added model class file

    # Save the model in MLflow's pyfunc format to a local directory
    mlflow.pyfunc.save_model(
        path=model_artifact_path, # The local directory to save the model package
        python_model=MaliciousTrafficClassifierModel(), # The custom Python model instance
        artifacts=artifacts, # Dictionary of artifact names and their local paths
        code_paths=code_paths_save, # List of specific code file paths needed
        conda_env="conda_malicious_traffic_classifier.yaml" # Path to the Conda environment file
    )
    print(f"Model package also saved locally to: {model_artifact_path}")

print("MLflow run completed.")