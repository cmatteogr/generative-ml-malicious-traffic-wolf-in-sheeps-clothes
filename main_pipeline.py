"""
traffic classifier pipeline
"""
import os
import shutil
from ml_models.malicious_traffic_classifier import MaliciousTrafficClassifierModel
from mlflow_api.mlflow_client import MlflowClientHandler
import mlflow
from pipeline.evaluation.evaluation_traffic_classifier import evaluation
from pipeline.preprocess.preprocess import preprocessing
from pipeline.train.train_traffic_classifier import train
from utils.constants import RELEVANT_COLUMNS, VALID_TRAFFIC_TYPES, VALID_PORT_RANGE

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Create a new MLFlow Experiment
mlflow.set_experiment("generative_ml_malicious")

# model deployed name
model_deployed_name = 'generative_ml_malicious_traffic'
eval_metric = 'f1_macro'

# Start an MLFlow run
with mlflow.start_run() as run:
    # results folder
    results_folder_path = './results'

    # define base traffic filepath
    base_traffic_filepath = './data/Weekly-WorkingHours_report.csv'

    # preprocess data
    train_traffic_filepath, test_traffic_filepath, preprocess_artifacts = preprocessing(base_traffic_filepath,
                                                                                        results_folder_path,
                                                                                        relevant_column=RELEVANT_COLUMNS,
                                                                                        valid_traffic_types=VALID_TRAFFIC_TYPES,
                                                                                        valid_port_range=VALID_PORT_RANGE)
    # train traffic classifier model
    model_filepath = train(train_traffic_filepath, results_folder_path)

    # evaluate model
    eval_data = evaluation(model_filepath, test_traffic_filepath, results_folder_path)

    # query the model deployed
    try:
        mlflow_client_handler = MlflowClientHandler()
        # find the metrics
        deployed_metric = mlflow_client_handler.get_model_metric(model_name=model_deployed_name, metric_name=eval_metric)
    except Exception as e:
        # model not found or
        if 'no model versions found' not in str(e):
            raise e
        else: deployed_metric = 0

    print(f" deployed metric: {deployed_metric}")

    # if deployed metric is none directly deploy
    if deployed_metric is not None:
        if eval_data['f1_macro'] < deployed_metric:
            # close current execution
            raise Exception(f"deployed model is better than trained model: deployed -> {deployed_metric}, trained -> {eval_data['f1_macro']}")

    # register model
    artifacts = preprocess_artifacts | {'model': model_filepath}
    # deploy model
    mlflow.pyfunc.log_model(
        artifact_path=model_deployed_name,
        python_model=MaliciousTrafficClassifierModel(),
        artifacts=artifacts,
        conda_env="conda_malicious_traffic_classifier.yaml",
        code_path=["ml_models",
                   "pipeline",
                   "utils"]
    )


    model_artifact_path = os.path.join('results', 'malicious_traffic_classifier_model')
    # remove folder if exist
    if os.path.exists(model_artifact_path):
        shutil.rmtree(model_artifact_path)
    code_paths = ['pipeline/preprocess/preprocess_base.py', 'utils/constants.py']
    # Save the custom model
    mlflow.pyfunc.save_model(
        path=model_artifact_path,
        python_model=MaliciousTrafficClassifierModel(),
        artifacts=artifacts,
        code_paths=code_paths,
        conda_env="conda_malicious_traffic_classifier.yaml"
    )

