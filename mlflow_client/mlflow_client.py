"""

"""
import mlflow
from mlflow.tracking import MlflowClient
from utils.constants import MLFLOW_HOST, MLFLOW_PORT


class MlflowClientHandler:

    def __init__(self, tracking_uri: str=f"http://{MLFLOW_HOST}:{MLFLOW_PORT}"):
        mlflow.set_tracking_uri(tracking_uri)
        self.tracking_uri = tracking_uri
        self.mlflow_client = MlflowClient()

    def get_deployed_model(self, model_name: str):
        # find model
        latest_versions = self.mlflow_client.get_latest_versions(model_name,
                                                                 stages=["None", "Staging", "Production", "Archived"])
        # check list of models
        if len(latest_versions) == 0:
            raise FileNotFoundError(f"No versions found for model: {model_name}")
        # return model
        return latest_versions[0]

    def get_model_metric(self, model_name: str, metric_name: str):
        # get last deployed model
        model_deployed = self.get_deployed_model(model_name)
        # get model deployed run id
        model_run = self.mlflow_client.get_run(model_deployed.run_id)
        # find metric
        metrics = model_run.data.metrics
        # check if metric is found
        if metric_name not in metrics.keys():
            raise ValueError(f"Metric {metric_name} not found")
        # return metric
        return metrics[metric_name]