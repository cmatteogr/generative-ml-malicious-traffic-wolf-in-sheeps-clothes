"""

"""
import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient
from utils.constants import MLFLOW_HOST, MLFLOW_PORT


class MlflowClientHandler:

    def __init__(self, tracking_uri: str=f"http://{MLFLOW_HOST}:{MLFLOW_PORT}"):
        mlflow.set_tracking_uri(tracking_uri)
        self.tracking_uri = tracking_uri
        self.mlflow_client = MlflowClient()

    def get_deployed_model(self, model_name: str, version='latest') -> ModelVersion:

        # if version is a number use the get model version service
        if isinstance(version, int):
            return self.mlflow_client.get_model_version(name=model_name, version=str(version))

        # get the version by index
        match version:
            case 'latest':
                index_version = -1
            case 'first':
                index_version = 0
            case _:
                raise Exception(f"invalid index version: {version}")

        # find model by name and get the version by index
        model_versions = self.mlflow_client.search_model_versions(f"name='{model_name}'")
        # check if model was found
        if len(model_versions) == 0:
            raise Exception(f"no model versions found for '{model_name}'")
        # get model index
        model_index = sorted(model_versions, key=lambda mv: int(mv.version))[index_version]
        # return model
        return model_index

    def get_model_metric(self, model_name: str, metric_name: str):
        # get last deployed model
        model_deployed: ModelVersion = self.get_deployed_model(model_name)
        # get model deployed run id
        model_run = self.mlflow_client.get_run(model_deployed.run_id)
        # find metric
        metrics = model_run.data.metrics
        # check if metric is found
        if metric_name not in metrics.keys():
            raise ValueError(f"Metric {metric_name} not found")
        # return metric
        return metrics[metric_name]