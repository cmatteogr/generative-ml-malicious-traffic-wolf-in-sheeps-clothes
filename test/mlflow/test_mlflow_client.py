"""

"""
from mlflow_api.mlflow_client import MlflowClientHandler


def test_mlflow_get_model_metric():
    mlflow_client_handler = MlflowClientHandler()
    # find the metrics
    metric = mlflow_client_handler.get_model_metric(model_name='regression_price_medellin', metric_name='rmse')
    print(f"metric : {metric}")

def test_get_deployed_model():
    mlflow_client_handler = MlflowClientHandler()
    # find the metrics
    model = mlflow_client_handler.get_deployed_model(model_name='regression_price_medellin', version='latest')
    pass

    # query the model deployed
    model_deployed_name = 'gen_ml_malicious_traffic'
    eval_metric = 'f1_macro'
    try:
        mlflow_client_handler = MlflowClientHandler()
        # find the metrics
        deployed_metric = mlflow_client_handler.get_model_metric(model_name=model_deployed_name,
                                                                 metric_name=eval_metric)
    except Exception as e:
        # model not found or
        if 'no model versions found' not in str(e):
            raise e
        else:
            deployed_metric = 0

    print(f" deployed metric: {deployed_metric}")
