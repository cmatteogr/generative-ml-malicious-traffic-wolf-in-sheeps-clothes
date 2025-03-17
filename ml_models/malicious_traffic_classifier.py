"""

"""
import joblib
import mlflow.pyfunc
from sklearn.ensemble import IsolationForest
import pipeline.preprocess.preprocess_base as pre_base
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, LabelEncoder
import xgboost as xgb


class MaliciousTrafficClassifierModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        # load register_models
        # Load Outlier detection model
        self.outliers_detection_model: IsolationForest = joblib.load(context.artifacts["iso_forest_model"])
        # Load Power transformer
        self.power_transformer: PowerTransformer = joblib.load(context.artifacts["power_transformer"])
        # Load One hot Encoder
        self.onehot_encoder: OneHotEncoder = joblib.load(context.artifacts["onehot_encoder"])
        # Load Label Encoder
        self.label_encoder: LabelEncoder = joblib.load(context.artifacts["label_encoder"])
        # Load the model
        self.loaded_model = xgb.XGBClassifier()
        self.loaded_model.load_model(context.artifacts["model"])

    def predict(self, context, input_data):
        traffic_df = input_data['traffic_df'].values()

        min_port = context.artifacts["min_port"]
        max_port = context.artifacts["max_port"]
        valid_traffic_types = context.artifacts["valid_traffic_types"]

        # apply filter valid values rules
        traffic_filtered_df = pre_base.filter_valid_traffic_features(traffic_df, min_port, max_port, valid_traffic_types)
        if traffic_filtered_df.shape[0] < traffic_df.shape[0]:
            print("WARNING", "some traffic instances contains invalid data!")

        # apply destination and source ports Well-Known Port Ranges for Standardized Functionalities transformations
        traffic_df['Source Port'] = traffic_df['Source Port'].map(pre_base.map_port_usage_category)
        traffic_df['Destination Port'] = traffic_df['Destination Port'].map(pre_base.map_port_usage_category)

        # apply power transformation
        power_columns = context.artifacts["power_columns"]
        traffic_df[power_columns] = self.power_transformer.transform(traffic_df[power_columns])

        # identify outliers
        # Predict anomalies (-1 for outliers and 1 for inliers)
        traffic_df['outlier'] = self.outliers_detection_model.predict(traffic_df)
        # check if there is any outlier
        if traffic_df.loc[traffic_df['outlier'] == 1].shape[0] > 0:
            print("WARNING", "traffic instance contains outliers!")

        # predict traffic
        y_pred = self.loaded_model.predict(traffic_df)

        # Apply inverse label encoder
        y_pred_label = self.label_encoder.inverse_transform(y_pred)

        # return list predictions
        return list(y_pred_label)