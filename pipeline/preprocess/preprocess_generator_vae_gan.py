"""
Author: Cesar M. Gonzalez

Preprocess script for network traffic data. Includes cleaning, feature engineering,
outlier removal, transformation, and sampling.
"""
import json
import os.path
from operator import index
from typing import List, Tuple, Dict
from sklearn.ensemble import IsolationForest
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from pipeline.preprocess.preprocess_base import map_port_usage_category, filter_valid_traffic_features, \
    undersampling_dataset
import mlflow
import joblib
from sklearn.preprocessing import LabelEncoder
from utils.constants import POWER_TRANSFORMER_NAME, ONEHOT_ENCODER_NAME, ISO_FOREST_MODEL_NAME, \
    PREPROCESS_PARAMS_NAME, SCALER_NAME, LABEL_ENCODER_NAME
from utils.utils import generate_profiling_report


def preprocessing(traffic_filepath: str,
                  results_folder_path: str,
                  relevant_column: List[str],
                  valid_traffic_types: List[str],
                  valid_port_range: Tuple[int, int],
                  n_instances_per_traffic_type: int = 150000,
                  test_size: float = 0.2) -> Tuple[str, str, Dict[str, str]]:
    """
    Performs preprocessing on raw network traffic data for the Generator model. B-TCVAE-GAN

    Preprocess include the following steps.
    - Reading and initial cleaning (column selection, renaming).
    - Filtering based on valid data ranges and traffic types.
    - Splitting data into training and testing sets.
    - Feature engineering: Apply transformations to features.
    - Outlier detection and removal using Z-score, KMeans (commented out test part), and Isolation Forest.
    - Feature transformation: Power transformation (Yeo-Johnson) for numerical features.
    - Feature encoding: One-Hot Encoding for categorical port features.
    - Undersampling the training data to balance classes.
    - Apply normalization using MinMaxScaler for generative model.
    - Saving preprocessed data and transformation artifacts (scalers, encoders, models).
    - Logging parameters and artifacts to MLflow.

    :param traffic_filepath: Path to the raw input traffic data CSV file.
    :type traffic_filepath: str
    :param results_folder_path: Path to the directory where results (preprocessed data, artifacts) will be saved.
    :type results_folder_path: str
    :param relevant_column: List of column names to keep from the raw data.
    :type relevant_column: List[str]
    :param valid_traffic_types: List of traffic labels considered valid (e.g., ['BENIGN', 'MALICIOUS_X']).
    :type valid_traffic_types: List[str]
    :param valid_port_range: A tuple containing the minimum and maximum valid port numbers (inclusive).
    :type valid_port_range: Tuple[int, int]
    :param n_instances_per_traffic_type: Target number of instances per traffic type after undersampling (for training data). Defaults to 150000.
    :type n_instances_per_traffic_type: int
    :param test_size: Proportion of the dataset to include in the test split. Defaults to 0.2.
    :type test_size: float
    :return:
             - Path to the preprocessed training data CSV file.
             - Path to the preprocessed test data CSV file.
             - Dictionary of paths to saved preprocessing artifacts (transformer, encoders, models, params).
    :rtype: Tuple[str, str, Dict[str, str]]
    """
    # --- Read dataset ---
    print("read traffic data")
    traffic_df = pd.read_csv(traffic_filepath, low_memory=False)

    print("clean data")
    # keep only relevant columns
    traffic_df = traffic_df[relevant_column]
    # standardize column names by stripping leading/trailing whitespace
    traffic_df.rename(columns=lambda x: x.strip(), inplace=True)

    # --- Filter Valid Data ---
    print('filtering by valid traffic conditions')
    min_port, max_port = valid_port_range
    # filter by rules defined in preprocess_base
    traffic_df = filter_valid_traffic_features(traffic_df, min_port, max_port, valid_traffic_types)
    print(f'shape after initial filtering: {traffic_df.shape}')

    # generate dataset profiling report, raw dataset
    raw_traffic_df = undersampling_dataset(valid_traffic_types, traffic_df.copy(), n_instances_per_traffic_type)
    raw_traffic_filepath = os.path.join(results_folder_path, 'raw_traffic.csv')
    raw_traffic_df.to_csv(raw_traffic_filepath, index=False)
    title = "Raw dataset Profiling"
    report_name = 'preprocessing_traffic_raw_dataset_profiling_generator'
    report_filepath = os.path.join(results_folder_path, f"{report_name}.html")
    type_schema = {'Label': "categorical"}
    generate_profiling_report(report_filepath=report_filepath, title=title, data_filepath=raw_traffic_filepath,
                              type_schema=type_schema, minimal=True)

    # --- Train/Test Split ---
    print(f'splitting data with test_size={test_size}')
    X_train, X_test = train_test_split(traffic_df, test_size=test_size, random_state=42)
    print(f'train shape: {X_train.shape}, test shape: {X_test.shape}')

    # --- Feature Engineering: Port Categories ---
    print('transform features')
    X_train.loc[:, 'Source Port'] = X_train['Source Port'].map(map_port_usage_category)
    X_train.loc[:, 'Destination Port'] = X_train['Destination Port'].map(map_port_usage_category)
    X_test.loc[:, 'Source Port'] = X_test['Source Port'].map(map_port_usage_category)
    X_test.loc[:, 'Destination Port'] = X_test['Destination Port'].map(map_port_usage_category)

    # --- Feature Engineering: Remove Outliers per Feature ---
    max_quantile_total_fwd_packets = X_train['Total Fwd Packets'].quantile(0.95)
    max_quantile_total_bwd_packets = X_train['Total Backward Packets'].quantile(0.95)
    max_quantile_total_length_fwd_packets = X_train['Total Length of Fwd Packets'].quantile(0.95)
    max_quantile_total_length_bwd_packets = X_train['Total Length of Bwd Packets'].quantile(0.95)
    max_quantile_fwd_header_length = X_train['Fwd Header Length'].quantile(0.95)
    max_quantile_bwd_header_length = X_train['Bwd Header Length'].quantile(0.95)
    max_quantile_subflow_fwd_packets = X_train['Subflow Fwd Packets'].quantile(0.95)
    max_quantile_subflow_bwd_packets = X_train['Subflow Bwd Packets'].quantile(0.95)
    max_quantile_act_data_pkt_fwd = X_train['act_data_pkt_fwd'].quantile(0.95)

    # the quantile 95 for 'Total Fwd Packets' is 26 and max value ~ 218658, a long tail, power transformation maybe unnecessary
    # the quantile 95 for 'Total Backward Packets' is 26 and max value ~ 218658, a long tail, power transformation maybe unnecessary
    #X_train = X_train.loc[X_train['Total Fwd Packets'] <= max_quantile_total_fwd_packets]
    #X_train = X_train.loc[X_train['Total Backward Packets'] <= max_quantile_total_bwd_packets]
    #X_train = X_train.loc[X_train['Total Length of Fwd Packets'] <= max_quantile_total_length_fwd_packets]
    #X_train = X_train.loc[X_train['Total Length of Bwd Packets'] <= max_quantile_total_length_bwd_packets]
    X_train = X_train.loc[X_train['Fwd Header Length'] <= max_quantile_fwd_header_length]
    X_train = X_train.loc[X_train['Bwd Header Length'] <= max_quantile_bwd_header_length]
    X_train = X_train.loc[X_train['Subflow Fwd Packets'] <= max_quantile_subflow_fwd_packets]
    X_train = X_train.loc[X_train['Subflow Bwd Packets'] <= max_quantile_subflow_bwd_packets]
    X_train = X_train.loc[X_train['act_data_pkt_fwd'] <= max_quantile_act_data_pkt_fwd]
    # ----
    #X_test = X_test.loc[X_test['Total Fwd Packets'] <= max_quantile_total_fwd_packets]
    #X_test = X_test.loc[X_test['Total Backward Packets'] <= max_quantile_total_bwd_packets]
    #X_test = X_test.loc[X_test['Total Length of Fwd Packets'] <= max_quantile_total_length_fwd_packets]
    #X_test = X_test.loc[X_test['Total Length of Bwd Packets'] <= max_quantile_total_length_bwd_packets]
    X_test = X_test.loc[X_test['Fwd Header Length'] <= max_quantile_fwd_header_length]
    X_test = X_test.loc[X_test['Bwd Header Length'] <= max_quantile_bwd_header_length]
    X_test = X_test.loc[X_test['Subflow Fwd Packets'] <= max_quantile_subflow_fwd_packets]
    X_test = X_test.loc[X_test['Subflow Bwd Packets'] <= max_quantile_subflow_bwd_packets]
    X_test = X_test.loc[X_test['act_data_pkt_fwd'] <= max_quantile_act_data_pkt_fwd]


    print('apply Power Transformation (Yeo-Johnson), for features with long tail')
    # Define columns for power transformation
    power_columns = [
        'Total Length of Fwd Packets',
        'Total Length of Bwd Packets',
        'Flow Duration',
        'Fwd Packet Length Mean', 'Fwd Packet Length Std',
        #'Bwd Packet Length Mean',
        #'Bwd Packet Length Std',
        'Flow IAT Mean', 'Flow IAT Std',
        #'Fwd IAT Total',
        'Fwd IAT Mean',
        'Fwd IAT Std', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',

        'Total Fwd Packets',
        'Total Backward Packets',
        #'Fwd Header Length',
        #'Bwd Header Length'
        #'Subflow Fwd Packets',
        #'Subflow Fwd Bytes',
        #'Subflow Bwd Packets',
        #'Subflow Bwd Bytes',
        #'Init_Win_bytes_backward',
        #'act_data_pkt_fwd',
        'Active Mean',
        'Active Std',
        'Idle Std'
    ]
    # Ensure all power columns exist in the dataframe after outlier removal
    power_columns = [col for col in power_columns if col in X_train.columns]
    pt_model = PowerTransformer(method='yeo-johnson')
    # Fit on training data only
    pt_model.fit(X_train[power_columns])
    # Transform train and test data
    X_train.loc[:, power_columns] = pt_model.transform(X_train[power_columns])
    X_test.loc[:, power_columns] = pt_model.transform(X_test[power_columns])
    # Save the fitted PowerTransformer
    power_transformer_filepath = os.path.join(results_folder_path, POWER_TRANSFORMER_NAME)
    joblib.dump(pt_model, power_transformer_filepath)
    print(f"PowerTransformer saved to {power_transformer_filepath}")

    # --- Feature Encoding: One-Hot Encoding ---
    print('applying One-Hot Encoding for port categories')
    one_hot_encoding_columns = ['Source Port', 'Destination Port']
    # Initialize OneHotEncoder
    onehot_encoder = OneHotEncoder(sparse_output=False)
    # Fit on training data
    onehot_encoder.fit(X_train[one_hot_encoding_columns])
    # transform train data
    encoded_data_train = onehot_encoder.transform(X_train[one_hot_encoding_columns])
    encoded_data_train_df = pd.DataFrame(encoded_data_train,
                                         columns=onehot_encoder.get_feature_names_out(one_hot_encoding_columns),
                                         index=X_train.index)
    # Concatenate and drop original columns
    X_train = pd.concat([X_train.drop(columns=one_hot_encoding_columns), encoded_data_train_df], axis=1)
    # Transform test data
    encoded_data_test = onehot_encoder.transform(X_test[one_hot_encoding_columns])
    encoded_data_test_df = pd.DataFrame(encoded_data_test,
                                        columns=onehot_encoder.get_feature_names_out(one_hot_encoding_columns),
                                        index=X_test.index)
    # Concatenate and drop original columns
    X_test = pd.concat([X_test.drop(columns=one_hot_encoding_columns), encoded_data_test_df], axis=1)

    # Save the fitted OneHotEncoder
    onehot_encoder_filepath = os.path.join(results_folder_path, ONEHOT_ENCODER_NAME)
    joblib.dump(onehot_encoder, onehot_encoder_filepath)
    print(f"OneHotEncoder saved to {onehot_encoder_filepath}")

    # save labels for later use.
    # NOTE: remove the label to apply outlier removal
    label_train = X_train.pop('Label')
    label_test = X_test.pop('Label')

    # --- Outlier Detection/Removal ---
    # Applying Isolation Forest on the transformed/encoded data
    print('applying Isolation Forest for global outlier removal')
    iso_forest = IsolationForest(n_estimators=200, contamination='auto', random_state=42)
    iso_forest.fit(X_train)
    # Predict anomalies (-1 for outliers, 1 for inliers)
    train_outliers = iso_forest.predict(X_train)
    test_outliers = iso_forest.predict(X_test)
    # Create masks for inliers
    iso_inliers_mask_train = train_outliers != -1
    iso_inliers_mask_test = test_outliers != -1
    # Remove outliers from train and test sets (X and y)
    X_train = X_train.loc[iso_inliers_mask_train]
    X_test = X_test.loc[iso_inliers_mask_test]
    print(f'shape after Isolation Forest: Train={X_train.shape}, Test={X_test.shape}')
    # Save IsolationForest model
    iso_forest_model_filepath = os.path.join(results_folder_path, ISO_FOREST_MODEL_NAME)
    joblib.dump(iso_forest, iso_forest_model_filepath)
    print(f"IsolationForest model saved to {iso_forest_model_filepath}")

    # --- Normalize data (Applied only to Training Data) ---
    # init Min Max scaler to normalize dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train)
    X_train_scaled_array = scaler.transform(X_train)
    X_test_scaled_array = scaler.transform(X_test)
    X_train = pd.DataFrame(X_train_scaled_array, index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(X_test_scaled_array, index=X_test.index, columns=X_test.columns)
    scaler_model_filepath = os.path.join(results_folder_path, SCALER_NAME)
    joblib.dump(scaler, scaler_model_filepath)

    # add labels to apply undersampling
    X_train['Label'] = label_train[X_train.index]
    X_test['Label'] = label_test[X_test.index]

    # --- Undersampling (Applied only to Training Data) ---
    print('applying undersampling to balance training data')
    X_train_temp = X_train.copy()  # Work on a copy

    X_train = undersampling_dataset(valid_traffic_types, X_train_temp, n_instances_per_traffic_type)

    # add labels
    X_train['Label'] = label_train[X_train.index]
    X_test['Label'] = label_test[X_test.index]

    # --- Apply Label encoding ---
    print('applying Label Encoding to target variable')
    le = LabelEncoder()
    le.fit(X_train['Label'])
    # transform train and test labels
    X_train['Label'] = le.transform(X_train['Label'])
    X_test['Label'] = le.transform(X_test['Label'])
    label_model_filepath = os.path.join(results_folder_path, LABEL_ENCODER_NAME)
    joblib.dump(le, label_model_filepath)

    mlflow.log_param("final_preprocess_columns", str(list(X_train.columns)))

    # --- Save Preprocessed Data ---
    print('saving preprocessed data')
    # define file paths
    train_normalized_traffic_filepath = os.path.join(results_folder_path, 'traffic_preprocessed_train_normalized.csv')
    test_normalized_traffic_filepath = os.path.join(results_folder_path, 'traffic_preprocessed_test_normalized.csv')
    # Save to CSV
    X_train.to_csv(train_normalized_traffic_filepath, index=False)
    X_test.to_csv(test_normalized_traffic_filepath, index=False)

    # generate dataset profiling report
    title = "Preprocessing Train dataset Profiling"
    report_name = 'preprocessing_traffic_train_dataset_profiling_generator'
    report_filepath = os.path.join(results_folder_path, f"{report_name}.html")
    type_schema = {'Label': "categorical"}
    generate_profiling_report(report_filepath=report_filepath, title=title,
                              data_filepath=train_normalized_traffic_filepath,
                              type_schema=type_schema, minimal=True)

    # --- 12. Log Parameters and Artifacts ---
    print('Logging parameters and preparing artifacts dictionary...')
    # Consolidate parameters for logging
    params = {
        "relevant_columns": relevant_column,
        "valid_traffic_types": valid_traffic_types,
        "min_port": min_port,
        "max_port": max_port,
        "power_columns": power_columns,
        "one_hot_encoding_columns": one_hot_encoding_columns,
        # Ensure variable exists
        "n_instances_per_traffic_type_target": n_instances_per_traffic_type,
        "test_size": test_size,
        # Add other relevant parameters like KMeans k, threshold, Isolation Forest params if desired
    }
    # Log parameters to MLflow
    mlflow.log_params({k: str(v) if isinstance(v, (list, dict)) else v for k, v in params.items()})

    # Save parameters locally as JSON
    preprocess_params_filepath = os.path.join(results_folder_path, PREPROCESS_PARAMS_NAME)
    with open(preprocess_params_filepath, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"Preprocessing parameters saved locally to {preprocess_params_filepath}")

    # collect artifact paths, handling cases where artifacts might not have been created
    preprocess_artifacts = {
        "power_transformer": power_transformer_filepath,
        "onehot_encoder": onehot_encoder_filepath,
        "iso_forest_model": iso_forest_model_filepath,
        "scaler": scaler_model_filepath,
        "preprocess_params": preprocess_params_filepath,
        "label_encoder": label_model_filepath,
    }

    # return paths to preprocessed data and the dictionary of artifact paths
    return train_normalized_traffic_filepath, test_normalized_traffic_filepath, preprocess_artifacts