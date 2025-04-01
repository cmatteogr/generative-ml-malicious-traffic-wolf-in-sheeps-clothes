"""
Author: Cesar M. Gonzalez

Preprocess
"""
import json
import os.path
from typing import List, Tuple
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PowerTransformer, power_transform
from pipeline.preprocess.preprocess_base import map_port_usage_category, filter_valid_traffic_features
from sklearn.preprocessing import LabelEncoder
import mlflow
import joblib

from utils.constants import POWER_TRANSFORMER_NAME, ONEHOT_ENCODER_NAME, ISO_FOREST_MODEL_NAME, LABEL_ENCODER_NAME, \
    PREPROCESS_PARAMS_NAME


# NOTE: Watch this video to understand about Kurtosis and Skewness:
# https://www.youtube.com/watch?v=EWuR4EGc9EY
# To learn about quantiles and percentiles:
# https://www.youtube.com/watch?v=IFKQLDmRK0Y
# To learn about standard deviation, mean, variance:
# https://www.youtube.com/watch?v=SzZ6GpcfoQY

def preprocessing(traffic_filepath: str, results_folder_path: str, relevant_column: List[str],
                  valid_traffic_types: List[str], valid_port_range: Tuple, n_instances_per_traffic_type:int =150000,
                  test_size: float = 0.2) -> Tuple[str, str, dict]:
    """
    Preprocess traffic data
    :param traffic_filepath: traffic data file path
    :param results_folder_path: results folder path
    :param relevant_column: relevant column name list
    :param valid_traffic_types: valid traffic types list
    :param valid_port_range: valid port range
    :param n_instances_per_traffic_type: number of instances per traffic type
    :param test_size: test size
    :return: preprocessed traffic data file path
    """
    # read dataset
    print("read traffic data")
    traffic_df = pd.read_csv(traffic_filepath, low_memory=False)
    # clean dataset
    print("data cleaning")
    # filter relevant columns
    traffic_df = traffic_df[relevant_column]
    # format the columns names
    traffic_df.rename(columns=lambda x: x.strip(), inplace=True)

    print('filter by valid data ranges')
    # filter instances with invalid values
    min_port, max_port = valid_port_range

    # apply filter valid values rules
    traffic_df = filter_valid_traffic_features(traffic_df, min_port, max_port, valid_traffic_types)
    print(f'traffic df shape: {traffic_df.shape}')

    # pop label
    label = traffic_df.pop('Label')
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(traffic_df, label, test_size=test_size, random_state=42)

    print('transform port types')
    # apply destination and source ports Well-Known Port Ranges for Standardized Functionalities transformations
    X_train['Source Port'] = X_train['Source Port'].map(map_port_usage_category)
    X_train['Destination Port'] = X_train['Destination Port'].map(map_port_usage_category)
    X_test['Source Port'] = X_test['Source Port'].map(map_port_usage_category)
    X_test['Destination Port'] = X_test['Destination Port'].map(map_port_usage_category)

    # Apply anomaly detection model to clean the distribution features. It includes features with mean and standard deviation
    # NOTE: Several methods could be applied like, z-score, quantiles, DBSCAN, isolation forest, KMeans, etc.
    # Z-score: https://vitalflux.com/outlier-detection-techniques-in-python/
    # Quantiles: https://vitalflux.com/outlier-detection-techniques-in-python/
    # KMeans: https://blog.zhaytam.com/2019/08/06/outliers-detection-in-pyspark-3-k-means/
    # DBSCAN: https://medium.com/@dilip.voleti/dbscan-algorithm-for-fraud-detection-outlier-detection-in-a-data-set-60a10ad06ea8
    # Isolation Forest: https://hands-on.cloud/using-python-and-isolation-forest-algorithm-for-anomalies-detection/
    # Mahalanobis Distance: https://blog.dailydoseofds.com/p/the-limitation-of-euclidean-distance

    print('remove outliers')
    # anomaly detection using z-score
    # Fwd Packet Length Mean
    fwd_packet_len_mean = X_train['Fwd Packet Length Mean']
    threshold_fwd_packet_length_mean = 3
    mean_fwd_packet_len_mean = np.mean(fwd_packet_len_mean)
    std_fwd_packet_len_mean = np.std(fwd_packet_len_mean)
    # Compute Z-scores
    z_scores_fwd_packet_length_mean = (fwd_packet_len_mean - mean_fwd_packet_len_mean) / std_fwd_packet_len_mean
    # Set threshold for anomaly detection
    z_scores_outliers_tags = np.abs(z_scores_fwd_packet_length_mean) <= threshold_fwd_packet_length_mean
    X_train = X_train[z_scores_outliers_tags]
    # zcore test
    fwd_packet_len_mean = X_test['Fwd Packet Length Mean']
    z_scores_fwd_packet_length_mean = (fwd_packet_len_mean - mean_fwd_packet_len_mean) / std_fwd_packet_len_mean
    # Set threshold for anomaly detection
    X_test = X_test[np.abs(z_scores_fwd_packet_length_mean) <= threshold_fwd_packet_length_mean]

    # Fwd Packet Length Std
    fwd_packet_len_std = X_train['Fwd Packet Length Std']
    threshold_fwd_packet_length_std = 3
    mean_fwd_packet_len_std = np.mean(fwd_packet_len_std)
    std_fwd_packet_len_std = np.std(fwd_packet_len_std)
    # compute Z-scores
    z_scores_fwd_packet_length_std = (fwd_packet_len_std - mean_fwd_packet_len_std) / std_fwd_packet_len_std
    # Set threshold for anomaly detection
    z_scores_outliers_tags = np.abs(z_scores_fwd_packet_length_std) <= threshold_fwd_packet_length_std
    X_train = X_train[z_scores_outliers_tags]
    # test
    fwd_packet_len_std = X_test['Fwd Packet Length Std']
    # compute Z-scores
    z_scores_fwd_packet_length_std = (fwd_packet_len_std - mean_fwd_packet_len_std) / std_fwd_packet_len_std
    # Set threshold for anomaly detection
    X_test = X_test[np.abs(z_scores_fwd_packet_length_std) <= threshold_fwd_packet_length_std]

    # anomaly detection using KMeans
    # apply normalization
    bwd_packet_len_columns = ['Bwd Packet Length Mean', 'Bwd Packet Length Std']
    bwd_packet_length_distribution = X_train[bwd_packet_len_columns]
    scaler = MinMaxScaler(feature_range=(0, 1))  # Scaling between [0,1]
    scaler.fit(bwd_packet_length_distribution)
    train_bwd_packet_len_dis_scaled = scaler.transform(bwd_packet_length_distribution)
    test_bwd_packet_len_dis_scaled = scaler.transform(X_test[bwd_packet_len_columns])
    # Apply K-Means clustering
    k = 3  # Number of clusters. What is the best number. Elbow?
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(train_bwd_packet_len_dis_scaled)
    # compute distances to the nearest cluster centroid
    distances = np.linalg.norm(train_bwd_packet_len_dis_scaled - kmeans.cluster_centers_[kmeans.labels_], axis=1)
    # define threshold. based on the quantiles
    threshold = np.percentile(distances, 95)
    # filter by the centroid distance
    X_train = X_train[distances <= threshold]
    # Compute distances to the nearest cluster centroid
    #distances = np.linalg.norm(test_bwd_packet_len_dis_scaled - kmeans.cluster_centers_[kmeans.labels_], axis=1)
    # filter by the centroid distance
    #X_test = X_test[distances <= threshold]

    print('apply power transformation')
    # apply power transformation
    pt_model = PowerTransformer(method='yeo-johnson')
    power_columns = ['Total Length of Fwd Packets', 'Total Length of Bwd Packets',
                     'Flow Duration', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Mean',
                     'Bwd Packet Length Std', 'Flow IAT Mean', 'Flow IAT Std', 'Fwd IAT Total', 'Fwd IAT Mean',
                     'Fwd IAT Std', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std']
    pt_model.fit(X_train[power_columns])
    X_train[power_columns] = pt_model.transform(X_train[power_columns])
    X_test[power_columns] = pt_model.transform(X_test[power_columns])
    # save power transformer
    power_transformer_filepath = os.path.join(results_folder_path, POWER_TRANSFORMER_NAME)
    joblib.dump(pt_model, power_transformer_filepath)

    # Note: Below the under sampling you can find the
    print('apply categorical transformation')
    # one Hot Encoding
    onehot_encoder = OneHotEncoder(sparse_output=False)  # Set sparse=True for a sparse matrix
    # fit and transform data
    # define one hot encoding columns
    one_hot_encoding_columns = ['Source Port', 'Destination Port']
    # train encoding
    encoded = onehot_encoder.fit(X_train[one_hot_encoding_columns])
    encoded_data = encoded.transform(X_train[one_hot_encoding_columns])
    # encoded data
    encoded_data_df = pd.DataFrame(encoded_data, columns=encoded.get_feature_names_out(one_hot_encoding_columns),
                                   index=X_train.index)
    # Concatenate the original DataFrame with the drivetrain encoded DataFrame
    X_train = pd.concat([X_train, encoded_data_df], axis=1)
    # remove categorical columns
    X_train = X_train.loc[:, ~X_train.columns.isin(one_hot_encoding_columns)]
    # test encoding
    encoded_data = encoded.transform(X_test[one_hot_encoding_columns])
    # encoded data
    encoded_data_df = pd.DataFrame(encoded_data, columns=encoded.get_feature_names_out(one_hot_encoding_columns),
                                   index=X_test.index)
    # Concatenate the original DataFrame with the drivetrain encoded DataFrame
    X_test = pd.concat([X_test, encoded_data_df], axis=1)
    # remove categorical columns
    X_test = X_test.loc[:, ~X_test.columns.isin(one_hot_encoding_columns)]
    # save one hot encoder transformer
    onehot_encoder_filepath = os.path.join(results_folder_path, ONEHOT_ENCODER_NAME)
    joblib.dump(onehot_encoder, onehot_encoder_filepath)

    print('remove outliers')
    # remove outliers
    iso_forest = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
    # Fit the model
    iso_forest.fit(X_train)
    # Predict anomalies (-1 for outliers and 1 for inliers)
    X_train['outlier'] = iso_forest.predict(X_train)
    X_test['outlier'] = iso_forest.predict(X_test)
    # Remove global outliers
    X_train = X_train[X_train['outlier'] != -1]
    y_train = y_train.loc[X_train.index]
    X_train.drop(columns='outlier', inplace=True)
    X_test = X_test[X_test['outlier'] != -1]
    y_test = y_test.loc[X_test.index]
    X_test.drop(columns='outlier', inplace=True)

    # merge features and labels in single dataframe
    y_train = y_train.loc[X_train.index]
    y_test = y_test.loc[X_test.index]
    X_train['Label'] = y_train
    X_test['Label'] = y_test
    # save iso forest model
    iso_forest_model_filepath = os.path.join(results_folder_path, ISO_FOREST_MODEL_NAME)
    joblib.dump(iso_forest, iso_forest_model_filepath)

    print('apply under sampling')
    # Apply Down sampling to solve the unbalanced
    x_train_traffic_df_list = []
    for valid_traffic_type in valid_traffic_types:
        # filter by type and sample
        x_train_traffic_type_df = X_train.loc[X_train['Label'] == valid_traffic_type]
        # check if there are enough instances to apply sampling
        if x_train_traffic_type_df.shape[0] > n_instances_per_traffic_type:
            x_train_traffic_type_df = x_train_traffic_type_df.sample(n_instances_per_traffic_type)
        # concatenate result
        x_train_traffic_df_list.append(x_train_traffic_type_df)
    # concatenate traffic df
    train_traffic_df = pd.concat(x_train_traffic_df_list, axis=0)
    test_traffic_df = X_test.copy()

    # transform labels using label encoder
    le = LabelEncoder()
    le.fit(train_traffic_df['Label'])
    train_traffic_df['Label'] = pd.Series(le.transform(train_traffic_df['Label']), index=train_traffic_df.index)
    test_traffic_df['Label'] = pd.Series(le.transform(test_traffic_df['Label']), index=test_traffic_df.index)
    # get encoding mapping
    label_encoding_mapping = {label: idx for idx, label in enumerate(le.classes_)}
    # save Label encoder
    label_encoder_filepath = os.path.join(results_folder_path, LABEL_ENCODER_NAME)
    joblib.dump(le, label_encoder_filepath)

    print('save preprocessed data')
    # save preprocessed data
    train_traffic_filepath = os.path.join(results_folder_path, 'traffic_preprocessed_train.csv')
    train_traffic_df.to_csv(train_traffic_filepath, index=False)
    test_traffic_filepath = os.path.join(results_folder_path, 'traffic_preprocessed_test.csv')
    test_traffic_df.to_csv(test_traffic_filepath, index=False)

    # Log the hyperparameters
    params = {
        "relvant_columns": relevant_column,
        "valid_traffic": valid_traffic_types,
        "label_encoding_mapping": label_encoding_mapping,
        "min_port": min_port,
        "max_port": max_port,
        "power_columns": power_columns
    }
    mlflow.log_params(params)
    preprocess_params_filepath = os.path.join(results_folder_path, PREPROCESS_PARAMS_NAME)
    with open(preprocess_params_filepath, 'w') as f:
        json.dump(params, f, indent=4)

    # get preprocess artifacts
    preprocess_artifacts = {
        "power_transformer": power_transformer_filepath,
        "onehot_encoder": onehot_encoder_filepath,
        "iso_forest_model": iso_forest_model_filepath,
        "label_encoder": label_encoder_filepath,
        "preprocess_params": preprocess_params_filepath
    }

    # return preprocessed data filepath
    return train_traffic_filepath, test_traffic_filepath, preprocess_artifacts