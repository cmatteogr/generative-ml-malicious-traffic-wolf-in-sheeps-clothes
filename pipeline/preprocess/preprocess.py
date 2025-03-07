"""
Author: Cesar M. Gonzalez

Preprocess
"""
from typing import List, Tuple

from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PowerTransformer
from pipeline.preprocess.preprocess_base import map_port_usage_category
from sklearn.preprocessing import LabelEncoder
import mlflow
from utils.constants import RELEVANT_COLUMNS, VALID_TRAFFIC_TYPES, VALID_PORT_RANGE, VALID_PROTOCOL_VALUES


# NOTE: Watch this video to understand about Kurtosis and Skewness:
# https://www.youtube.com/watch?v=EWuR4EGc9EY
# To learn about quantiles and percentiles:
# https://www.youtube.com/watch?v=IFKQLDmRK0Y
# To learn about standard deviation, mean, variance:
# https://www.youtube.com/watch?v=SzZ6GpcfoQY

def preprocessing(traffic_filepath: str, relevant_column: List[str], valid_traffic_types: List[str],
                  valid_port_range: Tuple, valid_protocol_values: List[str], n_instances_per_traffic_type:int =150000,
                  test_size: float = 0.2) -> str:
    # read dataset
    traffic_df = pd.read_csv(traffic_filepath, low_memory=False)
    # clean dataset
    print("data cleaning")
    # filter relevant columns
    traffic_df = traffic_df[relevant_column]
    # format the columns names
    traffic_df.rename(columns=lambda x: x.strip(), inplace=True)

    # generate data profiling report
    #base_traffic_filepath = 'traffic_preprocessed_base.csv'
    #traffic_df.to_csv(base_traffic_filepath, index=False)

    print('filter by valid data ranges')
    # filter instances with invalid values
    min_port, max_port = valid_port_range

    valid_mask = (
        # filter by valid port, source and destination
        traffic_df['Source Port'].between(min_port, max_port, inclusive='both') &
        traffic_df['Destination Port'].between(min_port, max_port, inclusive='both') &

        # filter by valid protocols
        #traffic_df['Protocol'].isin(valid_protocol_values) &

        # Flow Duration >= 0
        (traffic_df['Flow Duration'] >= 0) &

        # Packet counts >= 0
        (traffic_df['Total Fwd Packets'] >= 0) &
        (traffic_df['Total Backward Packets'] >= 0) &

        # Total Length of Fwd/Bwd Packets >= 0
        (traffic_df['Total Length of Fwd Packets'] >= 0) &
        (traffic_df['Total Length of Bwd Packets'] >= 0) &

        # Forward/Backward Packet Length stats >= 0
        (traffic_df['Fwd Packet Length Mean'] >= 0) &
        (traffic_df['Fwd Packet Length Std'] >= 0) &
        (traffic_df['Bwd Packet Length Mean'] >= 0) &
        (traffic_df['Bwd Packet Length Std'] >= 0) &

        # Flow IAT and Fwd/Bwd IAT stats >= 0
        (traffic_df['Flow IAT Mean'] >= 0) &
        (traffic_df['Flow IAT Std'] >= 0) &
        (traffic_df['Fwd IAT Total'] >= 0) &
        (traffic_df['Fwd IAT Mean'] >= 0) &
        (traffic_df['Fwd IAT Std'] >= 0) &
        (traffic_df['Bwd IAT Total'] >= 0) &
        (traffic_df['Bwd IAT Mean'] >= 0) &
        (traffic_df['Bwd IAT Std'] >= 0) &

        # Header lengths >= 0
        (traffic_df['Fwd Header Length'] >= 0) &
        (traffic_df['Bwd Header Length'] >= 0) &

        # Min/Max Packet Length >= 0, plus means/stdev/variance
        (traffic_df['Packet Length Mean'] >= 0) &
        (traffic_df['Packet Length Std'] >= 0) &
        (traffic_df['Packet Length Variance'] >= 0) &

        # Flag counts >= 0
        (traffic_df['PSH Flag Count'] >= 0) &
        (traffic_df['ACK Flag Count'] >= 0) &

        # Subflows >= 0
        (traffic_df['Subflow Fwd Packets'] >= 0) &
        (traffic_df['Subflow Fwd Bytes'] >= 0) &
        (traffic_df['Subflow Bwd Packets'] >= 0) &
        (traffic_df['Subflow Bwd Bytes'] >= 0) &

        # TCP initial windows >= 0
        (traffic_df['Init_Win_bytes_forward'] >= 0) &
        (traffic_df['Init_Win_bytes_backward'] >= 0) &

        # Active data pkts and min seg size >= 0
        (traffic_df['act_data_pkt_fwd'] >= 0) &
        (traffic_df['min_seg_size_forward'] >= 0) &

        # Active and Idle stats >= 0
        (traffic_df['Active Mean'] >= 0) &
        (traffic_df['Active Std'] >= 0) &
        (traffic_df['Idle Mean'] >= 0) &
        (traffic_df['Idle Std'] >= 0) &

        # filter traffic types
        traffic_df['Label'].isin(valid_traffic_types)
    )
    # apply filter valid values rules
    traffic_df = traffic_df[valid_mask].copy()
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

    print('handle data skewness')
    # compute the threshold quantile
    total_fwd_packets_q_threshold = X_train['Total Fwd Packets'].quantile(0.95)
    # filter rows where 'Total Fwd Packets' is less than or equal to the threshold quantile
    X_train = X_train[X_train['Total Fwd Packets'] <= total_fwd_packets_q_threshold]
    X_test = X_test[X_test['Total Fwd Packets'] <= total_fwd_packets_q_threshold]
    print(f'traffic df shape: {X_train.shape}')

    # compute the threshold quantile
    total_backward_packets_q_threshold = X_train['Total Backward Packets'].quantile(0.95)
    # filter rows where 'Total Backward Packets' is less than or equal to the threshold quantile
    X_train = X_train[X_train['Total Backward Packets'] <= total_backward_packets_q_threshold]
    X_test = X_test[X_test['Total Backward Packets'] <= total_backward_packets_q_threshold]

    # compute the threshold quantile
    total_length_fwd_packets_q_threshold = X_train['Total Length of Fwd Packets'].quantile(0.95)
    # filter rows where 'Total Length of Fwd Packets' is less than or equal to the threshold quantile
    X_train = X_train[X_train['Total Length of Fwd Packets'] <= total_length_fwd_packets_q_threshold]
    X_test = X_test[X_test['Total Length of Fwd Packets'] <= total_length_fwd_packets_q_threshold]

    # compute the threshold quantile
    total_length_bwd_packets_q_threshold = X_train['Total Length of Bwd Packets'].quantile(0.95)
    # filter rows where 'Total Length of Bwd Packets' is less than or equal to the threshold quantile
    X_train = X_train[X_train['Total Length of Bwd Packets'] <= total_length_bwd_packets_q_threshold]
    X_test = X_test[X_test['Total Length of Bwd Packets'] <= total_length_bwd_packets_q_threshold]
    
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

    # Note: Below the under sampling you can find the
    print('apply categorical transformation')
    # one Hot Encoding
    encoder = OneHotEncoder(sparse_output=False)  # Set sparse=True for a sparse matrix
    # fit and transform data
    # define one hot encoding columns
    one_hot_encoding_columns = ['Source Port', 'Destination Port']
    # train encoding
    encoded = encoder.fit(X_train[one_hot_encoding_columns])
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

    # transform labels using label encoder
    le = LabelEncoder()
    le.fit(y_train)
    y_train = pd.Series(le.transform(y_train), index=y_train.index)
    y_test = pd.Series(le.transform(y_test), index=y_test.index)

    # merge features and labels in single dataframe
    y_train = y_train.loc[X_train.index]
    y_test = y_test.loc[X_test.index]
    X_train['Label'] = y_train
    X_test['Label'] = y_test

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

    print('save preprocessed data')
    # save preprocessed data
    train_traffic_filepath = 'traffic_preprocessed_train.csv'
    train_traffic_df.to_csv(train_traffic_filepath, index=False)
    test_traffic_filepath = 'traffic_preprocessed_test.csv'
    test_traffic_df.to_csv(test_traffic_filepath, index=False)

    # Log the hyperparameters
    params = {
        "relvant_columns": relevant_column,
        "valid_traffic": valid_traffic_types
    }
    mlflow.log_params(params)

    # return preprocessed data filepath
    return traffic_filepath

if __name__ == '__main__':
    base_traffic_filepath = '/home/cesarealice/PycharmProjects/generative-ml-malicious-traffic-wolf-in-sheeps-clothes/data/Weekly-WorkingHours_report.csv'

    base_traffic_cleaned_filepath = preprocessing(base_traffic_filepath, relevant_column=RELEVANT_COLUMNS,
                                                  valid_traffic_types=VALID_TRAFFIC_TYPES,
                                                  valid_port_range=VALID_PORT_RANGE,
                                                  valid_protocol_values=VALID_PROTOCOL_VALUES)
    print(base_traffic_cleaned_filepath)