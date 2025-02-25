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
from sklearn.preprocessing import RobustScaler
import mlflow


# NOTE: Watch this video to understand about Kurtosis and Skewness:
# https://www.youtube.com/watch?v=EWuR4EGc9EY
# To learn about quantiles and percentiles:
# https://www.youtube.com/watch?v=IFKQLDmRK0Y
# To learn about standard deviation, mean, variance:
# https://www.youtube.com/watch?v=SzZ6GpcfoQY

def preprocessing(traffic_filepath: str, relevant_column: List[str], valid_traffic_types: List[str],
                  valid_port_range: Tuple, valid_protocol_values: List[str], n_instances_per_traffic_type:int =150000,
                  test_size: float = 0.2) -> str:

    traffic_df = pd.read_csv(traffic_filepath)
    #
    print("data cleaning")
    # filter relevant columns
    traffic_df = traffic_df[relevant_column]
    # format the columns names
    traffic_df.rename(columns=lambda x: x.strip(), inplace=True)

    base_traffic_filepath = 'traffic_preprocessed_base.csv'
    traffic_df.to_csv(base_traffic_filepath, index=False)


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

    # apply destination and source ports Well-Known Port Ranges for Standardized Functionalities transformations
    traffic_df['Source Port'] = traffic_df['Source Port'].map(map_port_usage_category)
    traffic_df['Destination Port'] = traffic_df['Destination Port'].map(map_port_usage_category)

    # compute the threshold quantile
    total_fwd_packets_q_threshold = traffic_df['Total Fwd Packets'].quantile(0.95)
    # filter rows where 'Total Fwd Packets' is less than or equal to the threshold quantile
    traffic_df = traffic_df[traffic_df['Total Fwd Packets'] <= total_fwd_packets_q_threshold]

    # compute the threshold quantile
    total_backward_packets_q_threshold = traffic_df['Total Backward Packets'].quantile(0.95)
    # filter rows where 'Total Backward Packets' is less than or equal to the threshold quantile
    traffic_df = traffic_df[traffic_df['Total Backward Packets'] <= total_backward_packets_q_threshold]

    # compute the threshold quantile
    total_length_fwd_packets_q_threshold = traffic_df['Total Length of Fwd Packets'].quantile(0.95)
    # filter rows where 'Total Length of Fwd Packets' is less than or equal to the threshold quantile
    traffic_df = traffic_df[traffic_df['Total Length of Fwd Packets'] <= total_length_fwd_packets_q_threshold]

    # compute the threshold quantile
    total_length_bwd_packets_q_threshold = traffic_df['Total Length of Bwd Packets'].quantile(0.95)
    # filter rows where 'Total Length of Bwd Packets' is less than or equal to the threshold quantile
    traffic_df = traffic_df[traffic_df['Total Length of Bwd Packets'] <= total_length_bwd_packets_q_threshold]
    
    # Apply anomaly detection model to clean the distribution features. It includes features with mean and standard deviation
    # NOTE: Several methods could be applied like, z-score, quantiles, DBSCAN, isolation forest, KMeans, etc.
    # Z-score: https://vitalflux.com/outlier-detection-techniques-in-python/
    # Quantiles: https://vitalflux.com/outlier-detection-techniques-in-python/
    # KMeans: https://blog.zhaytam.com/2019/08/06/outliers-detection-in-pyspark-3-k-means/
    # DBSCAN: https://medium.com/@dilip.voleti/dbscan-algorithm-for-fraud-detection-outlier-detection-in-a-data-set-60a10ad06ea8
    # Isolation Forest: https://hands-on.cloud/using-python-and-isolation-forest-algorithm-for-anomalies-detection/
    # Mahalanobis Distance: https://blog.dailydoseofds.com/p/the-limitation-of-euclidean-distance

    # anomaly detection using z-score
    # Fwd Packet Length Mean
    fwd_packet_length_mean = traffic_df['Fwd Packet Length Mean']
    threshold_fwd_packet_length_mean = 3
    mean_fwd_packet_length_mean = np.mean(fwd_packet_length_mean)
    std_dev_fwd_packet_length_mean = np.std(fwd_packet_length_mean)
    # Compute Z-scores
    z_scores_fwd_packet_length_mean = (fwd_packet_length_mean - mean_fwd_packet_length_mean) / std_dev_fwd_packet_length_mean
    # Set threshold for anomaly detection
    traffic_df = traffic_df[np.abs(z_scores_fwd_packet_length_mean) <= threshold_fwd_packet_length_mean]
    # Fwd Packet Length Std
    fwd_packet_length_std = traffic_df['Fwd Packet Length Std']
    threshold_fwd_packet_length_std = 3
    mean_fwd_packet_length_std = np.mean(fwd_packet_length_std)
    std_dev_fwd_packet_length_std = np.std(fwd_packet_length_std)
    # compute Z-scores
    z_scores_fwd_packet_length_std = (fwd_packet_length_std - mean_fwd_packet_length_std) / std_dev_fwd_packet_length_std
    # Set threshold for anomaly detection
    traffic_df = traffic_df[np.abs(z_scores_fwd_packet_length_std) <= threshold_fwd_packet_length_std]

    # anomaly detection using KMeans
    # apply normalization
    bwd_packet_length_distribution = traffic_df[['Bwd Packet Length Mean', 'Bwd Packet Length Std']]
    scaler = MinMaxScaler(feature_range=(0, 1))  # Scaling between [0,1]
    bwd_packet_length_distribution_scaled = scaler.fit_transform(bwd_packet_length_distribution)
    # Apply K-Means clustering
    k = 3  # Number of clusters. What is the best number. Elbow?
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(bwd_packet_length_distribution_scaled)
    # Compute distances to the nearest cluster centroid
    distances = np.linalg.norm(bwd_packet_length_distribution_scaled - kmeans.cluster_centers_[kmeans.labels_], axis=1)
    # Define threshold. based on the quantiles
    threshold = np.percentile(distances, 95)
    # filter by the centroid distance
    traffic_df = traffic_df[distances <= threshold]


    #trans = RobustScaler()
    #traffic_df['Total Fwd Packets'] = trans.fit_transform(traffic_df[['Total Fwd Packets']])
    #trans = RobustScaler()
    #traffic_df['Bwd IAT Total Robust'] = trans.fit_transform(traffic_df[['Bwd IAT Total']])
    pt = PowerTransformer(method='yeo-johnson')
    power_columns = ['Flow Duration', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Mean',
                     'Bwd Packet Length Std', 'Flow IAT Mean', 'Flow IAT Std', 'Fwd IAT Total', 'Fwd IAT Mean',
                     'Fwd IAT Std', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std']
    traffic_df[power_columns] = pt.fit_transform(traffic_df[power_columns])


    # Note: Below the under sampling you can find the

    # one Hot Encoding
    encoder = OneHotEncoder(sparse_output=False)  # Set sparse=True for a sparse matrix
    # fit and transform data
    # define one hot encoding columns
    one_hot_encoding_columns = ['Source Port', 'Destination Port']
    #
    encoded = encoder.fit(traffic_df[one_hot_encoding_columns])
    encoded_data = encoded.transform(traffic_df[one_hot_encoding_columns])
    # encoded data
    encoded_data_df = pd.DataFrame(encoded_data, columns=encoded.get_feature_names_out(one_hot_encoding_columns),
                                   index=traffic_df.index)
    # Concatenate the original DataFrame with the drivetrain encoded DataFrame
    traffic_df = pd.concat([traffic_df, encoded_data_df], axis=1)
    # remove categorical columns
    traffic_df = traffic_df.loc[:, ~traffic_df.columns.isin(one_hot_encoding_columns)]

    # Apply Down sampling to solve the unbalanced
    traffic_df_list = []
    for valid_traffic_type in valid_traffic_types:
        # filter by type and sample
        traffic_type_df = traffic_df.loc[traffic_df['Label'] == valid_traffic_type]
        # check if there are enough instances to apply sampling
        if traffic_type_df.shape[0] > n_instances_per_traffic_type:
            traffic_type_df = traffic_type_df.sample(n_instances_per_traffic_type)
        # concatenate result
        traffic_df_list.append(traffic_type_df)
    # concatenate traffic df
    traffic_df = pd.concat(traffic_df_list, axis=0)

    # pop label
    label = traffic_df.pop('Label')

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(traffic_df, label, test_size=test_size, random_state=42)

    # Or you could remove the
    iso_forest = IsolationForest(n_estimators=200, contamination='auto', random_state=42)
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

    # Save Isolation Forest model
    #iso_filepath = os.path.join(artifacts_folder, PR_OUTLIER_DETECTION_MODEL_NAME)
    #joblib.dump(iso_forest, iso_filepath)

    train_traffic_df = X_train.copy()
    test_traffic_df = X_test.copy()

    # Add price columns
    train_traffic_df['Label'] = y_train
    test_traffic_df['Label'] = y_test

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