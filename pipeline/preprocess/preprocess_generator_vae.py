"""
Author: Cesar M. Gonzalez

Preprocess script for network traffic data. Includes cleaning, feature engineering,
outlier removal, transformation, and sampling.
"""
import json
import os.path
from operator import index
from typing import List, Tuple, Dict # Added Dict for type hint
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PowerTransformer # Removed unused power_transform
from pipeline.preprocess.preprocess_base import map_port_usage_category, filter_valid_traffic_features, \
    undersampling_dataset
import mlflow
import joblib

from utils.constants import POWER_TRANSFORMER_NAME, ONEHOT_ENCODER_NAME, ISO_FOREST_MODEL_NAME, LABEL_ENCODER_NAME, \
    PREPROCESS_PARAMS_NAME, SCALER_NAME
from utils.utils import generate_profiling_report


# NOTE: Watch this video to understand about Kurtosis and Skewness:
# https://www.youtube.com/watch?v=EWuR4EGc9EY
# To learn about quantiles and percentiles:
# https://www.youtube.com/watch?v=IFKQLDmRK0Y
# To learn about standard deviation, mean, variance:
# https://www.youtube.com/watch?v=SzZ6GpcfoQY

def preprocessing(traffic_filepath: str, results_folder_path: str, relevant_column: List[str],
                  valid_traffic_types: List[str], valid_port_range: Tuple[int, int], # Made tuple more specific
                  n_instances_per_traffic_type: int = 150000,
                  test_size: float = 0.2) -> Tuple[str, str, Dict[str, str]]:
    """
    Performs preprocessing on raw network traffic data for the Generator model

    Steps include:
    1. Reading and initial cleaning (column selection, renaming).
    2. Filtering based on valid data ranges and traffic types.
    3. Splitting data into training and testing sets.
    4. Feature engineering: Mapping port numbers to categories.
    5. Outlier detection and removal using Z-score, KMeans (commented out test part), and Isolation Forest.
    6. Feature transformation: Power transformation (Yeo-Johnson) for numerical features.
    7. Feature encoding: One-Hot Encoding for categorical port features.
    9. Undersampling the training data to balance classes.
    10. Apply normalization using MinMaxScaler.
    11. Saving preprocessed data and transformation artifacts (scalers, encoders, models).
    12. Logging parameters and artifacts to MLflow.

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
    :return: A tuple containing:
             - Path to the preprocessed training data CSV file.
             - Path to the preprocessed test data CSV file.
             - Dictionary of paths to saved preprocessing artifacts (transformer, encoders, models, params).
    :rtype: Tuple[str, str, Dict[str, str]]
    """
    # --- 1. Read and Initial Clean ---
    print("Reading traffic data...")
    # Use low_memory=False for potentially mixed-type columns, but monitor memory usage
    traffic_df = pd.read_csv(traffic_filepath, low_memory=False)

    print("Initial data cleaning...")
    # Keep only specified relevant columns
    traffic_df = traffic_df[relevant_column]
    # Standardize column names by stripping leading/trailing whitespace
    traffic_df.rename(columns=lambda x: x.strip(), inplace=True)

    # --- 2. Filter Valid Data ---
    print('Filtering by valid data ranges and traffic types...')
    min_port, max_port = valid_port_range
    # Apply filtering rules defined in preprocess_base
    traffic_df = filter_valid_traffic_features(traffic_df, min_port, max_port, valid_traffic_types)
    print(f'Shape after initial filtering: {traffic_df.shape}')
    if traffic_df.empty:
        raise ValueError("DataFrame is empty after initial filtering. Check filter criteria and input data.")

    raw_traffic_df = undersampling_dataset(valid_traffic_types, traffic_df.copy(), n_instances_per_traffic_type)
    raw_traffic_filepath = os.path.join(results_folder_path, 'raw_traffic.csv')
    raw_traffic_df.to_csv(raw_traffic_filepath, index=False)
    title = "Raw dataset Profiling"
    report_name = 'preprocessing_traffic_raw_dataset_profiling_generator'
    report_filepath = os.path.join(results_folder_path, f"{report_name}.html")
    type_schema = {'Label': "categorical"}
    generate_profiling_report(report_filepath=report_filepath, title=title, data_filepath=raw_traffic_filepath,
                              type_schema=type_schema, minimal=True)

    # --- 3. Train/Test Split ---
    print(f'Splitting data with test_size={test_size}...')
    # Stratify is often useful for classification tasks to maintain class proportions, consider adding stratify=label
    X_train, X_test = train_test_split(traffic_df, test_size=test_size, random_state=42)
    print(f'Train shape: {X_train.shape}, Test shape: {X_test.shape}')

    # --- 4. Feature Engineering: Port Categories ---
    print('Transforming port numbers to categories...')
    # Use .loc to avoid potential SettingWithCopyWarning
    X_train.loc[:, 'Source Port'] = X_train['Source Port'].map(map_port_usage_category)
    X_train.loc[:, 'Destination Port'] = X_train['Destination Port'].map(map_port_usage_category)
    X_test.loc[:, 'Source Port'] = X_test['Source Port'].map(map_port_usage_category)
    X_test.loc[:, 'Destination Port'] = X_test['Destination Port'].map(map_port_usage_category)

    # --- 5. Outlier Detection/Removal (Applied sequentially) ---
    # Note: Applying outlier removal sequentially can interact. Consider the order carefully.
    # It's also applied *before* power transformation and one-hot encoding.

    # --- 5a. Z-Score Outlier Removal (Example on 'Fwd Packet Length Mean') ---
    print('Applying Z-score outlier removal for Fwd Packet Length Mean...')
    fwd_packet_len_mean_col = 'Fwd Packet Length Mean'
    threshold_z = 3
    # Calculate mean and std *only* from training data
    mean_fwd_packet_len_mean = X_train[fwd_packet_len_mean_col].mean()
    std_fwd_packet_len_mean = X_train[fwd_packet_len_mean_col].std()
    # Avoid division by zero if std is 0
    # Compute Z-scores for train
    z_scores_train = (X_train[fwd_packet_len_mean_col] - mean_fwd_packet_len_mean) / std_fwd_packet_len_mean
    train_inliers_mask = np.abs(z_scores_train) <= threshold_z
    # Compute Z-scores for test using train mean/std
    z_scores_test = (X_test[fwd_packet_len_mean_col] - mean_fwd_packet_len_mean) / std_fwd_packet_len_mean
    test_inliers_mask = np.abs(z_scores_test) <= threshold_z

    # Filter both X and y based on train mask
    X_train = X_train.loc[train_inliers_mask]
    # Filter test set
    X_test = X_test.loc[test_inliers_mask]
    print(f'Shape after Z-score on {fwd_packet_len_mean_col}: Train={X_train.shape}, Test={X_test.shape}')

    # --- 5b. Z-Score Outlier Removal (Example on 'Fwd Packet Length Std') ---
    # (Repeating the pattern for another column)
    print('Applying Z-score outlier removal for Fwd Packet Length Std...')
    fwd_packet_len_std_col = 'Fwd Packet Length Std'
    # Calculate mean and std *only* from training data
    mean_fwd_packet_len_std = X_train[fwd_packet_len_std_col].mean()
    std_fwd_packet_len_std = X_train[fwd_packet_len_std_col].std()
    # Compute Z-scores for train
    z_scores_train_std = (X_train[fwd_packet_len_std_col] - mean_fwd_packet_len_std) / std_fwd_packet_len_std
    train_inliers_mask_std = np.abs(z_scores_train_std) <= threshold_z
    # Compute Z-scores for test using train mean/std
    z_scores_test_std = (X_test[fwd_packet_len_std_col] - mean_fwd_packet_len_std) / std_fwd_packet_len_std
    test_inliers_mask_std = np.abs(z_scores_test_std) <= threshold_z

    # Filter both X and y based on train mask
    X_train = X_train.loc[train_inliers_mask_std]
    # Filter test set
    X_test = X_test.loc[test_inliers_mask_std]
    print(f'Shape after Z-score on {fwd_packet_len_std_col}: Train={X_train.shape}, Test={X_test.shape}')


    # --- 5c. KMeans Outlier Removal (Example on Bwd Packet Length features) ---
    # Note: This scales, fits KMeans, calculates distances, finds a threshold, and filters.
    # The test set filtering part was commented out in the original code.
    # Consider if this step is necessary given Isolation Forest is applied later.
    print('Applying KMeans outlier removal for Bwd Packet Length Mean/Std...')
    bwd_packet_len_columns = ['Bwd Packet Length Mean', 'Bwd Packet Length Std']
    # Check if columns exist before proceeding
    bwd_packet_length_distribution_train = X_train[bwd_packet_len_columns]
    bwd_packet_length_distribution_test = X_test[bwd_packet_len_columns]

    # Scale data (fit on train, transform train and test)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(bwd_packet_length_distribution_train)
    train_bwd_packet_len_dis_scaled = scaler.transform(bwd_packet_length_distribution_train)
    test_bwd_packet_len_dis_scaled = scaler.transform(bwd_packet_length_distribution_test)  # Scale test data too

    # Apply K-Means clustering (fit only on train)
    k = 3  # Number of clusters. Consider using Elbow method or Silhouette score to find optimal k.
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # n_init='auto' in newer sklearn
    kmeans.fit(train_bwd_packet_len_dis_scaled)

    # Compute distances to the nearest cluster centroid for training data
    distances_train = np.linalg.norm(train_bwd_packet_len_dis_scaled - kmeans.cluster_centers_[kmeans.labels_],
                                     axis=1)
    # Define threshold based on percentile of distances (removes top 5% furthest points)
    threshold_kmeans = np.percentile(distances_train, 95)

    # Filter training data based on distance threshold
    kmeans_inliers_mask_train = distances_train <= threshold_kmeans
    X_train = X_train.loc[kmeans_inliers_mask_train]

    # --- Filter Test Data using KMeans (Optional - was commented out) ---
    # If applying to test set, calculate distances for test points to *trained* centroids
    # test_labels = kmeans.predict(test_bwd_packet_len_dis_scaled)
    # distances_test = np.linalg.norm(test_bwd_packet_len_dis_scaled - kmeans.cluster_centers_[test_labels], axis=1)
    # kmeans_inliers_mask_test = distances_test <= threshold_kmeans # Use same threshold from train
    # X_test = X_test.loc[kmeans_inliers_mask_test]
    # y_test = y_test.loc[kmeans_inliers_mask_test] # Keep y aligned
    # print(f'Shape after KMeans outlier removal: Train={X_train.shape}, Test={X_test.shape}')
    print(f'Shape after KMeans outlier removal (Train only): Train={X_train.shape}, Test={X_test.shape}')  # Updated print statement

    # --- 6. Feature Transformation: Power Transformation ---
    print('Applying Power Transformation (Yeo-Johnson)...')
    # Define columns for power transformation
    power_columns = ['Total Length of Fwd Packets', 'Total Length of Bwd Packets',
                     'Flow Duration', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Mean',
                     'Bwd Packet Length Std', 'Flow IAT Mean', 'Flow IAT Std', 'Fwd IAT Total', 'Fwd IAT Mean',
                     'Fwd IAT Std', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std']
    # Ensure all power columns exist in the dataframe after outlier removal
    power_columns = [col for col in power_columns if col in X_train.columns]
    pt_model = PowerTransformer(method='yeo-johnson')
    # Fit on training data only
    pt_model.fit(X_train[power_columns])
    # Transform both train and test data
    # Use .loc to assign back safely
    X_train.loc[:, power_columns] = pt_model.transform(X_train[power_columns])
    X_test.loc[:, power_columns] = pt_model.transform(X_test[power_columns])
    # Save the fitted PowerTransformer
    power_transformer_filepath = os.path.join(results_folder_path, POWER_TRANSFORMER_NAME)
    joblib.dump(pt_model, power_transformer_filepath)
    print(f"PowerTransformer saved to {power_transformer_filepath}")

    #label_train = X_train['Label']
    #label_test = X_test['Label']
    label_train = X_train.pop('Label')
    label_test = X_test.pop('Label')

    # --- 7. Feature Encoding: One-Hot Encoding ---
    print('Applying One-Hot Encoding for port categories...')
    #one_hot_encoding_columns = ['Source Port', 'Destination Port', 'Label']
    one_hot_encoding_columns = ['Source Port', 'Destination Port']
    # Ensure columns exist
    one_hot_encoding_columns = [col for col in one_hot_encoding_columns if col in X_train.columns]
    # Initialize OneHotEncoder
    # handle_unknown='ignore' prevents errors if test set has categories not seen in train
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    # Fit on training data
    onehot_encoder.fit(X_train[one_hot_encoding_columns])

    # Transform train data
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

    # --- 8. Outlier Detection/Removal (Isolation Forest - Applied AFTER transformations) ---
    # Applying Isolation Forest on the transformed/encoded data
    print('Applying Isolation Forest for global outlier removal...')
    # Ensure X_train is not empty before fitting

    iso_forest = IsolationForest(n_estimators=200, contamination='auto',
                                 random_state=42)  # 'auto' contamination is often a good starting point
    # Fit the model only on training data
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

    print(f'Shape after Isolation Forest: Train={X_train.shape}, Test={X_test.shape}')

    # Save the fitted IsolationForest model
    iso_forest_model_filepath = os.path.join(results_folder_path, ISO_FOREST_MODEL_NAME)
    joblib.dump(iso_forest, iso_forest_model_filepath)
    print(f"IsolationForest model saved to {iso_forest_model_filepath}")

    # --- 9. Normalize data (Applied only to Training Data) ---
    # init Min Max scaler to normalize dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
    scaler_model_filepath = os.path.join(results_folder_path, SCALER_NAME)
    joblib.dump(scaler, scaler_model_filepath)

    # --- 9. Undersampling (Applied only to Training Data) ---
    print('Applying undersampling to balance training data...')
    # Re-attach labels temporarily for sampling
    X_train_temp = X_train.copy()  # Work on a copy

    X_train_temp['Label'] = label_train
    X_test['Label'] = label_test

    x_train_traffic_df_list = []
    for valid_traffic_type in valid_traffic_types:
        # Filter by type
        x_train_traffic_type_df = X_train_temp.loc[X_train_temp['Label'] == valid_traffic_type]
        n_available = x_train_traffic_type_df.shape[0]

        # Check if there are enough instances and if sampling is needed
        if n_available > n_instances_per_traffic_type:
            print(
                f"Sampling {n_instances_per_traffic_type} instances for type '{valid_traffic_type}' (available: {n_available})")
            x_train_traffic_type_df = x_train_traffic_type_df.sample(n=n_instances_per_traffic_type, random_state=42)
        elif n_available > 0:
            print(
                f"Keeping all {n_available} instances for type '{valid_traffic_type}' (less than target {n_instances_per_traffic_type})")
        else:
            print(f"Warning: No instances found for type '{valid_traffic_type}' after previous steps.")
            continue

        # Append the sampled/kept data
        x_train_traffic_df_list.append(x_train_traffic_type_df)

    # Concatenate sampled dataframes if list is not empty
    train_traffic_df = pd.concat(x_train_traffic_df_list, axis=0)
    # Separate features and labels again
    y_train_sampled = train_traffic_df.pop('Label')
    X_train_sampled = train_traffic_df
    print(f"Shape after undersampling: Train={X_train_sampled.shape}")
    print("Class distribution in sampled training data:")
    print(y_train_sampled.value_counts())

    # Keep the original (but filtered) test set
    test_traffic_df = X_test.copy()

    mlflow.log_param("final_preprocess_columns", str(list(X_train_sampled.columns)))

    # --- 11. Save Preprocessed Data ---
    print('Saving preprocessed data...')
    # Define file paths
    train_traffic_filepath = os.path.join(results_folder_path, 'traffic_preprocessed_train.csv')
    test_traffic_filepath = os.path.join(results_folder_path, 'traffic_preprocessed_test_bvae.csv')
    # Save to CSV
    X_train_sampled.to_csv(train_traffic_filepath, index=False)
    test_traffic_df.to_csv(test_traffic_filepath, index=False)
    print(f"Preprocessed train data saved to {train_traffic_filepath}")
    print(f"Preprocessed test data saved to {test_traffic_filepath}")

    # generate dataset profiling report
    title = "Preprocessing Train dataset Profiling"
    report_name = 'preprocessing_traffic_train_dataset_profiling_generator'
    report_filepath = os.path.join(results_folder_path, f"{report_name}.html")
    type_schema = {'Label': "categorical"}
    generate_profiling_report(report_filepath=report_filepath, title=title,
                              data_filepath=train_traffic_filepath,
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
        "z_score_threshold": threshold_z,
        # Add other relevant parameters like KMeans k, threshold, Isolation Forest params if desired
    }
    # Log parameters to MLflow
    mlflow.log_params({k: str(v) if isinstance(v, (list, dict)) else v for k, v in
                       params.items()})  # Convert list/dict to str for MLflow UI

    # Save parameters locally as JSON
    preprocess_params_filepath = os.path.join(results_folder_path, PREPROCESS_PARAMS_NAME)
    with open(preprocess_params_filepath, 'w') as f:
        # Use json.dumps for better handling of potential non-serializable types if needed
        json.dump(params, f, indent=4)
    print(f"Preprocessing parameters saved locally to {preprocess_params_filepath}")

    # Collect artifact paths, handling cases where artifacts might not have been created
    preprocess_artifacts = {
        "power_transformer": power_transformer_filepath,
        "onehot_encoder": onehot_encoder_filepath,
        "iso_forest_model": iso_forest_model_filepath,
        "scaler": scaler_model_filepath,
        "preprocess_params": preprocess_params_filepath
    }
    # Filter out None values before returning
    preprocess_artifacts = {k: v for k, v in preprocess_artifacts.items() if v is not None}

    # Return paths to preprocessed data and the dictionary of artifact paths
    return train_traffic_filepath, test_traffic_filepath, preprocess_artifacts