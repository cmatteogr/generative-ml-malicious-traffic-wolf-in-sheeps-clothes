"""
Author: Cesar M. Gonzalez

Preprocess
"""
from typing import List, Tuple
import pandas as pd
from pipeline.preprocess.preprocess_base import map_port_usage_category

# NOTE: Watch this video to understand about Kurtosis and Skewness:
# https://www.youtube.com/watch?v=EWuR4EGc9EY
# To learn about quantiles and percentiles:
# https://www.youtube.com/watch?v=IFKQLDmRK0Y
# To learn about standard deviation, mean, variance:
# https://www.youtube.com/watch?v=SzZ6GpcfoQY

def preprocessing(traffic_filepath: str, relevant_column: List[str], valid_traffic_types: List[str],
                  valid_port_range: Tuple, valid_protocol_values: List[str]) -> str:

    traffic_df = pd.read_csv(traffic_filepath)
    #
    print("data cleaning")
    # filter relevant columns
    traffic_df = traffic_df[relevant_column]
    # format the columns names
    traffic_df.rename(columns=lambda x: x.strip(), inplace=True)

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
        #(traffic_df['Fwd Packet Length Max'] >= 0) &
        #(traffic_df['Fwd Packet Length Min'] >= 0) &
        (traffic_df['Fwd Packet Length Mean'] >= 0) &
        (traffic_df['Fwd Packet Length Std'] >= 0) &
        #(traffic_df['Bwd Packet Length Max'] >= 0) &
        #(traffic_df['Bwd Packet Length Min'] >= 0) &
        (traffic_df['Bwd Packet Length Mean'] >= 0) &
        (traffic_df['Bwd Packet Length Std'] >= 0) &

        # Rates (Bytes/s, Packets/s) >= 0
        #(traffic_df['Flow Bytes/s'] >= 0) &
        #(traffic_df['Flow Packets/s'] >= 0) &

        # Flow IAT and Fwd/Bwd IAT stats >= 0
        (traffic_df['Flow IAT Mean'] >= 0) &
        (traffic_df['Flow IAT Std'] >= 0) &
        #(traffic_df['Flow IAT Max'] >= 0) &
        #(traffic_df['Flow IAT Min'] >= 0) &
        (traffic_df['Fwd IAT Total'] >= 0) &
        (traffic_df['Fwd IAT Mean'] >= 0) &
        (traffic_df['Fwd IAT Std'] >= 0) &
        #(traffic_df['Fwd IAT Max'] >= 0) &
        #(traffic_df['Fwd IAT Min'] >= 0) &
        (traffic_df['Bwd IAT Total'] >= 0) &
        (traffic_df['Bwd IAT Mean'] >= 0) &
        (traffic_df['Bwd IAT Std'] >= 0) &
        #(traffic_df['Bwd IAT Max'] >= 0) &
        #(traffic_df['Bwd IAT Min'] >= 0) &

        # Header lengths >= 0
        (traffic_df['Fwd Header Length'] >= 0) &
        (traffic_df['Bwd Header Length'] >= 0) &

        # Fwd/Bwd Packets per second >= 0
        #(traffic_df['Fwd Packets/s'] >= 0) &
        #(traffic_df['Bwd Packets/s'] >= 0) &

        # Min/Max Packet Length >= 0, plus means/stdev/variance
        #(traffic_df['Min Packet Length'] >= 0) &
        #(traffic_df['Max Packet Length'] >= 0) &
        (traffic_df['Packet Length Mean'] >= 0) &
        (traffic_df['Packet Length Std'] >= 0) &
        (traffic_df['Packet Length Variance'] >= 0) &

        # Flag counts >= 0
        (traffic_df['PSH Flag Count'] >= 0) &
        (traffic_df['ACK Flag Count'] >= 0) &

        # Down/Up Ratio >= 0
        #(traffic_df['Down/Up Ratio'] >= 0) &

        # Average packet sizes >= 0
        #(traffic_df['Average Packet Size'] >= 0) &
        #(traffic_df['Avg Fwd Segment Size'] >= 0) &
        #(traffic_df['Avg Bwd Segment Size'] >= 0) &

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
        #(traffic_df['Active Max'] >= 0) &
        #(traffic_df['Active Min'] >= 0) &
        (traffic_df['Idle Mean'] >= 0) &
        (traffic_df['Idle Std'] >= 0) &
        #(traffic_df['Idle Max'] >= 0) &
        #(traffic_df['Idle Min'] >= 0) &

        # filter traffic types
        traffic_df['Label'].isin(valid_traffic_types)
    )
    # apply filter valid values rules
    traffic_df = traffic_df[valid_mask].copy()

    # apply destination and source ports Well-Known Port Ranges for Standardized Functionalities transformations
    traffic_df['Source Port'] = traffic_df['Source Port'].map(map_port_usage_category)
    traffic_df['Destination Port'] = traffic_df['Destination Port'].map(map_port_usage_category)



    # Apply Down sampling to solve the unbalanced
    n_instances_per_traffic_type = 150000
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

    # save preprocessed data
    traffic_filepath = 'traffic_preprocessed.csv'
    traffic_df.to_csv(traffic_filepath, index=False)

    # return preprocessed data filepath
    return traffic_filepath