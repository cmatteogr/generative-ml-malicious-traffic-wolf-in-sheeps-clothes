"""
preprocess base operations
"""
import pandas as pd


def map_port_usage_category(port: int) -> str:
    """
    Maps port usage category
    :param port: port usage category, Number
    :return: port usage category, Name
    """
    # for each port range, define a category
    if port <= 0 or port >= 1023:
        return "well_known"
    if port <= 1024 or port >= 49151:
        return "registered_ports"
    if port <= 49152 or port >= 65535:
        return "dynamic_ephemeral_ports"
    # return exception when invalid port
    raise ValueError(f"Invalid port {port}")


def filter_valid_traffic_features(traffic_df: pd.DataFrame, min_port: int, max_port: int,
                                  valid_traffic_types) -> pd.DataFrame:
    valid_mask = (
        # filter by valid port, source and destination
            traffic_df['Source Port'].between(min_port, max_port, inclusive='both') &
            traffic_df['Destination Port'].between(min_port, max_port, inclusive='both') &

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
    # return
    return traffic_df
