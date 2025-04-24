"""
preprocess base operations
"""
import pandas as pd
from typing import List # Import List for type hinting


def map_port_usage_category(port: int) -> str:
    """
    Maps a given network port number to its standard usage category based on IANA ranges.

    - Ports 0-1023: "well_known" (System Ports)
    - Ports 1024-49151: "registered_ports" (User Ports)
    - Ports 49152-65535: "dynamic_ephemeral_ports" (Dynamic/Private Ports)

    Note: The logic in the original implementation had errors comparing ranges.
          This version corrects the range checks.

    :param port: The network port number to categorize.
    :type port: int
    :return: The name of the port category ('well_known', 'registered_ports',
             'dynamic_ephemeral_ports').
    :rtype: str
    :raises ValueError: If the provided port number is outside the valid range (0-65535).
    """
    # Check for Well-Known Ports range (0-1023)
    # Port 0 is technically valid but often reserved/unused in practice.
    # Corrected logic: Use chained comparison or 'and' for ranges.
    if 0 <= port <= 1023:
        return "well_known"
    # Check for Registered Ports range (1024-49151)
    # Use elif for mutually exclusive conditions
    elif 1024 <= port <= 49151:
        return "registered_ports"
    # Check for Dynamic/Private/Ephemeral Ports range (49152-65535)
    elif 49152 <= port <= 65535:
        return "dynamic_ephemeral_ports"
    else:
        # Raise an error if the port is outside the 0-65535 range
        raise ValueError(f"Invalid port {port}. Port must be between 0 and 65535.")


def filter_valid_traffic_features(traffic_df: pd.DataFrame, min_port: int, max_port: int,
                                  valid_traffic_types: List[str]) -> pd.DataFrame:
    """
    Filters a DataFrame of network traffic data based on validity criteria.

    Removes rows where:
    - Source or Destination Port is outside [min_port, max_port].
    - Various numeric features (durations, counts, lengths, stats, etc.) are negative.
    - The 'Label' is not in the `valid_traffic_types` list.

    :param traffic_df: Input DataFrame with network traffic features.
    :type traffic_df: pd.DataFrame
    :param min_port: Minimum valid port number (inclusive).
    :type min_port: int
    :param max_port: Maximum valid port number (inclusive).
    :type max_port: int
    :param valid_traffic_types: List of allowed traffic labels (e.g., ['BENIGN', 'MALICIOUS']).
    :type valid_traffic_types: List[str]
    :return: A filtered DataFrame containing only valid rows.
    :rtype: pd.DataFrame
    """
    # Create a boolean mask for rows meeting all validity conditions
    valid_mask = (
        # --- Port Filtering ---
        # Ensure Source and Destination ports are within the specified valid range.
        traffic_df['Source Port'].between(min_port, max_port, inclusive='both') &
        traffic_df['Destination Port'].between(min_port, max_port, inclusive='both') &

        # --- Basic Flow Sanity Checks (Non-Negative Values) ---
        (traffic_df['Flow Duration'] >= 0) &
        (traffic_df['Total Fwd Packets'] >= 0) &
        (traffic_df['Total Backward Packets'] >= 0) &
        (traffic_df['Total Length of Fwd Packets'] >= 0) &
        (traffic_df['Total Length of Bwd Packets'] >= 0) &

        # --- Packet Length Statistics Sanity Checks (Non-Negative Values) ---
        (traffic_df['Fwd Packet Length Mean'] >= 0) &
        (traffic_df['Fwd Packet Length Std'] >= 0) &
        (traffic_df['Bwd Packet Length Mean'] >= 0) &
        (traffic_df['Bwd Packet Length Std'] >= 0) &
        (traffic_df['Packet Length Mean'] >= 0) &
        (traffic_df['Packet Length Std'] >= 0) &
        (traffic_df['Packet Length Variance'] >= 0) & # Variance must also be non-negative

        # --- Inter-Arrival Time (IAT) Sanity Checks (Non-Negative Values) ---
        (traffic_df['Flow IAT Mean'] >= 0) &
        (traffic_df['Flow IAT Std'] >= 0) &
        (traffic_df['Fwd IAT Total'] >= 0) &
        (traffic_df['Fwd IAT Mean'] >= 0) &
        (traffic_df['Fwd IAT Std'] >= 0) &
        (traffic_df['Bwd IAT Total'] >= 0) &
        (traffic_df['Bwd IAT Mean'] >= 0) &
        (traffic_df['Bwd IAT Std'] >= 0) &

        # --- Header and Segment Sanity Checks (Non-Negative Values) ---
        (traffic_df['Fwd Header Length'] >= 0) &
        (traffic_df['Bwd Header Length'] >= 0) &
        (traffic_df['min_seg_size_forward'] >= 0) &

        # --- Flag Count Sanity Checks (Non-Negative Values) ---
        (traffic_df['PSH Flag Count'] >= 0) &
        (traffic_df['ACK Flag Count'] >= 0) & # Add others (URG, FIN, SYN, RST) if present and needed

        # --- Subflow Sanity Checks (Non-Negative Values) ---
        (traffic_df['Subflow Fwd Packets'] >= 0) &
        (traffic_df['Subflow Fwd Bytes'] >= 0) &
        (traffic_df['Subflow Bwd Packets'] >= 0) &
        (traffic_df['Subflow Bwd Bytes'] >= 0) &

        # --- TCP Window Sanity Checks (Non-Negative Values) ---
        (traffic_df['Init_Win_bytes_forward'] >= 0) &
        (traffic_df['Init_Win_bytes_backward'] >= 0) &

        # --- Active/Idle Time Sanity Checks (Non-Negative Values) ---
        (traffic_df['act_data_pkt_fwd'] >= 0) &
        (traffic_df['Active Mean'] >= 0) &
        (traffic_df['Active Std'] >= 0) &
        (traffic_df['Idle Mean'] >= 0) &
        (traffic_df['Idle Std'] >= 0) &

        # --- Traffic Type Filtering ---
        # Ensure the 'Label' column contains one of the allowed traffic types.
        traffic_df['Label'].isin(valid_traffic_types)
    )

    # Apply the filter mask to the DataFrame.
    # Using .loc for boolean indexing is often preferred, and .copy() prevents SettingWithCopyWarning.
    filtered_df = traffic_df.loc[valid_mask].copy()

    # Return the filtered DataFrame
    return filtered_df