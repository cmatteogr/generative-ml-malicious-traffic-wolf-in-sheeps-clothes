"""
constants
"""
VALID_TRAFFIC_TYPES = [
    'BENIGN',
    'DoS Hulk',
    'PortScan',
    'DDoS',
    'DoS GoldenEye',
    'FTP-Patator'
]

VALID_PORT_RANGE = (0, 65535)
VALID_PROTOCOL_VALUES  = [6, 17] # 6 = TCP, 17 = UDP

# NOTE: Remove the min max from distributions, this could change in the future
RELEVANT_COLUMNS = [
    ' Source Port',
    ' Destination Port',

    ' Flow Duration',

    ' Total Fwd Packets',
    ' Total Backward Packets',

    'Total Length of Fwd Packets',
    ' Total Length of Bwd Packets',

    ' Fwd Packet Length Mean',
    ' Fwd Packet Length Std',
    ' Bwd Packet Length Mean',
    ' Bwd Packet Length Std',

    ' Flow IAT Mean',
    ' Flow IAT Std',

    'Fwd IAT Total',
    ' Fwd IAT Mean',
    ' Fwd IAT Std',
    'Bwd IAT Total',
    ' Bwd IAT Mean',
    ' Bwd IAT Std',

    ' Fwd Header Length',
    ' Bwd Header Length',

    ' Packet Length Mean',
    ' Packet Length Std',
    ' Packet Length Variance',

    ' PSH Flag Count',
    ' ACK Flag Count',

    'Subflow Fwd Packets',
    ' Subflow Fwd Bytes',
    ' Subflow Bwd Packets',
    ' Subflow Bwd Bytes',

    'Init_Win_bytes_forward',
    ' Init_Win_bytes_backward',

    ' act_data_pkt_fwd',
    ' min_seg_size_forward',

    'Active Mean',
    ' Active Std',

    'Idle Mean',
    ' Idle Std',

    ' Label'
]

# classification model name
TRAFFIC_CLASSIFICATION_MODEL_FILENAME: str = 'xgb_server_traffic_classifier.json'