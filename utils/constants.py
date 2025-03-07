
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
    # ' Protocol', # Non informative after filter by valid values
    ' Flow Duration',

    ' Total Fwd Packets',
    ' Total Backward Packets',

    'Total Length of Fwd Packets',
    ' Total Length of Bwd Packets',

    #' Fwd Packet Length Max',
    #' Fwd Packet Length Min',
    ' Fwd Packet Length Mean',
    ' Fwd Packet Length Std',
    #'Bwd Packet Length Max',
    #' Bwd Packet Length Min',
    ' Bwd Packet Length Mean',
    ' Bwd Packet Length Std',

    #'Flow Bytes/s',
    #' Flow Packets/s',

    ' Flow IAT Mean',
    ' Flow IAT Std',
    #' Flow IAT Max',
    #' Flow IAT Min',

    'Fwd IAT Total',
    ' Fwd IAT Mean',
    ' Fwd IAT Std',
    #' Fwd IAT Max',
    #' Fwd IAT Min',
    'Bwd IAT Total',
    ' Bwd IAT Mean',
    ' Bwd IAT Std',
    #' Bwd IAT Max',
    #' Bwd IAT Min',

    ' Fwd Header Length',
    ' Bwd Header Length',

    #'Fwd Packets/s',
    #' Bwd Packets/s',
    #' Min Packet Length',
    #' Max Packet Length',
    ' Packet Length Mean',
    ' Packet Length Std',
    ' Packet Length Variance',

    ' PSH Flag Count',
    ' ACK Flag Count',

    #' Down/Up Ratio',

    #' Average Packet Size',
    #' Avg Fwd Segment Size',
    #' Avg Bwd Segment Size',

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
    #' Active Max',
    #' Active Min',

    'Idle Mean',
    ' Idle Std',
    #' Idle Max',
    #' Idle Min',
    ' Label'
]