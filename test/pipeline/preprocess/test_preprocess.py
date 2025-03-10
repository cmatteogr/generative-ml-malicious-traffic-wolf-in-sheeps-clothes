"""
Test case preprocess
"""
from pipeline.preprocess.preprocess import preprocessing
from utils.constants import RELEVANT_COLUMNS, VALID_TRAFFIC_TYPES, VALID_PORT_RANGE, VALID_PROTOCOL_VALUES


def test_preprocess():
    base_traffic_filepath = '/home/cesarealice/PycharmProjects/generative-ml-malicious-traffic-wolf-in-sheeps-clothes/data/Weekly-WorkingHours_report.csv'

    base_traffic_cleaned_filepath = preprocessing(base_traffic_filepath, relevant_column=RELEVANT_COLUMNS,
                                            valid_traffic_types=VALID_TRAFFIC_TYPES,
                                            valid_port_range=VALID_PORT_RANGE)
    print(base_traffic_cleaned_filepath)
    assert base_traffic_cleaned_filepath, "Filepath is empty"