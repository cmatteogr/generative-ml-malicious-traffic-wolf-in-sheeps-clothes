"""

"""
from pipeline.evaluation.evaluation_traffic_intersection import traffic_intersection_report


def test_traffic_intersection_report():
    data_filepath = 'Weekly-WorkingHours_report.csv'
    results_folder_path ='./results'
    label_a = 'BENIGN'
    label_b = "PortScan"
    traffic_intersection_report(data_filepath, results_folder_path, label_a, label_b)