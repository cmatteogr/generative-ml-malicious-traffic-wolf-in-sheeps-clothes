"""

"""
from utils.utils import generate_profiling_report
def test_generate_profiling_report():
    title= "Report Profiling"
    report_filepath = "traffic_preprocessed.html"
    data_filepath = "/home/cesarealice/PycharmProjects/generative-ml-malicious-traffic-wolf-in-sheeps-clothes/test/pipeline/preprocess/traffic_preprocessed.csv"
    type_schema = {'Label': "categorical"}

    generate_profiling_report(report_filepath=report_filepath, title=title, data_filepath=data_filepath,
                              type_schema=type_schema)