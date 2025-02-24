"""

"""
from utils.utils import generate_profiling_report, compare_profiling_report


def test_generate_profiling_report():
    title= "Report Profiling"
    report_name = 'traffic_preprocessed'
    report_filepath = f"{report_name}.html"
    data_filepath = f"/home/cesarealice/PycharmProjects/generative-ml-malicious-traffic-wolf-in-sheeps-clothes/test/pipeline/preprocess/{report_name}.csv"
    type_schema = {'Label': "categorical"}

    generate_profiling_report(report_filepath=report_filepath, title=title, data_filepath=data_filepath,
                              type_schema=type_schema, minimal=False)


def test_compare_profiling_reports():
    title= "Report Profiling"
    report_name = 'compare_profiling_reports'
    report_filepath = f"{report_name}.html"
    data_filepath_a = f"/home/cesarealice/PycharmProjects/generative-ml-malicious-traffic-wolf-in-sheeps-clothes/test/pipeline/preprocess/traffic_preprocessed.csv"
    data_filepath_b = f"/home/cesarealice/PycharmProjects/generative-ml-malicious-traffic-wolf-in-sheeps-clothes/test/pipeline/preprocess/traffic_preprocessed_train.csv"
    type_schema = {'Label': "categorical"}

    compare_profiling_report(report_filepath=report_filepath, title=title, data_filepath_a=data_filepath_a,
                             data_filepath_b=data_filepath_b, type_schema=type_schema, minimal=False)