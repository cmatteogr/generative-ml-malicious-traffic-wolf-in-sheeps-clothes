"""

"""
from utils.utils import generate_profiling_report, compare_profiling_report


def test_generate_profiling_report():
    title= "Report Profiling"
    report_name = 'traffic_preprocessed_train'
    report_filepath = f"{report_name}.html"
    data_filepath = f"/home/cesarealice/PycharmProjects/generative-ml-malicious-traffic-wolf-in-sheeps-clothes/pipeline/preprocess/{report_name}.csv"
    type_schema = {'Label': "categorical"}

    generate_profiling_report(report_filepath=report_filepath, title=title, data_filepath=data_filepath,
                              type_schema=type_schema, minimal=True)


def test_compare_profiling_reports():
    report_name = 'compare_profiling_reports'
    report_filepath = f"{report_name}.html"

    type_schema = {'Label': "categorical"}

    data_a = {
        'filepath': f"/home/cesarealice/PycharmProjects/generative-ml-malicious-traffic-wolf-in-sheeps-clothes/test/pipeline/preprocess/traffic_preprocessed.csv",
        'title': 'base dataset',
        'type_schema': type_schema,
    }

    data_b = {
        'filepath': f"/home/cesarealice/PycharmProjects/generative-ml-malicious-traffic-wolf-in-sheeps-clothes/test/pipeline/preprocess/traffic_preprocessed_train.csv",
        'title': 'preprocessed dataset',
        'type_schema': type_schema,
    }

    compare_profiling_report(report_filepath=report_filepath, data_a=data_a, data_b=data_b, minimal=False)