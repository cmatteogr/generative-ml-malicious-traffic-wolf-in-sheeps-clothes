"""
Reports utils auxiliar
"""
from ydata_profiling import ProfileReport

def compare_y_data_profiling(a_df, a_report_title, b_df, b_report_title, comparison_report_filepath):
    # Use ydata-profiling to compare input and generated data
    a_data_report = ProfileReport(a_df, title=a_report_title)
    # generate report with generated data
    b_data_report = ProfileReport(b_df, title=b_report_title)
    # compare reports, original data and generated data
    comparison_report = a_data_report.compare(b_data_report)
    # save report
    comparison_report.to_file(comparison_report_filepath)