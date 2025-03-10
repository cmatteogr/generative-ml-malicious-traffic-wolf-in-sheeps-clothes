"""
Utils
"""
import pandas as pd
from ydata_profiling import ProfileReport


def generate_profiling_report(title:str, report_filepath:str,  df: pd.DataFrame=None, data_filepath:str=None,
                              type_schema=None, minimal:bool=True):
    """
    Generate a profiling report for a dataframe
    :param title: report title
    :param report_filepath: report filepath
    :param df: report dataframe
    :param data_filepath: data filepath
    :param type_schema: type schema
    :param minimal: minimal profiling data to minimize data usage
    """
    # if df and data_filepath has a value then raise an Exception
    if df and data_filepath:
        raise Exception("Data should be defined from df or data_filepath, not both")

    # read data from filepath if exist
    if data_filepath:
        df = pd.read_csv(data_filepath)
    # generate data profiling report
    df_profile = ProfileReport(df, title=title, minimal=minimal, type_schema=type_schema)
    # export profiling report
    df_profile.to_file(report_filepath)


def compare_profiling_report(report_filepath:str, data_a:dict, data_b:dict, minimal:bool=True):
    """
    Compare two profiling reports
    :param report_filepath: report filepath
    :param data_a: data a
    :param data_b: data b
    :param minimal: minimal profiling data to minimize data usage
    """
    # read dfs
    a_df = pd.read_csv(data_a['filepath'])
    b_df = pd.read_csv(data_b['filepath'])

    # generate data profiling report
    a_df_profile = ProfileReport(a_df, title=data_a['title'], minimal=minimal, type_schema=data_a['type_schema'])
    b_df_profile = ProfileReport(b_df, title=data_b['title'], minimal=minimal, type_schema=data_a['type_schema'])

    # compare reports
    comparison_report = a_df_profile.compare(b_df_profile)
    # export profiling report
    comparison_report.to_file(report_filepath)
