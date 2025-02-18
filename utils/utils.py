"""

"""
import pandas as pd
from ydata_profiling import ProfileReport


def generate_profiling_report(title:str, report_filepath:str,  df: pd.DataFrame=None, data_filepath:str=None,
                              type_schema=None, minimal:bool=True):
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