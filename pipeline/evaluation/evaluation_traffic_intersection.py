"""

"""
import pandas as pd
import os
from utils.reports import compare_y_data_profiling
from utils.utils import generate_profiling_report


def traffic_intersection_report(data_filepath: str, results_folder_path: str, label_a: str, label_b: str):
    data_traffic_df = pd.read_csv(data_filepath)

    # filter by targets
    data_traffic_a_df = data_traffic_df.loc[data_traffic_df[' Label'] == label_a]
    data_traffic_b_df = data_traffic_df.loc[data_traffic_df[' Label'] == label_b]

    data_traffic_a_df = data_traffic_a_df.sample(20000)
    data_traffic_b_df = data_traffic_b_df.sample(20000)

    a_vs_b_filepath = os.path.join(results_folder_path, f"{label_a}_data_vs_{label_b}_data.html")
    compare_y_data_profiling(data_traffic_a_df, label_a, data_traffic_b_df, label_b,
                             a_vs_b_filepath)

    # remove columns
    data_traffic_a_df.pop(' Label')
    data_traffic_b_df.pop(' Label')

    # where all columns have the sama value
    all_columns = data_traffic_a_df.columns.tolist()
    intersection_df = pd.merge(data_traffic_a_df, data_traffic_b_df, on=all_columns, how='inner')

    # merge
    title = f'intersection between {label_a} and {label_b} data'
    a_vs_b_intersection_filepath = os.path.join(results_folder_path, f"{label_a}_data_vs_{label_b}_data_intersection.html")
    generate_profiling_report(title, a_vs_b_intersection_filepath, intersection_df)