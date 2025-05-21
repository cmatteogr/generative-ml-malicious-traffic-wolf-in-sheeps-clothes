"""
Test case evaluation
"""
from pipeline.evaluation.evaluation_traffic_discriminator import evaluation


def test_evaluation():
    test_traffic_filepath = '/home/cesarealice/PycharmProjects/generative-ml-malicious-traffic-wolf-in-sheeps-clothes/test/pipeline/preprocess/traffic_preprocessed_test.csv'

    model_filepath = '/home/cesarealice/PycharmProjects/generative-ml-malicious-traffic-wolf-in-sheeps-clothes/test/pipeline/train/xgb_server_traffic_classifier.json'
    results_folder_path = '/home/cesarealice/PycharmProjects/generative-ml-malicious-traffic-wolf-in-sheeps-clothes/results'
    evaluation(model_filepath, test_traffic_filepath, results_folder_path)