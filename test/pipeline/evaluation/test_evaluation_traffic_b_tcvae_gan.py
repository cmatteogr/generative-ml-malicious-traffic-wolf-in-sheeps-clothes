"""
Test case evaluation
"""
from pipeline.evaluation.evaluation_traffic_b_tcvae_gan import evaluation


def test_evaluation():
    test_traffic_filepath = 'traffic_preprocessed_test_normalized.csv'

    model_filepath = 'xgb_server_traffic_classifier.json'
    label_encoder_model_filepath = 'label_encoder.pkl'
    results_folder_path = 'results'
    evaluation(model_filepath, test_traffic_filepath, results_folder_path, label_encoder_model_filepath)